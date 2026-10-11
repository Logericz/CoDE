"""同一短观测下的固定信号消融；直接预测动作损失差，不训练追加价值f。

特征选择单列在entropy_short_features，数值拟合复用冻结分析器。
所有融合规则复用C/H的同一份留出分数，答案只用于训练目标与兑现评价。
"""
from __future__ import annotations

from collections import Counter
import math
import numpy as np

from . import entropy_value_analysis as original
from .entropy_short_features import FEATURE_SPEC, feature_views

LEARNED = ("Z", "C", "H", "CH")
METHODS = (*LEARNED, "trainmean", "always_E", "always_T", "late_avg", "late_agree", "late_either")
COMPARISONS = (("CH", "C"), ("CH", "H"), ("C", "Z"), ("H", "Z"),
               ("late_avg", "C"), ("late_agree", "C"), ("late_either", "C"), ("CH", "late_avg"))
BOOTSTRAP_SEED = 20261911


def choose(score):
    if not math.isfinite(float(score)):
        raise ValueError("Decision score must be finite")
    return "T" if score > 0 else "E"


def combine_scores(scores):
    """零分选E；逻辑融合在动作层运行，不伪造可校准概率。"""
    actions = {name: choose(scores[name]) for name in (*LEARNED, "trainmean")}
    c, h = actions["C"], actions["H"]
    return {**actions, "always_E": "E", "always_T": "T",
            "late_avg": choose((scores["C"] + scores["H"]) / 2),
            "late_agree": "E" if c == h == "E" else "T",
            "late_either": "E" if "E" in (c, h) else "T"}


def matrix(rows, family):
    names = sorted(FEATURE_SPEC[family])
    return np.asarray([[feature_views(row)[family][key] for key in names] for row in rows], dtype=float)


def prepare(document):
    roster, rows, coverage = original.prepare_records(document)
    for row in rows:
        raw = document["records"][row["row"]]
        # 用短阶段可见状态定义主样本；不要复用含长观测信息的extension_eligible。
        row["primary_eligible"] = (len(raw["short"]["max_probabilities"]) == 21
                                   and not raw["short"]["ended_with_think"])
        value = raw.get("short_probe_ms")
        row["short_probe_ms"] = float(value) if original.finite_number(value) and value >= 0 else None
    return roster, rows, coverage


def predict_fold(train, held, penalty):
    """一折内每个信号组拟合一次；held答案完全不参与scores或actions。"""
    target = np.asarray([e - t for e, t in (original.action_losses(r, penalty) for r in train)])
    fitted = {name: original.fit_ridge(matrix(train, name), target, original.RIDGE_ALPHA)
              for name in LEARNED}
    scores = {name: model.predict(matrix(held, name)) for name, model in fitted.items()}
    output = []
    for i, row in enumerate(held):
        score = {name: float(values[i]) for name, values in scores.items()}
        score["trainmean"] = float(target.mean())
        score["late_avg"] = (score["C"] + score["H"]) / 2
        loss_e, loss_t = original.action_losses(row, penalty)
        output.append({key: row[key] for key in ("row", "question_id", "candidate_index",
            "primary_eligible", "short_probe_ms", "E_error", "T_error", "E_ms", "T_ms")} | {
                "scores": score, "actions": combine_scores(score),
                "loss_E_ms": loss_e, "loss_T_ms": loss_t, "delta_ms": loss_e - loss_t})
    return output


def selected_loss(record, method):
    return record["loss_" + record["actions"][method] + "_ms"]


def decision_summary(records, method):
    if not records:
        return {"status": "not_estimable", "records": 0}
    actions = [r["actions"][method] for r in records]
    errors = [r[a + "_error"] for r, a in zip(records, actions)]
    losses = [selected_loss(r, method) for r in records]
    questions = sorted({r["question_id"] for r in records})
    common = None if any(r["short_probe_ms"] is None for r in records) else sum(r["short_probe_ms"] for r in records)
    return {"status": "evaluated", "records": len(records), "questions": len(questions),
        "correct": len(records) - sum(errors), "incorrect": sum(errors), "action_counts": dict(Counter(actions)),
        "avoidable_answer_errors": sum(e > min(r["E_error"], r["T_error"]) for r, e in zip(records, errors)),
        "optimal_action_agreements": sum(loss == min(r["loss_E_ms"], r["loss_T_ms"]) for r, loss in zip(records, losses)),
        "remaining_ms": sum(r[a + "_ms"] for r, a in zip(records, actions)),
        "total_loss_ms": sum(losses), "mean_loss_ms": float(np.mean(losses)),
        "question_macro_mean_loss_ms": float(np.mean([np.mean([loss for r, loss in zip(records, losses)
            if r["question_id"] == q]) for q in questions])),
        "regret_to_pair_oracle_ms": sum(loss - min(r["loss_E_ms"], r["loss_T_ms"]) for r, loss in zip(records, losses)),
        "common_short_probe_ms": common,
        "cached_loss_plus_common_probe_ms": None if common is None else sum(losses) + common,
        "additional_probes": 0,
        "scope": "cached_remaining_time_plus_error_penalty; common_short_probe_separate; not_online_latency"}


def compare(records, candidate, reference, bootstrap_replicates):
    if not records:
        return {"status": "not_estimable", "records": 0}
    questions = sorted({r["question_id"] for r in records})
    changes, differences = [], []
    for r in records:
        before, after = r["actions"][reference], r["actions"][candidate]
        difference = selected_loss(r, candidate) - selected_loss(r, reference)
        differences.append(difference)
        if before != after:
            changes.append({"question_id": r["question_id"], "candidate_index": r["candidate_index"],
                "before": before, "after": after, "before_error": r[before + "_error"],
                "after_error": r[after + "_error"], "loss_difference_ms": difference})
    # 按题重采样整组点，不把同题的候选当成独立题。
    sums = np.asarray([sum(d for r, d in zip(records, differences) if r["question_id"] == q) for q in questions])
    counts = np.asarray([sum(r["question_id"] == q for r in records) for q in questions])
    interval = None
    if bootstrap_replicates and len(questions) > 1:
        indices = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(questions), (bootstrap_replicates, len(questions)))
        samples = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
        interval = [float(v) for v in np.quantile(samples, [.025, .975])]
    leave_one = [(sum(differences) - value) / (len(records) - count)
                 for value, count in zip(sums, counts) if len(records) > count]
    return {"status": "exploratory", "candidate": candidate, "reference": reference,
        "records": len(records), "questions": len(questions), "changed_decisions": len(changes),
        "correct_to_wrong": sum(c["before_error"] == 0 and c["after_error"] == 1 for c in changes),
        "wrong_to_correct": sum(c["before_error"] == 1 and c["after_error"] == 0 for c in changes),
        "same_correctness_less_cost": sum(c["before_error"] == c["after_error"] and c["loss_difference_ms"] < 0 for c in changes),
        "same_correctness_more_cost": sum(c["before_error"] == c["after_error"] and c["loss_difference_ms"] > 0 for c in changes),
        "total_loss_difference_ms": float(sum(differences)), "net_utility_ms": float(-sum(differences)),
        "mean_loss_difference_ms": float(np.mean(differences)),
        "question_macro_mean_loss_difference_ms": float(np.mean(sums / counts)),
        "conditional_question_bootstrap_mean_difference_95": interval,
        "leave_one_question_out_mean_difference_range": [float(min(leave_one)), float(max(leave_one))] if leave_one else None,
        "changes": changes}


def summarize(records, bootstrap_replicates):
    methods = {name: decision_summary(records, name) for name in METHODS}
    if not records:
        return {"status": "not_estimable", "methods": methods, "regression": {}, "comparisons": {}}
    regression = {name: original.regression_metrics([r["delta_ms"] for r in records],
                    [r["scores"][name] for r in records]) for name in (*LEARNED, "trainmean", "late_avg")}
    comparisons = {candidate + "_vs_" + reference: compare(records, candidate, reference, bootstrap_replicates)
                   for candidate, reference in COMPARISONS}
    oracle = sum(min(r["loss_E_ms"], r["loss_T_ms"]) for r in records)
    selection_oracle = sum(min(selected_loss(r, "C"), selected_loss(r, "H")) for r in records)
    conflicts = [{"question_id": r["question_id"], "candidate_index": r["candidate_index"],
                  "C": r["actions"]["C"], "H": r["actions"]["H"], "CH": r["actions"]["CH"],
                  "C_score_ms": r["scores"]["C"], "H_score_ms": r["scores"]["H"],
                  "loss_E_ms": r["loss_E_ms"], "loss_T_ms": r["loss_T_ms"]}
                 for r in records if r["actions"]["C"] != r["actions"]["H"]]
    return {"status": "exploratory", "methods": methods, "regression": regression,
        "comparisons": comparisons, "C_H_conflicts": conflicts,
        "oracle_diagnostics_only": {"pair_oracle_loss_ms": oracle,
            "C_H_selection_oracle_loss_ms": selection_oracle,
            "C_H_selection_headroom_over_C_ms": methods["C"]["total_loss_ms"] - selection_oracle,
            "accuracy_upper_bound_correct": sum(1 - min(r["E_error"], r["T_error"]) for r in records)}}


def check_reference(results, reference):
    caches = [c for c in reference["target_caches"] if c["g_features"] == "compact"]
    if len(caches) != 3 or {c["lambda_error_ms"] for c in caches} != set(original.ERROR_COSTS_MS):
        raise ValueError("Reference needs three unique compact selector caches")
    checks, largest = 0, 0.0
    for cache in caches:
        result = next(r for r in results if r["lambda_error_ms"] == cache["lambda_error_ms"])
        current = {r["row"]: r for r in result["predictions"] if r["primary_eligible"]}
        saved = [r for f in cache["folds"] for r in f["held_out_targets"]]
        if set(current) != {r["row"] for r in saved}:
            raise ValueError("Previous compact coverage differs from short-only primary cohort")
        for old in saved:
            new = current[old["row"]]
            for key in ("question_id", "candidate_index", "loss_E_ms", "loss_T_ms"):
                if new[key] != old[key]:
                    raise ValueError("Reference record differs: " + key)
                checks += 1
            if new["actions"]["CH"] != old["short_action"]:
                raise ValueError("CH changed previous compact short action")
            delta = abs(new["scores"]["CH"] - old["short_delta_ms"])
            if not math.isclose(new["scores"]["CH"], old["short_delta_ms"], rel_tol=1e-10, abs_tol=1e-6):
                raise ValueError("CH changed previous compact short score")
            largest = max(largest, delta)
            checks += 2
    return {"passed": True, "field_checks": checks, "score_checks": checks // 6,
            "max_absolute_score_difference_ms": largest}


def analyze(document, reference=None, bootstrap_replicates=2000):
    if type(bootstrap_replicates) is not int or bootstrap_replicates < 0:
        raise ValueError("Bootstrap count must be a nonnegative integer")
    roster, rows, coverage = prepare(document)
    folds = original.grouped_folds(roster)
    results = []
    for penalty in original.ERROR_COSTS_MS:
        predictions, audit = [], []
        for fold in range(4):
            train_ids = [q for q in roster if folds[q] != fold]
            test_ids = [q for q in roster if folds[q] == fold]
            train = [r for r in rows if r["question_id"] in train_ids]
            held = [r for r in rows if r["question_id"] in test_ids]
            item = {"outer_fold": fold, "train_question_ids": train_ids, "test_question_ids": test_ids,
                    "train_records": len(train), "test_records": len(held)}
            audit.append(item)
            if not held or not original._enough_training(train):
                item["status"] = "not_estimable"
                continue
            predictions.extend({**r, "outer_fold": fold} for r in predict_fold(train, held, penalty))
            item["status"] = "evaluated"
        scopes = {"primary": [r for r in predictions if r["primary_eligible"]], "all": predictions,
                  "other_short": [r for r in predictions if not r["primary_eligible"]]}
        results.append({"lambda_error_ms": penalty, "predictions": predictions, "fold_audit": audit,
                        "scopes": {name: summarize(items, bootstrap_replicates) for name, items in scopes.items()}})
    return {"schema_version": "entropy-short-fusion-v1", "role": "post_hoc_cached_development_only",
        "coverage": coverage, "primary_records": sum(r["primary_eligible"] for r in rows),
        "outer_question_folds": folds, "feature_names": FEATURE_SPEC, "results": results,
        "reference_reproduction": check_reference(results, reference) if reference is not None else None,
        "bootstrap": {"replicates": bootstrap_replicates, "seed": BOOTSTRAP_SEED,
                      "unit": "question", "refits": False},
        "limitations": ["Previously exposed algebra development cache, no independent confirmation.",
            "C/H are feature ablations, not complete CoDE-Stop or EntroCut methods.",
            "E/T are fixed local interventions, not complete adaptive rollouts.",
            "No long observation, no new generation; entropy/classifier deployment overhead unmeasured."]}
