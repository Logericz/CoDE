"""缓存上的固定2×2特征对照；g标签只算一次，f对照共用同一标签。

g负责选择退出E/继续T；f负责判断是否值得延长观测。改变g会改变f的
预测目标，所以跨g只比较相同E/T结局上的决策损失，不直接比较MAE。
所有数值原语、题级分折与超参数均来自冻结分析器。
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import numpy as np

from . import entropy_value_analysis as original
from .entropy_compact_features import compact_views

KINDS = ("original", "compact")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def view(row, role, kind):
    if kind not in KINDS or role not in ("f_B", "f_H", "g_short", "g_long"):
        raise ValueError("Unknown feature role or fixed feature family")
    if kind == "compact":
        return compact_views(row)[role]
    return row[{"f_B": "B", "f_H": "H", "g_short": "H", "g_long": "long_H"}[role]]


def matrix(rows, role, kind):
    names = sorted(view(rows[0], role, kind))
    return np.asarray([[view(row, role, kind)[key] for key in names] for row in rows], dtype=float)


def fit_actions(rows, kind, penalty):
    target = np.asarray([e - t for e, t in (original.action_losses(r, penalty) for r in rows)])
    return tuple(original.fit_ridge(matrix(rows, role, kind), target, original.RIDGE_ALPHA)
                 for role in ("g_short", "g_long"))


def scored_targets(train, held, kind, penalty):
    short, long = fit_actions(train, kind, penalty)
    targets = original.observation_targets(held,
        short.predict(matrix(held, "g_short", kind)),
        long.predict(matrix(held, "g_long", kind)), penalty)
    constant_delta = float(np.mean([e - t for e, t in
        (original.action_losses(r, penalty) for r in train)]))
    return [{"row": row["row"], "question_id": row["question_id"],
             "candidate_index": row["candidate_index"], "constant_delta_ms": constant_delta, **target}
            for row, target in zip(held, targets)]


def build_target_cache(roster, rows, g_kind, penalty):
    """只允许训练题的结局参与拟合；测试结局仅用于兑现选定动作。"""
    outer, folds = original.grouped_folds(roster), []
    for fold in range(4):
        train_ids = [q for q in roster if outer[q] != fold]
        test_ids = [q for q in roster if outer[q] == fold]
        train = [r for r in rows if r["question_id"] in train_ids]
        held = [r for r in rows if r["question_id"] in test_ids and r["extension_eligible"]]
        item = {"outer_fold": fold, "train_question_ids": train_ids, "test_question_ids": test_ids,
                "train_paired_records": len(train), "test_extension_records": len(held),
                "inner_folds": [], "training_targets": [], "held_out_targets": []}
        folds.append(item)
        if not held or not original._enough_training(train):
            item["status"] = "not_estimable:no_test_extensions_or_training_questions"
            continue
        inner = original.grouped_folds(train_ids, seed=original.FOLD_SEED + 100 + fold)
        for k in range(4):
            g_train = [r for r in train if inner[r["question_id"]] != k]
            g_held = [r for r in train if inner[r["question_id"]] == k and r["extension_eligible"]]
            audit = {"inner_fold": k, "g_train_question_ids": sorted({r["question_id"] for r in g_train}),
                     "z_question_ids": sorted({r["question_id"] for r in g_held})}
            item["inner_folds"].append(audit)
            if not g_held or not original._enough_training(g_train):
                audit["status"] = "not_estimable"
                continue
            item["training_targets"].extend(scored_targets(g_train, g_held, g_kind, penalty))
            audit["status"] = "cross_fitted"
        if len({r["question_id"] for r in item["training_targets"]}) < original.MIN_TRAIN_QUESTIONS:
            item["status"] = "not_estimable:insufficient_cross_fitted_questions"
            continue
        item["held_out_targets"] = scored_targets(train, held, g_kind, penalty)
        item.update(status="evaluated", f_training_records=len(item["training_targets"]),
                    f_training_questions=len({r["question_id"] for r in item["training_targets"]}))
    return {"g_features": g_kind, "lambda_error_ms": penalty,
            "outer_question_folds": outer, "folds": folds}


def predict_values(rows, cache, f_kind):
    """f只读共享cache；此函数没有动作选择器拟合调用。"""
    by_row = {r["row"]: r for r in rows}
    predictions = []
    for fold in cache["folds"]:
        if fold["status"] != "evaluated":
            continue
        train = [by_row[t["row"]] for t in fold["training_targets"]]
        held = [by_row[t["row"]] for t in fold["held_out_targets"]]
        targets = np.asarray([t["z_ms"] for t in fold["training_targets"]])
        scores = {name: original.fit_ridge(matrix(train, "f_" + name, f_kind), targets,
                                          original.RIDGE_ALPHA).predict(matrix(held, "f_" + name, f_kind))
                  for name in ("B", "H")}
        for index, (row, target) in enumerate(zip(held, fold["held_out_targets"])):
            predictions.append({**target, "outer_fold": fold["outer_fold"],
                "B_prediction_ms": float(scores["B"][index]), "H_prediction_ms": float(scores["H"][index]),
                "constant_prediction_ms": float(targets.mean()), "extra_probe_ms": row["extra_probe_ms"]})
    return predictions


def decision_metrics(rows, targets, choices, observed):
    """在同一组实测结局上计损失；不把gold最优动作交给预测器。"""
    if len(targets) != len(choices) or len(targets) != len(observed):
        raise ValueError("Decision outcomes and choices must align")
    by_row = {r["row"]: r for r in rows}
    if len({t["row"] for t in targets}) != len(targets):
        raise ValueError("Duplicate held-out decision")
    errors, remaining, loss, extra, regret = 0, 0.0, 0.0, 0.0, 0.0
    optimal, avoidable = 0, 0
    for target, action, use_long in zip(targets, choices, observed):
        row = by_row[target["row"]]
        if (action not in ("E", "T") or type(use_long) is not bool
                or row["question_id"] != target["question_id"]
                or row["candidate_index"] != target["candidate_index"]):
            raise ValueError("Decision identity or action mismatch")
        errors += row[action + "_error"]
        remaining += row[action + "_ms"]
        action_loss = target["loss_" + action + "_ms"]
        cost = row["extra_probe_ms"] if use_long else 0.0
        loss += action_loss + cost
        extra += cost
        regret += action_loss + cost - min(target["loss_E_ms"], target["loss_T_ms"])
        optimal += action_loss == min(target["loss_E_ms"], target["loss_T_ms"])
        avoidable += row[action + "_error"] > min(row["E_error"], row["T_error"])
    n = len(targets)
    return {"status": "evaluated" if n else "not_estimable", "records": n,
            "correct": n - errors if n else None, "incorrect": errors if n else None,
            "optimal_action_agreements": optimal if n else None, "avoidable_answer_errors": avoidable if n else None,
            "action_counts": dict(Counter(choices)), "additional_probes": sum(observed),
            "remaining_ms": remaining if n else None, "extra_probe_ms": extra if n else None,
            "total_loss_ms": loss if n else None, "regret_to_per_pair_oracle_ms": regret if n else None,
            "scope": "paired_remaining_cost_plus_error_penalty_and_selected_extension; common_paid_short_probe_excluded"}


def selector_metrics(rows, cache):
    targets = [t for fold in cache["folds"] for t in fold["held_out_targets"]]
    plans = {"short": [t["short_action"] for t in targets],
             "long": [t["long_action"] for t in targets],
             "always_E": ["E"] * len(targets), "always_T": ["T"] * len(targets),
             "training_mean_delta": ["T" if t["constant_delta_ms"] > 0 else "E" for t in targets],
             "oracle_diagnostic_only": ["T" if t["loss_T_ms"] < t["loss_E_ms"] else "E" for t in targets]}
    result = {name: decision_metrics(rows, targets, choices, [name == "long"] * len(targets))
              for name, choices in plans.items()}
    result["transitions"] = dict(Counter(t["short_action"] + "->" + t["long_action"] for t in targets))
    result["positive_z_records"] = sum(t["z_ms"] > 0 for t in targets)
    return result


def evaluate_cell(rows, cache, f_kind, bootstrap_replicates):
    before = digest(cache)
    predictions = predict_values(rows, cache, f_kind)
    policies = {}
    for name in ("B", "H", "constant"):
        selected = [r[name + "_prediction_ms"] > 0 for r in predictions]
        choices = [r["long_action"] if use else r["short_action"] for r, use in zip(predictions, selected)]
        policies[name] = decision_metrics(rows, predictions, choices, selected)
    if before != digest(cache):
        raise AssertionError("A value predictor mutated the shared target cache")
    return {"g_features": cache["g_features"], "f_features": f_kind,
            "lambda_error_ms": cache["lambda_error_ms"], "target_cache_sha256": before,
            "summary": original.summarize_predictions(predictions),
            "bootstrap": original.question_bootstrap(predictions, bootstrap_replicates),
            "pipeline_policies": policies, "held_out_predictions": predictions}


def check_reference(cells, reference):
    """旧g+旧f必须重现保存外折记录；仅容许浮点舍入差异。"""
    penalties = [row["lambda_error_ms"] for row in reference["results"]]
    if len(penalties) != len(original.ERROR_COSTS_MS) or set(penalties) != set(original.ERROR_COSTS_MS):
        raise ValueError("Reference must contain all three unique fixed error costs")
    checks, largest = 0, 0.0
    for saved in reference["results"]:
        cell = next(c for c in cells if c["g_features"] == c["f_features"] == "original"
                    and c["lambda_error_ms"] == saved["lambda_error_ms"])
        current = cell["held_out_predictions"]
        if len(current) != len(saved["held_out_predictions"]):
            raise ValueError("Original result coverage changed")
        for old, new in zip(saved["held_out_predictions"], current):
            for key, expected in old.items():
                if isinstance(expected, (float, int)) and not isinstance(expected, bool):
                    delta = abs(new[key] - expected)
                    largest = max(largest, delta)
                    if not math.isclose(new[key], expected, rel_tol=1e-10, abs_tol=1e-6):
                        raise ValueError("Original prediction differs: " + key)
                elif new[key] != expected:
                    raise ValueError("Original prediction identity differs: " + key)
                checks += 1
    return {"passed": True, "field_checks": checks, "max_absolute_numeric_difference": largest}


def analyze_ablation(document, reference=None, bootstrap_replicates=2000):
    roster, rows, coverage = original.prepare_records(document)
    caches, cells, selectors = [], [], []
    for g_kind in KINDS:
        for penalty in original.ERROR_COSTS_MS:
            cache = build_target_cache(roster, rows, g_kind, penalty)
            caches.append(cache)
            selectors.append({"g_features": g_kind, "lambda_error_ms": penalty,
                              "metrics": selector_metrics(rows, cache)})
            for f_kind in KINDS:
                cells.append(evaluate_cell(rows, cache, f_kind, bootstrap_replicates))
    return {"schema_version": "entropy-compact-ablation-v1", "role": "post_hoc_cached_development_only",
            "coverage": coverage, "target_caches": caches, "cells": cells, "selectors": selectors,
            "reference_reproduction": check_reference(cells, reference) if reference else None,
            "feature_names": {kind: {role: sorted(view(rows[0], role, kind)) if rows else []
                for role in ("f_B", "f_H", "g_short", "g_long")} for kind in KINDS},
            "limitations": ["No new samples; the cache already informed this feature design.",
                "Targets differ across g families; do not compare their MAE as a common prediction task.",
                "Short/long action models remain separately fitted; dimension reduction does not remove that confound.",
                "All policy losses are cached paired evaluations, not measured online acceleration."]}
