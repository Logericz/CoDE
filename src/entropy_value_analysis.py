"""CPU analysis primitives for the bounded entropy/value development experiment.

This module does not load a language model.  Its regressors predict the value of
an observation; they never update the reasoning model's weights.  Unknown answer
grades are missing outcomes, not incorrect answers or zero-cost examples.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import math
from typing import Iterable, Sequence

import numpy as np

SCHEMA_VERSION = "entropy-value-v1"
FOLD_SEED = 20261010
RIDGE_ALPHA = 1.0
MIN_TRAIN_QUESTIONS = 4
ERROR_COSTS_MS = (60000.0, 30000.0, 120000.0)
HISTORY_FIELDS = ("valid_count", "previous_confidence", "previous_D", "candidate_gap", "token_gap")


def grouped_folds(question_ids: Iterable[str], n_splits: int = 4,
                  seed: int = 20261010) -> dict[str, int]:
    """Assign whole questions, including those without records, to fixed folds."""
    ids = sorted(set(question_ids))
    if n_splits < 2:
        raise ValueError("At least two folds are required")
    if not ids or any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("Question ids must be nonempty strings")
    order = np.random.default_rng(seed).permutation(len(ids))
    return {ids[int(index)]: rank % n_splits for rank, index in enumerate(order)}


def curve_summary(values: Sequence[float]) -> dict[str, float]:
    """Retain distribution and within-probe shape, not just a mean confidence.

    Equal-width bins use normalized token position, so short and long probes
    have the same feature schema.  Actual length is retained separately.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("A curve must be a nonempty finite one-dimensional array")
    differences = np.diff(x)
    positions = np.linspace(0.0, 1.0, len(x))
    centered = positions - positions.mean()
    slope = float(np.dot(centered, x - x.mean()) / np.dot(centered, centered)) if len(x) > 1 else 0.0
    result = {"mean": float(x.mean()), "std": float(x.std()),
              "min": float(x.min()), "max": float(x.max()),
              "q25": float(np.quantile(x, .25)), "median": float(np.median(x)),
              "q75": float(np.quantile(x, .75)), "first": float(x[0]),
              "last": float(x[-1]), "slope": slope,
              "mean_abs_change": float(np.abs(differences).mean()) if len(differences) else 0.0,
              "largest_increase": max(float(differences.max()), 0.0) if len(differences) else 0.0,
              "largest_decrease": max(float(-differences.min()), 0.0) if len(differences) else 0.0}
    # Empty bins on very short valid probes carry NaN; fold-local preprocessing
    # handles them and also creates a missing-value indicator.
    for index, piece in enumerate(np.array_split(x, 4)):
        result[f"quarter_{index + 1}_mean"] = float(piece.mean()) if len(piece) else float("nan")
    return result


@dataclass
class RidgeModel:
    """Small fixed-alpha ridge with all preprocessing learned on training data.

    Missing features are mean-imputed, with indicators.  No held-out value is
    consulted when fitting imputation, scaling, intercept, or coefficients.
    """
    feature_mean: np.ndarray
    expanded_mean: np.ndarray
    expanded_scale: np.ndarray
    coefficients: np.ndarray
    intercept: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.feature_mean):
            raise ValueError("Feature matrix does not match the fitted schema")
        missing = ~np.isfinite(x)
        filled = np.where(missing, self.feature_mean, x)
        expanded = np.concatenate([filled, missing.astype(float)], axis=1)
        normalized = (expanded - self.expanded_mean) / self.expanded_scale
        return self.intercept + normalized @ self.coefficients


def fit_ridge(features: np.ndarray, targets: np.ndarray,
              alpha: float = 1.0) -> RidgeModel:
    """Fit a fixed model; selecting alpha on experimental outcomes is forbidden."""
    x, y = np.asarray(features, dtype=float), np.asarray(targets, dtype=float)
    if x.ndim != 2 or not len(x) or y.shape != (len(x),) or not np.isfinite(y).all():
        raise ValueError("Training needs aligned rows and finite observed targets")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Ridge alpha must be finite and positive")
    valid = np.isfinite(x)
    counts = valid.sum(axis=0)
    means = np.divide(np.where(valid, x, 0.0).sum(axis=0), counts,
                      out=np.zeros(x.shape[1]), where=counts > 0)
    expanded = np.concatenate([np.where(valid, x, means), (~valid).astype(float)], axis=1)
    expanded_mean = expanded.mean(axis=0)
    scale = expanded.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    z = (expanded - expanded_mean) / scale
    intercept = float(y.mean())
    # The dual system is small when there are many curve summaries but few
    # questions.  It is algebraically the same fixed-alpha ridge regression.
    coefficients = z.T @ np.linalg.solve(z @ z.T + alpha * np.eye(len(z)), y - intercept)
    return RidgeModel(means, expanded_mean, scale, coefficients, intercept)


def regression_metrics(targets: Sequence[float], predictions: Sequence[float]) -> dict[str, float]:
    y, p = np.asarray(targets, dtype=float), np.asarray(predictions, dtype=float)
    if y.ndim != 1 or not len(y) or y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Metrics require paired finite targets and predictions")
    return {"mae_ms": float(np.abs(p - y).mean()),
            "mse_ms2": float(np.square(p - y).mean()),
            "mean_prediction_ms": float(p.mean()), "mean_target_ms": float(y.mean())}


def finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _optional_number(value: object) -> float:
    return float(value) if finite_number(value) else float("nan")


def _probe_features(probe: dict, include_entropy: bool) -> dict[str, float]:
    probabilities = np.asarray(probe["max_probabilities"], dtype=float)
    features = {"confidence": float(probe["confidence"]),
                "probe_tokens": float(len(probabilities)),
                "ended_with_think": float(probe["ended_with_think"])}
    features.update({"prob_" + key: value for key, value in curve_summary(probabilities).items()})
    logs = np.log(np.maximum(probabilities, 1e-12))
    features.update(prob_mean_log=float(logs.mean()), prob_std_log=float(logs.std()))
    if include_entropy:
        entropy = np.asarray(probe["entropies_nats"], dtype=float)
        # Entropy uses the collector's fp32 distribution; the preserved legacy
        # maximum probabilities may be BF16.  Do not derive tail entropy by
        # subtracting h2(p): those values need not describe one distribution.
        features.update({"entropy_" + key: value for key, value in curve_summary(entropy).items()})
    return features


def feature_views(record: dict) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Return short B, short H, and long-observation E/T views.

    The long E/T model retains *all* short H features before adding long probe
    features.  Both E/T models may use entropy; only the value-predictor B arm
    is entropy-free.  Neither value predictor receives an E/T model score.
    """
    shared = {"candidate_index": float(record["candidate_index"]),
              "prefix_tokens": _optional_number(record.get("prefix_tokens")),
              "D_short": float(record["short"]["D"])}
    history = record.get("history", {})
    shared.update({"history_" + name: _optional_number(history.get(name)) for name in HISTORY_FIELDS})
    base = dict(shared)
    base.update({"short_" + key: value for key, value in _probe_features(record["short"], False).items()})
    entropy = dict(shared)
    entropy.update({"short_" + key: value for key, value in _probe_features(record["short"], True).items()})
    long = dict(entropy)
    long.update({"long_" + key: value for key, value in _probe_features(record["long"], True).items()})
    # Extended observations do not get inserted into the old D statistic.
    return base, entropy, long


def _validate_probe(probe: object, cap: int) -> bool:
    if not isinstance(probe, dict) or not finite_number(probe.get("confidence")):
        return False
    if not 0 <= probe["confidence"] <= 1 or type(probe.get("ended_with_think")) is not bool:
        return False
    p, h = probe.get("max_probabilities"), probe.get("entropies_nats")
    if not isinstance(p, list) or not isinstance(h, list) or not 1 <= len(p) <= cap or len(p) != len(h):
        return False
    return (all(finite_number(v) and 0 <= v <= 1 for v in p)
            and all(finite_number(v) and v >= 0 for v in h))


def prepare_records(document: dict) -> tuple[list[str], list[dict], dict]:
    """Validate outcome availability without turning unknown grades into errors."""
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Expected schema_version={SCHEMA_VERSION}")
    roster = document.get("question_ids")
    if (not isinstance(roster, list) or not roster or any(not isinstance(q, str) or not q for q in roster)
            or len(set(roster)) != len(roster)):
        raise ValueError("question_ids must contain each planned question exactly once")
    if not isinstance(document.get("records"), list):
        raise ValueError("records must be an array")
    kept, excluded, seen, counts, statuses = [], [], set(), Counter(), Counter()
    for number, record in enumerate(document["records"]):
        if not isinstance(record, dict) or record.get("question_id") not in roster:
            raise ValueError("Every record must identify a planned question")
        qid, status = record["question_id"], record.get("status")
        statuses[str(status)] += 1
        identity = {"row": number, "question_id": qid, "candidate_index": record.get("candidate_index")}
        if status != "ok":
            excluded.append({**identity, "reason": "record_status:" + str(status)})
            continue
        index = record.get("candidate_index")
        if type(index) is not int or not 1 <= index <= 3:
            raise ValueError("Only the first three predeclared candidate indices are allowed")
        key = (qid, index)
        if key in seen:
            raise ValueError("Duplicate question/candidate record")
        seen.add(key)
        counts[qid] += 1
        if counts[qid] > 3:
            raise ValueError("At most three predeclared candidates per question are allowed")
        if not _validate_probe(record.get("short"), 21) or not _validate_probe(record.get("long"), 42):
            excluded.append({**identity, "reason": "invalid_probe_features"})
            continue
        short, long = record["short"], record["long"]
        if not finite_number(short.get("D")) or short["D"] < 0:
            excluded.append({**identity, "reason": "invalid_short_D"})
            continue
        nshort, nlong = len(short["max_probabilities"]), len(long["max_probabilities"])
        if nlong < nshort or any(short[field] != long[field][:nshort]
                                for field in ("max_probabilities", "entropies_nats")):
            raise ValueError("Long observation must extend the exact short probability/entropy prefix")
        actions = record.get("actions", {})
        valid_actions = all(isinstance(actions.get(a), dict)
                            and finite_number(actions[a].get("remaining_ms"))
                            and actions[a]["remaining_ms"] >= 0 for a in ("E", "T"))
        if not valid_actions or not finite_number(record.get("extra_probe_ms")) or record["extra_probe_ms"] < 0:
            excluded.append({**identity, "reason": "unknown_or_invalid_cost"})
            continue
        if any(type(actions[a].get("error")) is not int or actions[a]["error"] not in (0, 1) for a in ("E", "T")):
            excluded.append({**identity, "reason": "unknown_or_invalid_answer_grade"})
            continue
        base, entropy, extended = feature_views(record)
        eligible = not short["ended_with_think"] and nlong > nshort
        kept.append({**identity, "B": base, "H": entropy, "long_H": extended,
                     "E_ms": float(actions["E"]["remaining_ms"]),
                     "T_ms": float(actions["T"]["remaining_ms"]),
                     "E_error": actions["E"]["error"], "T_error": actions["T"]["error"],
                     "extra_probe_ms": float(record["extra_probe_ms"]), "extension_eligible": eligible})
    coverage = {"planned_questions": len(roster), "input_records": len(document["records"]),
                "valid_paired_records": len(kept), "valid_paired_questions": len({r["question_id"] for r in kept}),
                "extension_eligible_records": sum(r["extension_eligible"] for r in kept),
                "status_counts": dict(statuses), "excluded_records": excluded,
                "excluded_reason_counts": dict(Counter(r["reason"] for r in excluded)),
                "planned_questions_without_valid_pairs": [q for q in roster if not any(r["question_id"] == q for r in kept)],
                "valid_pairs_without_extension": [{"question_id": r["question_id"], "candidate_index": r["candidate_index"]}
                                                  for r in kept if not r["extension_eligible"]]}
    return roster, kept, coverage


def _matrix(rows: list[dict], view: str) -> np.ndarray:
    names = sorted(rows[0][view])
    return np.asarray([[row[view][name] for name in names] for row in rows], dtype=float)


def action_losses(row: dict, lambda_error_ms: float) -> tuple[float, float]:
    return (row["E_ms"] + lambda_error_ms * row["E_error"],
            row["T_ms"] + lambda_error_ms * row["T_error"])


def _fit_action_models(rows: list[dict], lambda_error_ms: float) -> tuple[RidgeModel, RidgeModel]:
    # Positive L_E-L_T favors T.  These are utility regressions, not calibrated
    # probabilities, and do not see the held-out question's answer labels.
    target = np.asarray([action_losses(row, lambda_error_ms)[0] - action_losses(row, lambda_error_ms)[1]
                         for row in rows])
    return fit_ridge(_matrix(rows, "H"), target, RIDGE_ALPHA), fit_ridge(_matrix(rows, "long_H"), target, RIDGE_ALPHA)


def observation_targets(rows: list[dict], short_delta: Sequence[float], long_delta: Sequence[float],
                        lambda_error_ms: float) -> list[dict]:
    """Evaluate decisions selected by predictions, never by the best gold loss."""
    if len(rows) != len(short_delta) or len(rows) != len(long_delta):
        raise ValueError("Action scores must align with held-out records")
    result = []
    for row, before, after in zip(rows, short_delta, long_delta):
        if not math.isfinite(float(before)) or not math.isfinite(float(after)):
            raise ValueError("Nonfinite E/T score")
        loss_e, loss_t = action_losses(row, lambda_error_ms)
        short_action, long_action = ("T" if before > 0 else "E"), ("T" if after > 0 else "E")
        losses = {"E": loss_e, "T": loss_t}
        result.append({"z_ms": losses[short_action] - losses[long_action] - row["extra_probe_ms"],
                       "short_action": short_action, "long_action": long_action,
                       "short_delta_ms": float(before), "long_delta_ms": float(after),
                       "loss_E_ms": loss_e, "loss_T_ms": loss_t})
    return result


def _enough_training(rows: list[dict]) -> bool:
    return len({row["question_id"] for row in rows}) >= MIN_TRAIN_QUESTIONS


def nested_predictions(roster: list[str], rows: list[dict], lambda_error_ms: float) -> tuple[list[dict], list[dict]]:
    """Outer question test folds; inner question cross-fitting creates f labels.

    The two f arms share exactly the same inner and outer labels.  An outer test
    question is excluded from g fitting, f fitting, imputation and scaling.
    Within an outer training split, inner out-of-fold g decisions create z;
    f is then fitted on these cross-fitted z values rather than in-sample g wins.
    """
    outer = grouped_folds(roster)
    predictions, audits = [], []
    for fold in range(4):
        train_ids = [q for q in roster if outer[q] != fold]
        test_ids = [q for q in roster if outer[q] == fold]
        train = [r for r in rows if r["question_id"] in train_ids]
        test = [r for r in rows if r["question_id"] in test_ids and r["extension_eligible"]]
        audit = {"outer_fold": fold, "train_question_ids": train_ids, "test_question_ids": test_ids,
                 "train_paired_records": len(train), "test_extension_records": len(test), "inner_folds": []}
        audits.append(audit)
        if not test or not _enough_training(train):
            audit["status"] = "not_estimable:no_test_extensions_or_training_questions"
            continue
        inner = grouped_folds(train_ids, seed=FOLD_SEED + 100 + fold)
        f_rows, f_targets = [], []
        for inner_fold in range(4):
            g_train = [r for r in train if inner[r["question_id"]] != inner_fold]
            g_test = [r for r in train if inner[r["question_id"]] == inner_fold and r["extension_eligible"]]
            item = {"inner_fold": inner_fold,
                    "g_train_question_ids": sorted({r["question_id"] for r in g_train}),
                    "z_question_ids": sorted({r["question_id"] for r in g_test})}
            audit["inner_folds"].append(item)
            if not g_test or not _enough_training(g_train):
                item["status"] = "not_estimable"
                continue
            short_model, long_model = _fit_action_models(g_train, lambda_error_ms)
            targets = observation_targets(g_test, short_model.predict(_matrix(g_test, "H")),
                                          long_model.predict(_matrix(g_test, "long_H")), lambda_error_ms)
            f_rows.extend(g_test)
            f_targets.extend(t["z_ms"] for t in targets)
            item["status"] = "cross_fitted"
        if not _enough_training(f_rows):
            audit["status"] = "not_estimable:insufficient_cross_fitted_questions"
            continue
        targets_array = np.asarray(f_targets)
        base_model = fit_ridge(_matrix(f_rows, "B"), targets_array, RIDGE_ALPHA)
        entropy_model = fit_ridge(_matrix(f_rows, "H"), targets_array, RIDGE_ALPHA)
        short_model, long_model = _fit_action_models(train, lambda_error_ms)
        held_targets = observation_targets(test, short_model.predict(_matrix(test, "H")),
                                          long_model.predict(_matrix(test, "long_H")), lambda_error_ms)
        base_predictions = base_model.predict(_matrix(test, "B"))
        entropy_predictions = entropy_model.predict(_matrix(test, "H"))
        constant = float(targets_array.mean())
        for row, target, baseline, entropy in zip(test, held_targets, base_predictions, entropy_predictions):
            predictions.append({"question_id": row["question_id"], "candidate_index": row["candidate_index"],
                                "outer_fold": fold, **target, "B_prediction_ms": float(baseline),
                                "H_prediction_ms": float(entropy), "constant_prediction_ms": constant,
                                "extra_probe_ms": row["extra_probe_ms"]})
        audit.update(status="evaluated", f_training_records=len(f_rows),
                     f_training_questions=len({r["question_id"] for r in f_rows}), constant_prediction_ms=constant)
    return predictions, audits


def _policy_summary(rows: list[dict], model: str) -> dict:
    selected = [r for r in rows if r[model + "_prediction_ms"] > 0]
    return {"selected_records": len(selected), "selected_questions": len({r["question_id"] for r in selected}),
            "realized_net_utility_sum_ms": float(sum(r["z_ms"] for r in selected)),
            "mean_net_utility_per_opportunity_ms": float(sum(r["z_ms"] for r in selected) / len(rows)),
            "positive_realized_records": sum(r["z_ms"] > 0 for r in selected),
            "negative_realized_records": sum(r["z_ms"] < 0 for r in selected)}


def matched_budget(rows: list[dict], fraction: float) -> dict:
    """Match additional-probe counts, not milliseconds or observed-value ranks."""
    positives = {model: [r for r in rows if r[model + "_prediction_ms"] > 0] for model in ("B", "H")}
    count = min(math.ceil(len(rows) * fraction), len(positives["B"]), len(positives["H"]))
    result = {"maximum_fraction": fraction, "matched_count": count,
              "budget_unit": "additional_probe_count", "elapsed_ms_matched": False,
              "status": "evaluated" if count else "not_estimable:no_common_positive_budget"}
    for model in ("B", "H"):
        selected = sorted(positives[model], key=lambda r: (-r[model + "_prediction_ms"], r["question_id"], r["candidate_index"]))[:count]
        result[model] = {"realized_net_utility_sum_ms": float(sum(r["z_ms"] for r in selected)),
                         "positive_realized_records": sum(r["z_ms"] > 0 for r in selected),
                         "selected_ids": [[r["question_id"], r["candidate_index"]] for r in selected]}
    result["H_minus_B_net_utility_ms"] = (result["H"]["realized_net_utility_sum_ms"]
                                             - result["B"]["realized_net_utility_sum_ms"]) if count else None
    return result


def summarize_predictions(rows: list[dict]) -> dict:
    if not rows:
        return {"status": "not_estimable", "evaluated_records": 0, "evaluated_questions": 0}
    targets = [r["z_ms"] for r in rows]
    metrics = {model: regression_metrics(targets, [r[model + "_prediction_ms"] for r in rows])
               for model in ("B", "H", "constant")}
    reference = metrics["constant"]
    for model in ("B", "H"):
        metrics[model].update(mae_improvement_over_constant_ms=reference["mae_ms"] - metrics[model]["mae_ms"],
                              mse_improvement_over_constant_ms2=reference["mse_ms2"] - metrics[model]["mse_ms2"],
                              mae_ratio_to_constant=(metrics[model]["mae_ms"] / reference["mae_ms"]
                                                     if reference["mae_ms"] else None),
                              mse_ratio_to_constant=(metrics[model]["mse_ms2"] / reference["mse_ms2"]
                                                     if reference["mse_ms2"] else None))
    return {"status": "exploratory", "evaluated_records": len(rows),
            "evaluated_questions": len({r["question_id"] for r in rows}), "metrics": metrics,
            "H_minus_B_mae_ms": metrics["H"]["mae_ms"] - metrics["B"]["mae_ms"],
            "H_minus_B_mse_ms2": metrics["H"]["mse_ms2"] - metrics["B"]["mse_ms2"],
            "positive_prediction_policies": {model: _policy_summary(rows, model) for model in ("B", "H")},
            "matched_positive_budgets": [matched_budget(rows, fraction) for fraction in (.25, .5)]}


def question_bootstrap(rows: list[dict], replicates: int = 2000) -> dict:
    """Cluster bootstrap of saved outer predictions; it does not refit models.

    Intervals describe this small development sample conditional on the fitted
    folds.  They are not confirmatory intervals or model-selection correction.
    """
    groups = {qid: [r for r in rows if r["question_id"] == qid] for qid in sorted({r["question_id"] for r in rows})}
    if len(groups) < 4 or replicates < 1:
        return {"status": "not_estimable", "reason": "at_least_four_evaluated_questions_and_positive_replicates_required"}
    ids, rng = list(groups), np.random.default_rng(FOLD_SEED + 900)
    values = {"H_minus_B_mae_ms": [], "H_minus_B_mse_ms2": [],
              "H_minus_B_positive_policy_mean_utility_ms": []}
    for _ in range(replicates):
        sample = [row for i in rng.integers(0, len(ids), size=len(ids)) for row in groups[ids[int(i)]]]
        y = np.asarray([r["z_ms"] for r in sample])
        b = np.asarray([r["B_prediction_ms"] for r in sample])
        h = np.asarray([r["H_prediction_ms"] for r in sample])
        values["H_minus_B_mae_ms"].append(float(np.mean(np.abs(h - y) - np.abs(b - y))))
        values["H_minus_B_mse_ms2"].append(float(np.mean((h - y) ** 2 - (b - y) ** 2)))
        values["H_minus_B_positive_policy_mean_utility_ms"].append(float(np.mean(y * ((h > 0).astype(float) - (b > 0).astype(float)))))
    return {"status": "exploratory_conditional_on_fitted_outer_models", "resampling_unit": "question",
            "replicates": replicates, "seed": FOLD_SEED + 900,
            "percentile_95_intervals": {name: [float(v) for v in np.quantile(data, [.025, .975])]
                                        for name, data in values.items()}}


def analyze(document: dict, bootstrap_replicates: int = 2000) -> dict:
    roster, rows, coverage = prepare_records(document)
    results = []
    for index, error_cost in enumerate(ERROR_COSTS_MS):
        predictions, audits = nested_predictions(roster, rows, error_cost)
        results.append({"lambda_error_ms": error_cost, "role": "primary" if index == 0 else "sensitivity_only",
                        "summary": summarize_predictions(predictions), "fold_audit": audits,
                        "bootstrap": question_bootstrap(predictions, bootstrap_replicates),
                        "held_out_predictions": predictions})
    return {"analysis_schema_version": SCHEMA_VERSION, "status": "development_analysis_only",
            "protocol": {"outer_folds": 4, "inner_folds": 4, "fold_seed": FOLD_SEED,
                         "grouping": "question", "ridge_alpha": RIDGE_ALPHA,
                         "minimum_training_questions": MIN_TRAIN_QUESTIONS,
                         "primary_lambda_error_ms": ERROR_COSTS_MS[0],
                         "sensitivity_lambda_error_ms": list(ERROR_COSTS_MS[1:]),
                         "short_probe_cap": 21, "long_probe_cap": 42, "main_continuation_cap": 256,
                         "finalizer_cap": 128, "maximum_candidates_per_question": 3,
                         "short_g_features": "short_H", "long_g_features": "short_H_plus_long_H",
                         "g_target": "L_E-L_T; choose T iff prediction>0; ties choose E",
                         "value_target": "L_selected_by_short_g-L_selected_by_long_g-extra_probe_ms",
                         "value_features": "B:short_probability_curve_history_length; H:B_plus_short_entropy",
                         "unknown_grade_policy": "exclude_from_supervised_targets_and_report",
                         "metric_weighting": "candidate-level; bootstrap groups all candidates of a question",
                         "budget_matching": "top positive predictions, shared additional-probe count capped at 25% and 50%; milliseconds are not matched",
                         "followup_policy": "one prescribed decision opportunity per paired branch; no repeated deployment claim"},
            "coverage": coverage, "outer_question_folds": grouped_folds(roster),
            "feature_names": {view: sorted(rows[0][view]) if rows else [] for view in ("B", "H", "long_H")},
            "results": results,
            "limitations": ["Development questions are exposed; no benchmark or novelty conclusion follows.",
                            "Costs and grades are supplied measured outcomes; analysis does not generate or grade answers.",
                            "E/T regressions are utility scores, not correctness probabilities or risk guarantees.",
                            "The short and long selectors differ in observation budget and are separately fitted; finite-sample model differences remain.",
                            "Bootstrap reuses fitted out-of-fold predictions and does not capture all training uncertainty.",
                            "Up to three paired opportunities per question do not represent repeated adaptive deployment."]}
