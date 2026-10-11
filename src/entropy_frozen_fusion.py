"""Freeze the old-data short selectors, then evaluate new questions without fitting.

The artifact stores only model numbers, feature order and provenance/identities.
It contains no question text, generated answer or per-example training label.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np

from . import entropy_short_fusion as fusion
from . import entropy_value_analysis as original

SCHEMA = "entropy-frozen-fusion-v1"
PARAMETERS = ("feature_mean", "expanded_mean", "expanded_scale", "coefficients")


def digest(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def _payload(artifact):
    return {key: value for key, value in artifact.items() if key != "payload_sha256"}


def serialize_model(model):
    return {name: getattr(model, name).tolist() for name in PARAMETERS} | {"intercept": float(model.intercept)}


def deserialize_model(values, dimensions):
    if set(values) != {*PARAMETERS, "intercept"}:
        raise ValueError("Unexpected ridge parameter schema")
    arrays = {}
    for name in PARAMETERS:
        array = np.asarray(values[name], dtype=float)
        width = dimensions if name == "feature_mean" else dimensions * 2
        if array.shape != (width,) or not np.isfinite(array).all():
            raise ValueError("Invalid ridge parameter dimensions or nonfinite values: " + name)
        arrays[name] = array
    if np.any(arrays["expanded_scale"] <= 0) or not original.finite_number(values["intercept"]):
        raise ValueError("Ridge scales must be positive and intercept finite")
    return original.RidgeModel(**arrays, intercept=float(values["intercept"]))


def validate_artifact(artifact):
    if artifact.get("schema_version") != SCHEMA:
        raise ValueError("Unknown frozen fusion schema")
    if artifact.get("payload_sha256") != digest(_payload(artifact)):
        raise ValueError("Frozen model payload hash mismatch")
    if artifact.get("ridge_alpha") != original.RIDGE_ALPHA:
        raise ValueError("Frozen ridge alpha changed")
    if artifact.get("lambda_error_ms") != list(original.ERROR_COSTS_MS):
        raise ValueError("Frozen error penalties changed")
    expected = {name: sorted(fusion.FEATURE_SPEC[name]) for name in fusion.LEARNED}
    if artifact.get("feature_names") != expected:
        raise ValueError("Frozen feature names or order changed")
    settings = artifact.get("models", [])
    if [item["lambda_error_ms"] for item in settings] != list(original.ERROR_COSTS_MS):
        raise ValueError("Frozen model settings must have the three fixed penalties")
    for item in settings:
        if set(item["learned"]) != set(fusion.LEARNED) or not original.finite_number(item["trainmean"]):
            raise ValueError("Frozen learned model set or mean is invalid")
        for name, values in item["learned"].items():
            deserialize_model(values, len(expected[name]))
    return artifact


def freeze(document, training_identity):
    """Fit each fixed family on all valid old pairs, exactly once per penalty.

    training_identity supplies input_sha256 and source_sha256. The CLI checks
    their actual bytes before and after fitting. This function never reads files.
    """
    if set(training_identity) != {"input_sha256", "source_sha256"}:
        raise ValueError("Training identity needs input_sha256 and source_sha256")
    hashes = [training_identity["input_sha256"], *training_identity["source_sha256"].values()]
    if not training_identity["source_sha256"] or any(not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h) for h in hashes):
        raise ValueError("Training identity needs valid SHA256 digests")
    roster, rows, _ = fusion.prepare(document)
    if not original._enough_training(rows):
        raise ValueError("At least four valid training questions are required")
    settings = []
    for penalty in original.ERROR_COSTS_MS:
        target = np.asarray([e - t for e, t in (original.action_losses(r, penalty) for r in rows)])
        settings.append({"lambda_error_ms": penalty, "trainmean": float(target.mean()),
            "learned": {name: serialize_model(original.fit_ridge(fusion.matrix(rows, name), target, original.RIDGE_ALPHA))
                        for name in fusion.LEARNED}})
    artifact = {"schema_version": SCHEMA, "role": "old_data_frozen_selectors_new_question_evaluation_only",
        "ridge_alpha": original.RIDGE_ALPHA, "lambda_error_ms": list(original.ERROR_COSTS_MS),
        "feature_names": {name: sorted(fusion.FEATURE_SPEC[name]) for name in fusion.LEARNED},
        "training_identity": training_identity,
        "training": {"planned_question_ids": roster, "valid_records": len(rows),
            "valid_questions": len({r["question_id"] for r in rows}),
            "record_ids": [{key: r[key] for key in ("question_id", "candidate_index")} for r in rows]},
        "models": settings}
    artifact["payload_sha256"] = digest(artifact)
    return validate_artifact(artifact)


def predict_frozen(artifact, rows, penalty):
    """Use only saved preprocessing and coefficients; row outcomes are reporting data."""
    validate_artifact(artifact)
    if penalty not in original.ERROR_COSTS_MS:
        raise ValueError("Only the three frozen penalties are supported")
    if not rows:
        return []
    setting = next(item for item in artifact["models"] if item["lambda_error_ms"] == penalty)
    scores = {name: deserialize_model(values, len(artifact["feature_names"][name])).predict(fusion.matrix(rows, name))
              for name, values in setting["learned"].items()}
    output = []
    for i, row in enumerate(rows):
        score = {name: float(values[i]) for name, values in scores.items()}
        score["trainmean"] = setting["trainmean"]
        score["late_avg"] = (score["C"] + score["H"]) / 2
        loss_e, loss_t = original.action_losses(row, penalty)
        output.append({key: row[key] for key in ("row", "question_id", "candidate_index", "primary_eligible", "short_probe_ms",
            "E_error", "T_error", "E_ms", "T_ms")} | {"scores": score, "actions": fusion.combine_scores(score),
                "loss_E_ms": loss_e, "loss_T_ms": loss_t, "delta_ms": loss_e - loss_t})
    return output


def evaluate(document, artifact, bootstrap_replicates=2000):
    """Labels on new questions affect reported losses, never a fitted parameter."""
    validate_artifact(artifact)
    if type(bootstrap_replicates) is not int or bootstrap_replicates < 0:
        raise ValueError("Bootstrap count must be a nonnegative integer")
    roster, rows, coverage = fusion.prepare(document)
    overlap = set(roster) & set(artifact["training"]["planned_question_ids"])
    if overlap:
        raise ValueError("Evaluation questions overlap training roster: " + ", ".join(sorted(overlap)))
    results = []
    for penalty in original.ERROR_COSTS_MS:
        predictions = predict_frozen(artifact, rows, penalty)
        scopes = {"primary": [r for r in predictions if r["primary_eligible"]], "all": predictions,
                  "other_short": [r for r in predictions if not r["primary_eligible"]]}
        results.append({"lambda_error_ms": penalty, "predictions": predictions,
                        "scopes": {name: fusion.summarize(items, bootstrap_replicates) for name, items in scopes.items()}})
    return {"schema_version": "entropy-frozen-fusion-evaluation-v1", "role": "new_questions_fixed_model_bounded_evaluation",
        "model_payload_sha256": artifact["payload_sha256"], "training_input_sha256": artifact["training_identity"]["input_sha256"],
        "question_ids": roster, "coverage": coverage, "primary_records": sum(r["primary_eligible"] for r in rows),
        "new_label_fits": 0, "results": results,
        "bootstrap": {"replicates": bootstrap_replicates, "seed": fusion.BOOTSTRAP_SEED, "unit": "question", "refits": False},
        "limitations": ["Frozen models fitted to the old development cache only; no new-label training or selection.",
            "C/H are feature ablations, not complete CoDE-Stop or EntroCut implementations.",
            "Fixed local E/T interventions are not whole-question online policy rollouts.",
            "Entropy/classifier deployment overhead is unmeasured; common short probe cost is separate."]}
