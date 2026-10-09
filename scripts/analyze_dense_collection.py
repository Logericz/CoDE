#!/usr/bin/env python3
"""Audit a ten-question dense collection against its saved development pilot.

CPU/file analysis only. No model, gold-based selection, answer extraction or
grading. Adaptive replay uses times observed during dense collection, never
reports them as an online sparse-policy latency, and stops at its first exit.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from online_engine import json_safe
from online_protocol import CostObservation, ProbeObservation, ProtocolConfig, ProtocolController, ScheduleConfig
from online_source_manifest import METHOD_IDENTITY_FILES

LABEL = "dense-collect-no-stop"
METHOD = "dense_collect_no_stop"
PHASES = ("prefill", "reason", "probe_cache", "answer")
CORE_SOURCE_FILES = ("src/online_engine.py", "src/online_protocol.py", "src/online_contract.py",
                     "src/torch_online_backend.py", *METHOD_IDENTITY_FILES)
REPLAY_SOURCE_FILES = ("src/online_protocol.py", *METHOD_IDENTITY_FILES)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def measured_sum(values):
    values = list(values)
    return sum(values) if all(finite(x) and x >= 0 for x in values) else None


def strict_json(data):
    def reject(value):
        raise ValueError(f"Nonstandard JSON number {value}; retain the engine's explicit nonfinite tag")
    return json.loads(data, parse_constant=reject)


class Audit:
    def __init__(self):
        self.checks, self.inputs = [], {}

    def check(self, name, condition, detail=None):
        self.checks.append({"check": name, "passed": bool(condition),
                            **({"detail": detail} if detail is not None else {})})

    def read(self, path):
        path = Path(path).resolve()
        data = path.read_bytes()
        self.inputs[str(path)] = hashlib.sha256(data).hexdigest()
        return strict_json(data)

    def bind(self, path):
        path = Path(path).resolve()
        self.inputs[str(path)] = sha256(path)
        return self.inputs[str(path)]

    def finish(self):
        for path, digest in list(self.inputs.items()):
            try:
                unchanged = Path(path).is_file() and sha256(path) == digest
            except OSError:
                unchanged = False
            self.check("input_unchanged:" + path, unchanged)


def inside(root, relative):
    relative = Path(relative)
    root = Path(root).resolve()
    target = (root / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not target.is_relative_to(root):
        raise ValueError(f"Evidence path escapes run: {relative}")
    return target


def load_run(audit, root, expected_count):
    root = Path(root).resolve()
    manifest, summary = (audit.read(root / name) for name in ("manifest.json", "summary.json"))
    plan, executed = manifest["requests"], summary["requests"]
    prefix = root.name + ":"
    audit.check(prefix + "planned_denominator", len(plan) == expected_count
                == manifest.get("planned_count") == summary.get("planned_count"))
    audit.check(prefix + "unique_plan", len({p["relative_directory"] for p in plan}) == len(plan)
                and [p["request_index"] for p in plan] == list(range(len(plan))))
    audit.check(prefix + "execution_is_plan_prefix", len(executed) <= len(plan) and all(
        all(row.get(k) == value for k, value in job.items()) for row, job in zip(executed, plan)))
    failed = sum(row["status"] != "completed" for row in executed)
    counts = {"planned": expected_count, "executed": len(executed),
              "completed": len(executed) - failed, "failed": failed,
              "unexecuted": expected_count - len(executed)}
    audit.check(prefix + "summary_counts", all(summary.get(k + "_count") == v for k, v in counts.items())
                and summary.get("unexecuted_requests") == plan[len(executed):])
    before, after = (audit.read(root / name) for name in ("integrity-before.json", "integrity-after.json"))
    audit.check(prefix + "recorded_source_integrity", bool(before) and before == after
                == manifest.get("source_input_sha256")
                and summary.get("source_input_identity_integrity") == "unchanged")
    records = {}
    for row in executed:
        path = inside(root, row["relative_directory"]) / "request.json"
        record = audit.read(path)
        audit.check(prefix + row["relative_directory"] + ":request_hash",
                    audit.inputs[str(path)] == row["source_sha256"])
        audit.check(prefix + row["relative_directory"] + ":source_status",
                    record["status"] == row.get("source_execution_status", row["status"]))
        key = (row["development_id"], row["configuration"])
        if key in records:
            raise ValueError("Duplicate executed request identity")
        records[key] = (row, record)
    observed = {str(p.resolve()) for p in root.glob("requests/*/*/request.json")}
    expected = {str(inside(root, row["relative_directory"]) / "request.json") for row in executed}
    audit.check(prefix + "no_unlisted_request_files", observed == expected)
    return {"root": root, "manifest": manifest, "summary": summary,
            "counts": counts, "records": records}


def observation(row, markers):
    raw = row["raw_observation"]
    result = ProbeObservation(**raw)
    if result.ended_with_think != bool(result.token_ids and result.token_ids[-1] == markers["end_think"]):
        raise ValueError("Probe ending flag differs from its saved last token")
    # Mirror the frozen engine's post-raw-evidence EOS invalidation.
    if any(token in markers["eos_ids"] for token in result.token_ids):
        result = replace(result, invalid_reason="probe_contains_eos")
    return result


def adaptive_replay(probes, markers, config):
    """Forward only: skipped raw observations never enter controller history.

    Sum adjacent dense reason intervals when skipping queries. Those intervals
    are observed dense-run costs, not measurements of a sparse execution.
    Explicit invalid/missing times stay unavailable and trigger the protocol's
    own fallback; they are not replaced with invented positive times.
    """
    controller = ProtocolController(config, ScheduleConfig(kind="adaptive"))
    decisions, skipped, intervals = [], [], []
    queried_before, stopped = False, False
    for row in probes:
        if stopped:
            break
        if "decision" not in row:
            return {"status": "partial_probe_without_decision", "decisions": decisions,
                    "skipped_candidate_j": skipped, "stopped": False}
        if queried_before:
            intervals.append(row["decision"]["cost"].get("reason_elapsed_ms"))
        j = row["candidate_j"]
        if not controller.should_probe(j):
            skipped.append(j)
            continue
        reason_ms = measured_sum(intervals) if queried_before else None
        decision = controller.observe(j, row["token_position"], observation(row, markers),
            CostObservation(probe_elapsed_ms=row.get("probe_elapsed_ms"), reason_elapsed_ms=reason_ms))
        decisions.append(json_safe(asdict(decision)))
        queried_before, intervals, stopped = True, [], decision.should_stop
    reasons = Counter(reason for d in decisions for reason in d["forced_dense_reasons"])
    # This is a separate algebraic diagnostic, not an h_cost the controller
    # actually evaluated: a fallback or a stop may bypass that formula.
    cost_caps = [{"candidate_j": d["candidate_j"], "rho": d["rho"],
                  "cost_cap_if_formula_evaluated": min(max(math.ceil(d["rho"] / 0.5), 1), 8)}
                 for d in decisions if not d["stop_reasons"] and finite(d["rho"]) and d["rho"] >= 0]
    return {"status": "replayed_saved_prefix", "stopped": stopped,
            "stop_candidate_j": decisions[-1]["candidate_j"] if stopped else None,
            "queried_candidate_j": [d["candidate_j"] for d in decisions],
            "skipped_candidate_j": skipped, "forced_dense_reason_counts": dict(reasons),
            "formula_evaluated_count": sum(d["h_signal"] is not None for d in decisions),
            "scheduled_sparse_decisions_h_next_gt_1": sum(d["h_next"] is not None and d["h_next"] > 1 for d in decisions),
            "continuing_decisions": sum(not d["stop_reasons"] for d in decisions),
            "hypothetical_cost_cap_from_recorded_rho": cost_caps,
            "hypothetical_cost_cap_one_count": sum(d["cost_cap_if_formula_evaluated"] == 1 for d in cost_caps),
            "decisions": decisions, "time_source": "saved_dense_collection_intervals",
            "online_latency_estimate_ms": None,
            "interpretation": "Only queried observations enter history; nothing after the first stop is consumed. "
                              "A scheduled next query beyond the saved endpoint is not an observed skip."}


def probe_counts(probes, markers):
    observed = [observation(row, markers) for row in probes]
    count = len(observed)
    incomplete = sum(not row.ended_with_think for row in observed)
    return {"count": count, "incomplete": incomplete,
            "incomplete_fraction": incomplete / count if count else None,
            "invalid": sum(not row.confidence_valid for row in observed),
            "valid_incomplete": sum(row.confidence_valid and not row.ended_with_think for row in observed)}


def compare_question(audit, job, entry, vanilla_entry, dense_entry):
    summary_row, record = entry
    _, vanilla = vanilla_entry
    _, dense = dense_entry
    name = job["development_id"]
    config = job["request_configuration"]
    complete = summary_row["status"] == "completed" and record["status"] == "completed"
    samples, probes, candidates = (record.get(k, []) for k in ("main_samples", "probes", "candidates"))
    markers = record["backend_metadata"]["markers"]
    audit.check(name + ":identity", all(record.get(k) == job[k] for k in
                ("sample_id", "development_id", "configuration")) and record.get("method") == METHOD
                and record.get("request_configuration") == config and record.get("seed_reason") == config["seed_reason"]
                and record.get("protocol_config") == config["protocol_config"]
                and record.get("schedule_config") == config["schedule_config"]
                and record.get("eligible_for_primary_speed_comparison") is False)
    audit.check(name + ":paired_generation_identity", all(record.get(k) == vanilla.get(k) == dense.get(k)
                for k in ("sample_id", "seed_reason", "rollout_id", "max_new_tokens", "prompt_token_ids", "protocol_config"))
                and all(record["backend_metadata"].get(k) == vanilla["backend_metadata"].get(k)
                        == dense["backend_metadata"].get(k) for k in
                        ("model_revision", "tokenizer_revision", "sampling", "probe", "markers", "model_parameter_dtype",
                         "attention_implementation", "runner_protocol")))
    common = min(len(samples), len(vanilla["main_samples"]))
    prefix_exact = samples[:common] == vanilla["main_samples"][:common]
    audit.check(name + ":vanilla_main_prefix", prefix_exact)
    audit.check(name + ":dense_main_prefix", samples[:min(len(samples), len(dense["main_samples"]))]
                == dense["main_samples"][:min(len(samples), len(dense["main_samples"]))])
    if complete:
        audit.check(name + ":complete_vanilla_main_trajectory", bool(samples) and samples == vanilla["main_samples"]
                    and record["stop_reason"] == vanilla["stop_reason"])
    # Candidate positions count all generated main tokens, not only reason
    # tokens; Wait after the first natural think exit is never a candidate.
    expected_positions, thinking, phases_valid = [], True, True
    for index, sample in enumerate(samples):
        token = sample["token_id"]
        phases_valid &= sample.get("generation_index") == index and sample.get("phase") == ("reason" if thinking else "answer")
        phases_valid &= all(finite(sample.get(k)) and 0 <= sample[k] <= 1 for k in ("raw_probability", "sample_probability"))
        if token in markers["eos_ids"]:
            phases_valid &= index == len(samples) - 1
            break
        if thinking and token == markers["end_think"]:
            thinking = False
        elif thinking and token == markers["wait"]:
            expected_positions.append(index)
    actual_positions = [row["token_position"] for row in candidates]
    # A deadline can fire after sampling Wait but before recording its event.
    audit.check(name + ":sample_phases", phases_valid)
    audit.check(name + ":candidate_positions", actual_positions == expected_positions if complete
                else actual_positions == expected_positions[:len(actual_positions)])
    audit.check(name + ":candidate_ids", [c["candidate_j"] for c in candidates] == list(range(1, len(candidates) + 1))
                and all(c["pending_token_id"] == markers["wait"] for c in candidates))
    queried = [c["candidate_j"] for c in candidates if c["queried"]]
    audit.check(name + ":probe_query_sequence", [p["candidate_j"] for p in probes] == queried
                and (not complete or len(probes) == len(candidates)))
    positions = {c["candidate_j"]: c["token_position"] for c in candidates}
    audit.check(name + ":probe_positions", all(p["token_position"] == positions[p["candidate_j"]] for p in probes))
    original = {p["candidate_j"]: p for p in dense["probes"]}
    shared = [p for p in probes if p["candidate_j"] in original]
    audit.check(name + ":shared_dense_raw_observations", all(
        p["token_position"] == original[p["candidate_j"]]["token_position"]
        and p["raw_observation"] == original[p["candidate_j"]]["raw_observation"] for p in shared))
    if complete:
        audit.check(name + ":old_dense_probe_coverage", len(shared) == len(original))
    controller = ProtocolController(ProtocolConfig(**config["protocol_config"]), ScheduleConfig(), stop_enabled=False)
    for p in probes:
        if "decision" not in p:
            audit.check(name + ":unfinished_probe_is_failed", not complete)
            break
        decision = controller.observe(p["candidate_j"], p["token_position"], observation(p, markers),
                                      CostObservation(**p["decision"]["cost"]))
        audit.check(name + f":dense_decision:{p['candidate_j']}", json_safe(asdict(decision)) == p["decision"]
                    and p["decision"]["cost"]["probe_elapsed_ms"] == p.get("probe_elapsed_ms")
                    and p.get("would_stop") == decision.would_stop
                    and p.get("should_stop") is False and p.get("stop_applied") is False)
    first = next((p for p in probes if p.get("would_stop")), None)
    previous_stop = next((p for p in dense["probes"] if p.get("stop_applied")), None)
    stop_covered = complete or (previous_stop is not None and any(p["candidate_j"] == previous_stop["candidate_j"] for p in probes))
    first_key = (first["candidate_j"], first["token_position"]) if first else None
    old_key = (previous_stop["candidate_j"], previous_stop["token_position"]) if previous_stop else None
    if stop_covered:
        audit.check(name + ":first_would_stop_matches_old_dense", first_key == old_key)
    tail = [p for p in probes if previous_stop and p["candidate_j"] > previous_stop["candidate_j"]]
    reason_tokens = sum(s["phase"] == "reason" for s in samples)
    natural_tokens = sum(s["phase"] == "answer" for s in samples)
    finals = record.get("finalization_samples", [])
    probe_tokens = sum(len(p["raw_observation"]["token_ids"]) for p in probes)
    phases = record.get("phase_elapsed_ms", {})
    audit.check(name + ":phase_times", set(phases) == set(PHASES)
                and all(finite(t) and t >= 0 for t in phases.values()))
    if "actual_generated_tokens" in record:
        audit.check(name + ":token_conservation", len(samples) == record["main_generated_tokens"]
                    == record["accepted_main_tokens"] and record["discarded_generated_tokens"] == 0
                    and reason_tokens == record["reason_tokens"] and natural_tokens == record["natural_answer_tokens"]
                    and len(finals) == record["finalization_tokens"] and probe_tokens == record["probe_tokens"]
                    and len(samples) + len(finals) + probe_tokens == record["actual_generated_tokens"])
        audit.check(name + ":probe_time_sum", measured_sum(p.get("probe_elapsed_ms") for p in probes) is not None
                    and abs(sum(p["probe_elapsed_ms"] for p in probes) - phases.get("probe_cache", -1)) <= 1e-6)
        audit.check(name + ":time_closure", finite(record.get("time_total_ms"))
                    and measured_sum(phases.values()) is not None and finite(record.get("time_other_ms"))
                    and record["time_other_ms"] >= 0
                    and abs(record["time_total_ms"] - sum(phases.values()) - record["time_other_ms"]) <= 1e-6)
    natural_end = complete and record.get("stop_reason") == "natural_eos"
    if natural_end:
        audit.check(name + ":natural_eos", samples[-1]["token_id"] in markers["eos_ids"]
                    and record["injected_prompt_tokens"] == record["finalization_tokens"] == 0)
    duration_field = next((k for k in ("time_total_ms", "elapsed_ms_before_error_report", "external_invocation_elapsed_ms")
                           if finite(record.get(k)) and record[k] >= 0), None)
    return {"development_id": name, "sample_id": job["sample_id"], "execution_status": summary_row["status"],
            "source_execution_status": record["status"], "source_sha256": summary_row["source_sha256"],
            "vanilla_common_main_samples": common, "vanilla_prefix_evidence_available": common > 0,
            "vanilla_prefix_exact": prefix_exact, "vanilla_full_main_exact": complete and samples == vanilla["main_samples"],
            "main_samples_saved": len(samples), "reason_samples_saved": reason_tokens,
            "natural_answer_samples_saved": natural_tokens, "finalization_samples_saved": len(finals),
            "probe_tokens_saved": probe_tokens, "candidate_count": len(candidates),
            "probes": probe_counts(probes, markers), "shared_old_dense_probes": len(shared),
            "first_would_stop": list(first_key) if first_key else None,
            "old_dense_stop": list(old_key) if old_key else None, "old_stop_comparison_covered": stop_covered,
            "probes_after_old_dense_stop": probe_counts(tail, markers),
            "natural_endpoint_covered": natural_end,
            "budget_endpoint_covered": complete and record.get("stop_reason") == "budget",
            "stop_reason": record.get("stop_reason", record.get("stop_reason_before_error")),
            "elapsed": {"field": duration_field, "milliseconds": record.get(duration_field) if duration_field else None},
            "phase_elapsed_ms": phases, "peak_allocated_bytes": record.get("peak_memory_bytes"),
            "peak_reserved_bytes": record.get("peak_reserved_bytes"),
            "adaptive_replay": adaptive_replay(probes, markers, ProtocolConfig(**config["protocol_config"]))}


def analyze(collection_root, development_root):
    audit, result = Audit(), {"schema_version": 1, "scope": "offline_dense_collection_diagnostic_not_online_speed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "questions": []}
    try:
        collection = load_run(audit, collection_root, 10)
        previous = load_run(audit, development_root, 40)
        cm, pm = collection["manifest"], previous["manifest"]
        audit.check("same_frozen_ten_questions", len(cm["questions"]) == 10 and cm["questions"] == pm["questions"])
        audit.check("same_data_identity", cm.get("data_identity") == pm.get("data_identity"))
        audit.check("previous_matrix_completed", previous["counts"]["completed"] == 40)
        evidence = cm["development_evidence"]
        audit.check("bound_development_manifests", evidence.get("manifest_sha256") == audit.inputs[str(previous["root"] / "manifest.json")]
                    and evidence.get("summary_sha256") == audit.inputs[str(previous["root"] / "summary.json")]
                    and evidence.get("planned_count") == 40)
        for question in cm["questions"]:
            dev = question["development_id"]
            for label in ("vanilla", "codestop-dense"):
                audit.check(f"bound_development_request:{dev}:{label}",
                    evidence["paired_sources"][dev][label]["sha256"] == previous["records"][(dev, label)][0]["source_sha256"])
        code = cm["code_sha256"]
        audit.check("same_engine_backend_protocol", all(code.get(k) == pm["code_sha256"].get(k)
                    and code.get(k) is not None for k in CORE_SOURCE_FILES))
        for filename in REPLAY_SOURCE_FILES:
            local_hash = audit.bind(ROOT / filename)
            audit.check("replay_source_identity:" + filename, code.get(filename) == local_hash)
            if code.get(filename) != local_hash:
                raise ValueError("Local replay source is not the collection's frozen source: " + filename)
        result["counts"] = collection["counts"]
        result["source_integrity_scope"] = "Saved before/after hashes compared; unavailable remote absolute paths are not re-read."
        for job in cm["requests"]:
            config = job["request_configuration"]
            payload = {k: v for k, v in config.items() if k != "config_hash"}
            audit.check(job["development_id"] + ":configuration", job["configuration"] == LABEL
                        and config["method"] == METHOD and config["stopping_enabled"] is False
                        and config["protocol_config"] == asdict(ProtocolConfig())
                        and config["schedule_config"] == asdict(ScheduleConfig())
                        and hashlib.sha256(encoded(payload)).hexdigest() == config["config_hash"])
            key = (job["development_id"], LABEL)
            if key not in collection["records"]:
                result["questions"].append({"development_id": job["development_id"],
                    "sample_id": job["sample_id"], "execution_status": "unexecuted"})
                continue
            result["questions"].append(compare_question(audit, job, collection["records"][key],
                previous["records"][(job["development_id"], "vanilla")],
                previous["records"][(job["development_id"], "codestop-dense")]))
        result["collection_complete"] = collection["counts"]["completed"] == 10
        result["natural_endpoint_count"] = sum(q.get("natural_endpoint_covered", False) for q in result["questions"])
    except (Exception, KeyboardInterrupt) as error:
        audit.check("analysis_exception", False, {"type": type(error).__name__, "message": str(error)})
    finally:
        audit.finish()
    result.update(status="passed" if all(c["passed"] for c in audit.checks) else "failed",
        checks=audit.checks, check_count=len(audit.checks), failed_checks=[c for c in audit.checks if not c["passed"]],
        input_sha256=audit.inputs, analysis_source_sha256=sha256(__file__),
        quality_policy="No grading or adjudication performed. Original strict grading remains authoritative; no gold is used.",
        limitations=["Ten exposed development questions, one seed; not calibration, parameter selection, or a test result.",
            "All ten planned questions remain in the denominator. Failed/partial records do not establish endpoint coverage.",
            "Natural-answer samples are separate from reasoning samples; the pending Wait is part of main generation cost.",
            "After-stop dense probes are diagnostic future observations, not information available to the earlier stopped policy.",
            "Adaptive replay uses dense-run intervals and cannot establish sparse online timing, speedup, or answer quality.",
            "Hypothetical cost caps only apply the default formula to already recorded rho; they are not executed h_cost or causal runtime evidence.",
            "Missing or invalid numerical evidence is retained; no confidence, latency or memory value is imputed.",
            "Saved same-KV comparisons do not establish equivalence to full-prefix recomputation."])
    return json_safe(result)


def markdown(report):
    rows = ["# Dense collection offline analysis", "", f"Audit: **{report['status']}**; {report['check_count']} checks.",
            "", "This is diagnostic collection and offline replay, not an online speed result.", "",
            "Counts: `" + json.dumps(report.get("counts", {}), sort_keys=True) + "`", "",
            "| Question | Execution | Main / reason / natural answer | Probes / incomplete | Added after old stop | Natural end | Adaptive skips |",
            "| --- | --- | --- | --- | --- | --- | --- |"]
    for q in report["questions"]:
        if "probes" not in q:
            rows.append(f"| {q['development_id']} | {q['execution_status']} | — | — | — | — | — |")
            continue
        rows.append(f"| {q['development_id']} | {q['execution_status']} | {q['main_samples_saved']} / {q['reason_samples_saved']} / "
            f"{q['natural_answer_samples_saved']} | {q['probes']['count']} / {q['probes']['incomplete']} | "
            f"{q['probes_after_old_dense_stop']['count']} | {q['natural_endpoint_covered']} | "
            f"{len(q['adaptive_replay']['skipped_candidate_j'])} |")
    rows += ["", report["quality_policy"], "", *["- " + x for x in report["limitations"]], ""]
    if report["failed_checks"]:
        rows += ["Failed checks:", "", *["- `" + x["check"] + "`" for x in report["failed_checks"]], ""]
    return "\n".join(rows)


def write_report(report, output_dir, input_roots):
    output = Path(output_dir).expanduser().resolve()
    if output.exists() or not output.is_relative_to((ROOT / "runs").resolve()):
        raise ValueError("Output must be a new directory within the repository's ignored runs/")
    if any(output.is_relative_to(Path(p).resolve()) or Path(p).resolve().is_relative_to(output) for p in input_roots):
        raise ValueError("Output must be separate from the original input evidence")
    output.mkdir(parents=True, exist_ok=False)
    (output / "analysis.json").write_bytes(encoded(report))
    (output / "analysis.md").write_text(markdown(report), encoding="utf-8")
    (output / "OUTPUT_SHA256.json").write_bytes(encoded({name: sha256(output / name)
                                                       for name in ("analysis.json", "analysis.md")}))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-run", required=True, type=Path)
    parser.add_argument("--development-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    report = analyze(args.collection_run, args.development_run)
    output = write_report(report, args.output_dir, (args.collection_run, args.development_run))
    print(json.dumps({"status": report["status"], "checks": report["check_count"],
                      "failed_checks": report["failed_checks"], "output": str(output)}, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
