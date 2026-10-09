#!/usr/bin/env python3
"""Replay one new guarded candidate on all ten frozen development trajectories.

The archived a08/a09 evidence is authenticated separately from current code.
This is a post hoc counterfactual diagnostic, not a source-gate waiver, a new
GPU run, an answer-quality evaluation, or an estimate of sparse online latency.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from online_engine import json_safe
from online_protocol import CostObservation, ProbeObservation, ProtocolConfig, ProtocolController, ScheduleConfig
from online_source_manifest import METHOD_IDENTITY_FILES
from analyze_stop_opportunities import ANCHORS, Evidence, digest, encoded, load_run, observations

SOURCE_FILES = (
    "scripts/replay_guarded_candidate.py", "scripts/analyze_stop_opportunities.py",
    "docs/GUARDED_SCHEDULER.md",
    "src/online_protocol.py", "src/online_engine.py", "src/online_contract.py",
    "src/torch_online_backend.py", *METHOD_IDENTITY_FILES,
)
LABELS = ("dense", "fixed-h2", "adaptive", "guarded")
WORK_ROOT = ROOT / "runs/guarded-candidate-20261009"


def schedules():
    """Exactly one new candidate and one h=2 control; no parameter search."""
    return {"dense": ScheduleConfig(kind="dense"),
            "fixed-h2": ScheduleConfig(kind="fixed", fixed_interval=2),
            "adaptive": ScheduleConfig(kind="adaptive"),
            "guarded": ScheduleConfig(kind="guarded", margin_m0=.05)}


def measured_sum(values):
    # An unavailable interval stays unavailable; zero is retained so the
    # controller can apply its own nonpositive-time recovery rule.
    return sum(values) if all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in values) else None


def saved_observation(row, markers):
    result = ProbeObservation(**row["raw_observation"])
    if result.ended_with_think != bool(result.token_ids and result.token_ids[-1] == markers["end_think"]):
        raise ValueError("Saved probe ending does not match its final token")
    if any(token in markers["eos_ids"] for token in result.token_ids):
        result = replace(result, invalid_reason="probe_contains_eos")
    return result


def replay(probes, markers, protocol, schedule, natural_main_tokens):
    """Only queried observations enter history; never create a terminal probe.

    Timing inputs sum dense intervals since the previous actual query. They
    support the old Adaptive control, but do not measure sparse execution.
    """
    controller = ProtocolController(protocol, schedule)
    decisions, skipped, intervals = [], [], []
    queried_before, stopped, probe_tokens, previous_position = False, False, 0, 0
    for expected_j, row in enumerate(probes, 1):
        if stopped:
            break
        j, position = row["candidate_j"], row["token_position"]
        if j != expected_j or type(position) is not int or not previous_position < position < natural_main_tokens:
            raise ValueError("Saved candidates must be ordered, contiguous and before the natural endpoint")
        previous_position = position
        if queried_before:
            intervals.append(row.get("decision", {}).get("cost", {}).get("reason_elapsed_ms"))
        if not controller.should_probe(j):
            skipped.append(j)
            continue
        obs = saved_observation(row, markers)
        decision = controller.observe(j, position, obs, CostObservation(
            probe_elapsed_ms=row.get("probe_elapsed_ms"),
            reason_elapsed_ms=measured_sum(intervals) if queried_before else None))
        value = json_safe(asdict(decision))
        # Persist the controller inputs as evidence, not just the chosen h.
        decisions.append(value)
        probe_tokens += len(obs.token_ids)
        queried_before, intervals, stopped = True, [], decision.should_stop
    stop = decisions[-1] if stopped else None
    main_tokens = stop["token_position"] + 1 if stop else natural_main_tokens
    return {
        "schedule_config": asdict(schedule), "stopped": stopped,
        "stop_candidate_j": stop["candidate_j"] if stop else None,
        "stop_token_position": stop["token_position"] if stop else None,
        "stop_reasons": stop["stop_reasons"] if stop else [],
        "endpoint_kind": "threshold_stop" if stopped else "natural_eos",
        "queried_candidate_j": [d["candidate_j"] for d in decisions],
        "skipped_candidate_j": skipped, "n_probes": len(decisions),
        "next_scheduled_candidate_j": controller.next_candidate_j,
        "valid_history_candidate_j": [entry.candidate_j for entry in controller.history],
        "sparse_schedule_decisions": sum(d["h_next"] is not None and d["h_next"] > 1 for d in decisions),
        "forced_dense_reason_counts": dict(Counter(reason for d in decisions for reason in d["forced_dense_reasons"])),
        "partial_tokens": {"main_generated_including_pending_wait": main_tokens,
            "probe_generated": probe_tokens, "main_plus_probe": main_tokens + probe_tokens,
            "forced_answer_tokens": None, "injected_prompt_tokens": None,
            "interpretation": "Main plus queried-probe tokens only; forced finalization is excluded."},
        "decisions": decisions, "online_latency_estimate_ms": None,
        "timing_source": "archived_dense_intervals_only",
    }


def same_endpoint(result, actual, collection):
    stops = [p for p in actual["probes"] if p.get("stop_applied")]
    if result["stopped"]:
        if len(stops) != 1:
            return False
        stop = stops[0]
        same = (stop["candidate_j"] == result["stop_candidate_j"]
                and stop["token_position"] == result["stop_token_position"]
                and stop["decision"]["stop_reasons"] == result["stop_reasons"]
                and actual["main_generated_tokens"] == result["partial_tokens"]["main_generated_including_pending_wait"])
    else:
        same = not stops and actual["stop_reason"] == collection["stop_reason"] == "natural_eos"
        same &= actual["main_generated_tokens"] == collection["main_generated_tokens"]
    return bool(same and actual["main_samples"] == collection["main_samples"][:len(actual["main_samples"])])


def answer_availability(result, previous, collection):
    matched = [label for label, actual in previous.items() if same_endpoint(result, actual, collection)]
    return {"status": "existing_answer_available" if matched else "unknown",
            "matching_archived_configurations": matched, "graded": False,
            "interpretation": "Endpoint and stop reasons match an archived request; no answer is copied, graded or claimed correct."}


def analyze(collection_root, development_root, prior_path):
    archived, sources = Evidence(), Evidence()
    report = {"schema_version": 1, "scope": "posthoc_ten_question_guarded_candidate_replay",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "planned_question_count": 10,
        "planned_configurations": list(LABELS), "gold_used": False, "parameter_grid_search": False,
        "online_latency_estimate_ms": None, "questions": []}
    try:
        for name in SOURCE_FILES:
            sources.read(ROOT / name, parse=False)
        configs = schedules()
        prior = archived.read(prior_path, ANCHORS["prior_analysis"])
        archived.check("historical_434_check_gate", prior["status"] == "passed"
            and prior["check_count"] == len(prior["checks"]) == 434 and all(c["passed"] for c in prior["checks"])
            and prior["failed_checks"] == [] and prior["collection_complete"] is True and prior["natural_endpoint_count"] == 10)
        cm, collected = load_run(archived, collection_root, "collection", 10, prior)
        dm, previous = load_run(archived, development_root, "development", 40, prior)
        archived.check("same_frozen_questions_and_seed", cm["questions"] == dm["questions"]
            and [q["development_id"] for q in cm["questions"]] == [f"dev{i:02d}" for i in range(1, 11)]
            and cm["master_seed"] == dm["master_seed"] == 42 and cm["rollout_id"] == dm["rollout_id"] == 0)
        for question in cm["questions"]:
            dev = question["development_id"]
            record = collected[(dev, "dense-collect-no-stop")]
            old = {label: value for (qid, label), value in previous.items() if qid == dev}
            probes, markers = record["probes"], record["backend_metadata"]["markers"]
            archived.check(dev + ":natural_collection_complete", record["stop_reason"] == "natural_eos"
                and record["collection_complete_to_termination_or_cap"] is True
                and record["main_samples"] == old["vanilla"]["main_samples"]
                and len(record["main_samples"]) == record["main_generated_tokens"]
                and record["output_token_ids"][-1] in markers["eos_ids"]
                and record["injected_prompt_tokens"] == record["finalization_tokens"] == 0
                and record["n_probes"] == record["n_candidates"] == len(probes) == len(record["candidates"]))
            observations(probes, markers)  # Reject unsupported archived invalid/EOS observations.
            archived.check(dev + ":candidate_positions", [(p["candidate_j"], p["token_position"]) for p in probes]
                == [(c["candidate_j"], c["token_position"]) for c in record["candidates"]]
                and all(record["main_samples"][p["token_position"]]["token_id"] == markers["wait"]
                    and record["main_samples"][p["token_position"]]["phase"] == "reason" for p in probes))
            protocol = ProtocolConfig(**record["protocol_config"])
            results = {label: replay(probes, markers, protocol, config, record["main_generated_tokens"])
                       for label, config in configs.items()}
            for label in ("dense", "adaptive"):
                archived.check(dev + ":current_" + label + "_reproduces_archived_endpoint",
                    same_endpoint(results[label], old["codestop-" + label], record))
                archived.check(dev + ":current_" + label + "_reproduces_archived_queries",
                    results[label]["queried_candidate_j"] == [p["candidate_j"] for p in old["codestop-" + label]["probes"]])
            for result in results.values():
                result["answer_availability"] = answer_availability(result, old, record)
                result["versus_dense"] = {
                    "probe_count_difference": result["n_probes"] - results["dense"]["n_probes"],
                    "main_token_difference": result["partial_tokens"]["main_generated_including_pending_wait"] - results["dense"]["partial_tokens"]["main_generated_including_pending_wait"],
                    "partial_main_plus_probe_difference": result["partial_tokens"]["main_plus_probe"] - results["dense"]["partial_tokens"]["main_plus_probe"],
                }
            report["questions"].append({"development_id": dev, "sample_id": question["original_id"],
                "natural_main_tokens": record["main_generated_tokens"], "saved_candidate_count": len(probes), "methods": results})
            print(json.dumps({"event": "offline_replay_question", "development_id": dev,
                "methods": {label: {"probes": r["n_probes"], "stop_candidate": r["stop_candidate_j"],
                    "skipped": r["skipped_candidate_j"], "partial_token_delta": r["versus_dense"]["partial_main_plus_probe_difference"]}
                    for label, r in results.items()}}, ensure_ascii=False), flush=True)
        archived.check("complete_replay_denominator", len(report["questions"]) == 10)
        report["summary"] = {label: {
            "question_count": 10,
            "n_probes": sum(q["methods"][label]["n_probes"] for q in report["questions"]),
            "actual_skipped_candidates": sum(len(q["methods"][label]["skipped_candidate_j"]) for q in report["questions"]),
            "main_tokens": sum(q["methods"][label]["partial_tokens"]["main_generated_including_pending_wait"] for q in report["questions"]),
            "probe_tokens": sum(q["methods"][label]["partial_tokens"]["probe_generated"] for q in report["questions"]),
            "partial_main_plus_probe": sum(q["methods"][label]["partial_tokens"]["main_plus_probe"] for q in report["questions"]),
            "answers_available": sum(q["methods"][label]["answer_availability"]["status"] == "existing_answer_available" for q in report["questions"]),
            "new_endpoint_without_archived_answer": sum(q["methods"][label]["answer_availability"]["status"] == "unknown" for q in report["questions"]),
            "quality_not_evaluated": 10, "online_latency_estimate_ms": None,
        } for label in LABELS}
    except (Exception, KeyboardInterrupt) as error:
        archived.checks.append({"check": "replay_exception", "passed": False,
            "error_type": type(error).__name__, "message": str(error)})
    finally:
        archive_after, source_after = archived.finish(), sources.finish()
    checks = archived.checks + sources.checks
    report.update(status="passed" if all(c["passed"] for c in checks) else "failed", check_count=len(checks),
        checks=checks, failed_checks=[c for c in checks if not c["passed"]],
        archived_input_sha256_before=archived.hashes, archived_input_sha256_after=archive_after,
        current_source_sha256_before=sources.hashes, current_source_sha256_after=source_after,
        limitations=["Ten already exposed development questions, one seed; observations are not independent problems.",
            "Archive hashes bind old evidence; separate current hashes identify a counterfactual policy, not GPU validation.",
            "Only actual queried observations enter controller history; no terminal probe or unavailable final answer is invented.",
            "Saved dense intervals are controller inputs, not sparse online latency estimates.",
            "Partial token totals exclude forced finalization; answer availability does not establish correctness.",
            "No gold, grading or parameter selection is used. Existing online method results remain unchanged."])
    return report


def write_report(report, output):
    output = Path(output).resolve()
    if output.exists() or not output.is_relative_to(WORK_ROOT.resolve()):
        raise ValueError("Output must be a new directory under runs/guarded-candidate-20261009")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "replay.json").open("xb") as stream:
        stream.write(encoded(report))
    with (output / "OUTPUT_SHA256.json").open("xb") as stream:
        stream.write(encoded({"replay.json": digest(output / "replay.json")}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-run", type=Path, default=ROOT / "runs/dense10-20261008/remote-a09-final/runs/dense10-001")
    parser.add_argument("--development-run", type=Path, default=ROOT / "runs/development10-20261008/remote-a08-final/runs/development10-001")
    parser.add_argument("--prior-analysis", type=Path, default=ROOT / "runs/dense10-20261008/analysis-001/analysis.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze(args.collection_run, args.development_run, args.prior_analysis)
    write_report(report, args.output_dir)
    print(json.dumps({"status": report["status"], "checks": report["check_count"],
        "failed_checks": report["failed_checks"], "summary": report.get("summary"),
        "output": str(args.output_dir)}, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
