#!/usr/bin/env python3
"""Bounded five-prefix validation under an explicit same-KV reference contract.

The common probe must exactly match the pinned original probe given the same
incrementally built main KV. The original full-prefix probe is run separately
as a cross-path sensitivity measurement, never an equivalence acceptance gate.
The previous full-prefix exact failure is retained as historical evidence.

Saved prefixes are teacher-forced, with one growing main KV per question.
No saved question is regenerated or searched for an answer. Per-question dense
rule histories remain separate. This is five checkpoints across two exposed
questions, not an accuracy, speed, or full-capacity experiment.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import diagnose_probe_parity as parity
import validate_online_gpu as validation
from online_protocol import (
    ProbeObservation, ProtocolConfig, candidate_threshold, degeneration_score, stopping_reasons,
)
from run_online_diagnostic import assert_gpu_idle, atomic_new, gpu_inventory, gpu_lock

SELECTORS = ("q001:0", "q001:1", "q002:0", "q002:1", "q002:2")
ISOLATION_SELECTOR = "q002:0"
REFERENCE_CONTRACT = "incremental_main_KV_pinned_original_probe_exact_v1"
SCOPE = "five_saved_prefix_same_KV_validation_with_cross_path_sensitivity"
SOURCE_FILES = ("scripts/validate_online_continuation.py", "scripts/diagnose_probe_parity.py",
                "scripts/validate_online_gpu.py", "scripts/run_online_diagnostic.py",
                "src/online_engine.py", *parity.CRITICAL_SOURCE_FILES)
DEER_GRID = (0.85, 0.87, 0.89, 0.91, 0.93, 0.95, 0.97, 0.98, 0.99)
CODESTOP_RMAX_GRID = (0.90, 0.95, 0.98)
CODESTOP_TAU_GRID = (1.0, 2.0, 4.0)


def load_cases(evidence_root):
    # Keep the old helper's three-prefix limit unchanged. This new plan explicitly
    # fixes five inputs and checks each selector through that existing reader.
    return [validation.load_saved_prefixes(evidence_root, [selector])[0] for selector in SELECTORS]


def load_identity(identity_root, cases):
    root = Path(identity_root).resolve()
    records, hashes = {}, {}
    for filename in ("plan.json", "environment.json", "source-hashes.json", "summary.json"):
        raw = (root / filename).read_bytes()
        records[filename], hashes[filename] = json.loads(raw), validation.sha(raw)
    plan, summary = records["plan.json"], records["summary.json"]
    if plan.get("scope") != parity.SCOPE or summary.get("status") != "diagnostic_completed":
        raise ValueError("Identity root must be a completed saved parity diagnostic")
    if plan.get("prefix_ids_sha256") != cases[0]["prefix_ids_sha256"]:
        raise ValueError("Prior diagnostic used a different first saved prefix")
    for filename in parity.CRITICAL_SOURCE_FILES:
        if records["source-hashes.json"].get(filename) != validation.sha((ROOT / filename).read_bytes()):
            raise ValueError(f"Backend/protocol changed since the recorded diagnostic: {filename}")
    if summary.get("same_KV_parity", {}).get("status") != "passed":
        raise ValueError("Prior diagnostic did not establish its single-prefix same-KV comparison")
    attention, seed = plan.get("attention_implementation"), plan.get("seed")
    if attention not in ("eager", "sdpa") or type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Prior diagnostic lacks its frozen attention implementation or seed")
    return {"root": str(root), "records": records, "sha256": hashes,
            "attention": attention, "seed": seed}


def observation_from_record(record, eos_ids, source):
    values = parity.comparable(record)
    return ProbeObservation(tuple(values["token_ids"]), tuple(values["token_probs"]),
                            values["confidence"], values["ended"], confidence_source=source,
                            invalid_reason="probe_generated_eos" if any(
                                token in eos_ids for token in values["token_ids"]) else None)


def candidate_identity(case, backend):
    """Distinguish a numerical prefix fixture from an actual online Wait event."""
    response = backend.tokenizer(case["sample_response"])["input_ids"]
    position = case["stop_token_idx"]
    pending = response[position]
    prefix = response[:position]
    exits = [index for index, token in enumerate(prefix)
             if token == backend.markers.end_think or token in backend.markers.eos_ids]
    reasons = []
    if pending in backend.markers.eos_ids:
        reasons.append("pending_EOS_terminates_without_online_probe")
    elif pending != backend.markers.wait:
        reasons.append("pending_token_is_not_Wait")
    if exits:
        reasons.append("prefix_already_left_reasoning_or_terminated")
    eligible = not reasons
    return {"pending_token_id": pending, "pending_is_Wait": pending == backend.markers.wait,
            "pending_is_EOS": pending in backend.markers.eos_ids,
            "prior_reasoning_exit_or_EOS_positions": exits,
            "online_candidate_eligible": eligible, "online_candidate_j": prefix.count(backend.markers.wait) + 1 if eligible else None,
            "exclusion_reasons": reasons}


class DenseRuleHistory:
    """Actual production arithmetic, with a separate dense history per question/path."""

    def __init__(self):
        self.config = ProtocolConfig()
        self.histories, self.last_candidate = {}, {}

    def observe(self, case, path, observation):
        key = (case["question"], path)
        identity = case.get("candidate_identity")
        if identity is not None and not identity["online_candidate_eligible"]:
            return {"online_candidate_eligible": False, "candidate_j": None,
                    "token_position": case["stop_token_idx"], "confidence_raw": observation.confidence_raw,
                    "confidence_valid": observation.confidence_valid, "invalid_reasons": list(observation.invalid_reasons),
                    "ended_with_think": observation.ended_with_think, "D": None,
                    "confidence_stop": False, "degeneration_stop": False, "combined_stop": False,
                    "stop_reasons": [], "stop_applied": False,
                    "exclusion_reasons": identity["exclusion_reasons"],
                    "purpose": "numerical fixture only; not entered into online Wait history"}
        candidate = identity["online_candidate_j"] if identity else case["probe_index"] + 1
        if candidate != self.last_candidate.get(key, 0) + 1:
            raise ValueError("Dense stopping-rule diagnostics require consecutive per-question candidates")
        history = self.histories.setdefault(key, [])
        position = case["stop_token_idx"]
        if history and position <= history[-1][0]:
            raise ValueError("Dense history positions must increase within each question")
        self.last_candidate[key] = candidate
        threshold = candidate_threshold(candidate, self.config)
        score = None
        if observation.confidence_valid:
            history.append((position, observation.confidence_raw))
            score = degeneration_score([item[0] for item in history], [item[1] for item in history])
        reasons = stopping_reasons(observation, score, threshold, self.config)
        return {"online_candidate_eligible": True, "candidate_j": candidate, "token_position": position,
                "confidence_raw": observation.confidence_raw, "confidence_valid": observation.confidence_valid,
                "invalid_reasons": list(observation.invalid_reasons), "ended_with_think": observation.ended_with_think,
                "confidence_threshold": threshold, "degeneration_threshold": self.config.tau,
                "D": score, "valid_history_count": len(history),
                "valid_history_positions": [item[0] for item in history],
                "valid_history_confidences": [item[1] for item in history],
                "confidence_stop": "confidence" in reasons, "degeneration_stop": "degeneration" in reasons,
                "combined_stop": bool(reasons), "stop_reasons": list(reasons),
                "stop_applied": False,
                "purpose": "dense saved-prefix decision diagnostic; continue after any would-stop"}


def advance_saved_prefix(backend, case, previous, state, run_root, seed):
    """Keep the accepted main KV growing only along verified saved token IDs."""
    if previous is None or previous["question"] != case["question"]:
        backend.start_request(seed)
        return validation.rebuild_online_prefix(backend, case, run_root)
    previous_ids, current_ids = previous["prefix_ids"], case["prefix_ids"]
    if (case["prompt_ids"] != previous["prompt_ids"] or current_ids[:len(previous_ids)] != previous_ids
            or len(current_ids) <= len(previous_ids) or state.length != len(previous_ids)):
        raise ValueError("Successive saved prefixes do not extend the same main KV trajectory")
    for index, token in enumerate(current_ids[len(previous_ids):], start=1):
        state = backend.extend(state, (token,))
        if index % 256 == 0:
            validation.event(run_root, "saved_prefix_replay_progress", case=case["name"],
                             additional_saved_tokens=index, accepted_prefix_tokens=state.length)
    return state


def original_record(result):
    return {"token_ids": result["token_ids"], "token_probs": result["token_probs"],
            "confidence": result["total_prob_max"], "ended": bool(result["ended_with_think"])}


def evaluate_prefix(backend, reference, case, state, run_root, history):
    """A/D are the acceptance pair; C is a separate measured sensitivity path."""
    name = case["name"]
    before = validation.state_fingerprint(backend, state)
    validation.event(run_root, "probe_path_start", case=name, path="online_incremental_KV")
    backend.synchronize()
    start = time.perf_counter()
    actual_observation = backend.probe(state)
    backend.synchronize()
    actual = parity.record_path(backend, run_root, f"{name}-online", parity.pack_common(actual_observation),
        seconds=time.perf_counter() - start, before=before, after=validation.state_fingerprint(backend, state),
        reference_contract=REFERENCE_CONTRACT)

    validation.event(run_root, "probe_path_start", case=name, path="pinned_original_same_KV")
    backend.synchronize()
    start = time.perf_counter()
    result = reference(backend.model,
        backend.torch.tensor([list(backend.markers.trial_prefix)], device=backend.device),
        state.cache, backend.tokenizer, method=0)
    backend.synchronize()
    same_kv = parity.record_path(backend, run_root, f"{name}-same-KV", original_record(result),
        seconds=time.perf_counter() - start, before=before, after=validation.state_fingerprint(backend, state),
        cache_policy="pinned original deepcopy and cache_to_device on the exact online main KV")
    gate = parity.compare_paths(same_kv, actual, "pinned_original_same_KV", "online_incremental_KV")
    validation.write_json(run_root / f"{name}-same-KV-gate.json", gate)
    validation.fail_on_difference(run_root, same_kv, actual, phase="same_KV_probe_numerics", case=name)

    validation.event(run_root, "probe_path_start", case=name, path="pinned_original_full_prefix_sensitivity")
    full = case["prefix_ids"] + list(backend.markers.trial_prefix)
    backend.synchronize()
    start = time.perf_counter()
    result = reference(backend.model, backend.torch.tensor([full], device=backend.device),
                       None, backend.tokenizer, method=0)
    backend.synchronize()
    full_prefix = parity.record_path(backend, run_root, f"{name}-full-prefix", original_record(result),
        seconds=time.perf_counter() - start, before=before, after=validation.state_fingerprint(backend, state),
        scope="cross-path sensitivity only; not the acceptance reference")
    cross = parity.compare_paths(same_kv, full_prefix, "pinned_original_same_KV", "pinned_original_full_prefix")
    cross["status"] = "identical" if cross.pop("exact") else "different"
    cross["is_acceptance_gate"] = False
    cross["interpretation"] = "An observed path difference does not establish its numerical cause or equivalence."

    decisions = {}
    for path, record in (("online", actual), ("same_KV", same_kv), ("full_prefix", full_prefix)):
        observation = observation_from_record(record, backend.markers.eos_ids, f"raw_{path}_BF16_probe")
        decisions[path] = history.observe(case, path, observation)
    validation.write_json(run_root / f"{name}-stopping-decisions.json", decisions)
    validation.fail_on_difference(run_root, decisions["same_KV"], decisions["online"],
                                  phase="same_KV_dense_stop_rules", case=name)
    cross["decision_differences"] = {
        field: {"same_KV": decisions["same_KV"][field], "full_prefix": decisions["full_prefix"][field]}
        for field in ("confidence_raw", "D", "confidence_valid", "ended_with_think", "confidence_stop",
                      "degeneration_stop", "combined_stop")
        if validation.first_difference(decisions["same_KV"][field], decisions["full_prefix"][field]) is not None}
    validation.write_json(run_root / f"{name}-cross-path-sensitivity.json", cross)
    validation.event(run_root, "prefix_completed", case=name, same_KV_gate=gate["status"], cross_path=cross["status"],
                     D_same_KV=decisions["same_KV"]["D"], D_full_prefix=decisions["full_prefix"]["D"],
                     same_KV_stop=decisions["same_KV"]["stop_reasons"],
                     full_prefix_stop=decisions["full_prefix"]["stop_reasons"])
    return {"case": name, "question": case["question"], "same_KV_gate": gate,
            "candidate_identity": case.get("candidate_identity"),
            "raw_observations": {"online": actual, "same_KV": same_kv, "full_prefix": full_prefix},
            "cross_path_sensitivity": cross, "stopping_decisions": decisions}


def threshold_sensitivity(completed, eos_ids):
    """Reuse measured raw probes and their per-question dense D; no model calls."""
    configs = [ProtocolConfig(r_max=r_max, tau=tau) for r_max in CODESTOP_RMAX_GRID for tau in CODESTOP_TAU_GRID]
    configs += [ProtocolConfig(rule="deer", deer_threshold=threshold) for threshold in DEER_GRID]
    entries = []
    for config in configs:
        rows, first_stops = [], {}
        for result in completed:
            by_path = {}
            question_stops = first_stops.setdefault(result["question"], dict.fromkeys(("online", "same_KV", "full_prefix")))
            for path, record in result["raw_observations"].items():
                dense = result["stopping_decisions"][path]
                eligible = dense["online_candidate_eligible"]
                threshold = candidate_threshold(dense["candidate_j"], config) if eligible else None
                observation = observation_from_record(record, eos_ids, f"measured_{path}_reused_for_threshold_grid")
                reasons = stopping_reasons(observation, dense["D"], threshold, config) if eligible else ()
                by_path[path] = {"online_candidate_eligible": eligible, "confidence_threshold": threshold,
                    "D": dense["D"], "confidence_stop": "confidence" in reasons,
                    "degeneration_stop": "degeneration" in reasons, "combined_stop": bool(reasons)}
                if reasons and question_stops[path] is None:
                    question_stops[path] = result["case"]
            rows.append({"case": result["case"], "question": result["question"], "by_path": by_path,
                "cross_path_stop_flags_differ": any(by_path["same_KV"][field] != by_path["full_prefix"][field]
                    for field in ("confidence_stop", "degeneration_stop", "combined_stop"))})
        entries.append({"config": asdict(config), "rows": rows, "first_would_stop_by_question": first_stops,
                        "cross_path_changed_checkpoint_count": sum(row["cross_path_stop_flags_differ"] for row in rows)})
    return {"scope": "offline_threshold_sensitivity_on_five_saved_checkpoints_only",
            "new_GPU_calls": 0, "same_KV_acceptance_gate_changed": False,
            "history_policy": "per-question dense measured history; excludes EOS and post-reasoning Wait",
            "DEER_thresholds": list(DEER_GRID), "CoDE_rmax": list(CODESTOP_RMAX_GRID), "CoDE_tau": list(CODESTOP_TAU_GRID),
            "entries": entries, "limitation": "No quality or latency conclusion; unobserved candidates are not imputed."}


def input_integrity_paths(cases, identity):
    expected = {str(ROOT / filename): validation.sha((ROOT / filename).read_bytes()) for filename in SOURCE_FILES}
    for case in cases:
        expected.update(case["input_sha256"])
    expected.update({str(Path(identity["root"]) / name): digest for name, digest in identity["sha256"].items()})
    return expected


def verify_integrity(run_root, expected):
    actual = {path: validation.sha(Path(path).read_bytes()) for path in expected}
    validation.write_json(run_root / "integrity-after.json", actual)
    validation.fail_on_difference(run_root, expected, actual, phase="source_input_identity_integrity", case="entire_run")


def budget(cases):
    final_positions = {}
    for case in cases:
        final_positions[case["question"]] = case["stop_token_idx"]
    isolation = next(case for case in cases if case["name"] == "q002-probe-0")
    return {"fixed_selectors": list(SELECTORS), "prefix_count": 5, "question_count": 2,
            "comparison_probes": 15, "isolation_probes": 3, "probe_token_cap_each": 21,
            "maximum_new_probe_tokens": 378, "maximum_new_continuation_tokens": 192,
            "comparison_saved_token_extends": sum(final_positions.values()),
            "isolation_saved_token_extends": 3 * isolation["stop_token_idx"],
            "isolation_prefix": ISOLATION_SELECTOR, "isolation_continuation_cap_each": 64}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model-dir", type=Path)
    result.add_argument("--evidence-root", type=Path, required=True)
    result.add_argument("--identity-root", type=Path, required=True, help="Prior completed same-KV parity diagnostic run")
    result.add_argument("--upstream-source-dir", type=Path)
    result.add_argument("--run-root", type=Path)
    result.add_argument("--inspect-only", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    cases = load_cases(args.evidence_root)
    identity = load_identity(args.identity_root, cases)
    raw_probe, probe_function, probe_metadata = validation.reference_source(args.upstream_source_dir)
    raw_cache, cache_function, cache_metadata = parity.cache_source(args.upstream_source_dir)
    plan = {"scope": SCOPE, "reference_contract": REFERENCE_CONTRACT,
            "acceptance_gate": "exact same incremental main KV original-probe parity plus KV/RNG isolation",
            "cross_path_policy": "full-prefix outputs are sensitivity evidence, not an equivalence acceptance gate",
            "tolerance": validation.TOLERANCE, "budget": budget(cases),
            "prefixes": [{key: case[key] for key in ("name", "question", "probe_index", "stop_token_idx",
                                                    "prefix_ids_sha256", "input_sha256")} for case in cases],
            "reference_sources": [probe_metadata, cache_metadata], "seed": identity["seed"],
            "attention_implementation": identity["attention"], "prior_identity_root": identity["root"],
            "prior_identity_sha256": identity["sha256"], "stop_rule_config": asdict(ProtocolConfig()),
            "stop_rule_history": "separate per question and path; all five saved observations collected despite would-stop",
            "D_arithmetic": "online_protocol.degeneration_score Python float; pinned code-log definition",
            "threshold_sensitivity": {"DEER": list(DEER_GRID), "CoDE_rmax": list(CODESTOP_RMAX_GRID),
                                      "CoDE_tau": list(CODESTOP_TAU_GRID), "additional_GPU_calls": 0},
            "coverage_limit": "q001 has two points and D=0; q002 has three points; no accuracy/speed/32K claim"}
    if args.inspect_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.model_dir is None or args.run_root is None:
        raise ValueError("Execution requires --model-dir and a new --run-root")
    for protected in (args.evidence_root, args.identity_root, args.model_dir):
        if args.run_root.resolve().is_relative_to(protected.resolve()):
            raise ValueError("New output directory must be outside read-only evidence, identity, and model directories")
    with validation.fresh_run(args.run_root) as run_root:
        validation.write_json(run_root / "plan.json", plan)
        validation.write_json(run_root / "inputs.json", cases)
        validation.write_json(run_root / "prior-identity.json", identity["records"])
        validation.write_json(run_root / "source-hashes.json", {
            filename: validation.sha((ROOT / filename).read_bytes()) for filename in SOURCE_FILES})
        expected_integrity = input_integrity_paths(cases, identity)
        validation.write_json(run_root / "integrity-before.json", expected_integrity)
        atomic_new(run_root / "pinned-probe.py", raw_probe, raw=True)
        atomic_new(run_root / "pinned-cache-utils.py", raw_cache, raw=True)
        completed = []
        try:
            inventory = gpu_inventory()
            with gpu_lock(inventory["uuid"]) as lock_path:
                idle = assert_gpu_idle(inventory["uuid"])
                validation.write_json(run_root / "gpu-preflight.json", {"gpu": inventory, "lock": lock_path, "idle": idle})
                validation.event(run_root, "model_load_start", gpu_uuid=inventory["uuid"], attention=identity["attention"])
                from torch_online_backend import TorchOnlineBackend
                backend = TorchOnlineBackend(args.model_dir, attention_implementation=identity["attention"])
                validation.write_json(run_root / "environment.json", backend.metadata)
                parity.verify_environment(identity["records"]["environment.json"], backend.metadata)
                for case in cases:
                    validation.validate_tokenizer_prefix(case, backend)
                    case["candidate_identity"] = candidate_identity(case, backend)
                validation.write_json(run_root / "candidate-identities.json", {
                    case["name"]: case["candidate_identity"] for case in cases})
                reference = parity.load_cached_reference(probe_function, cache_function, backend.torch)
                state, previous, history = None, None, DenseRuleHistory()
                for case in cases:
                    validation.event(run_root, "saved_prefix_start", case=case["name"], prefix_tokens=len(case["prefix_ids"]))
                    state = advance_saved_prefix(backend, case, previous, state, run_root, identity["seed"])
                    completed.append(evaluate_prefix(backend, reference, case, state, run_root, history))
                    previous = case
                del state
                sensitivity = threshold_sensitivity(completed, backend.markers.eos_ids)
                validation.write_json(run_root / "threshold-sensitivity.json", sensitivity)
                for entry in sensitivity["entries"]:
                    for row in entry["rows"]:
                        validation.fail_on_difference(run_root, row["by_path"]["same_KV"], row["by_path"]["online"],
                                                      phase="same_KV_threshold_grid_rules", case=row["case"])
                isolation_case = cases[SELECTORS.index(ISOLATION_SELECTOR)]
                validation.event(run_root, "isolation_start_on_third_prefix", case=isolation_case["name"], continuation_cap=64)
                isolation = validation.validate_isolation(backend, isolation_case, run_root, identity["seed"], 64)
                verify_integrity(run_root, expected_integrity)
                summary = {"schema_version": 1, "gate_passed": True,
                    "status": "passed_same_KV_contract_bounded_validation", "scope": SCOPE,
                    "reference_contract": REFERENCE_CONTRACT, "completed_prefixes": completed,
                    "isolation": {"case": isolation_case["name"], "status": "passed", "variants": isolation},
                    "full_prefix_equivalence": "not_established; former exact failure is retained",
                    "cross_path_different_count": sum(row["cross_path_sensitivity"]["status"] == "different" for row in completed),
                    "online_eligible_checkpoint_count": sum(case["candidate_identity"]["online_candidate_eligible"] for case in cases),
                    "threshold_sensitivity_file": "threshold-sensitivity.json", "source_input_identity_integrity": "unchanged",
                    "coverage": "five saved checkpoints across two exposed questions; no benchmark conclusion",
                    "answer_boundary_coverage": "not performed", "capacity_32k": "not performed",
                    "timing_interpretation": "diagnostic overhead included; no speed claim"}
                validation.write_json(run_root / "summary.json", summary)
                validation.event(run_root, "bounded_same_KV_validation_completed", prefixes=len(completed),
                                 cross_path_different_count=summary["cross_path_different_count"], isolation="passed",
                                 reference_contract=REFERENCE_CONTRACT)
                return 0
        except (Exception, KeyboardInterrupt) as exc:
            validation.write_json(run_root / "failure.json", {"schema_version": 1, "gate_passed": False,
                "status": "failed_or_interrupted", "scope": SCOPE,
                "reference_contract": REFERENCE_CONTRACT, "completed_prefixes": completed,
                "error_type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
            validation.event(run_root, "bounded_validation_failed", completed_prefixes=len(completed),
                             error_type=type(exc).__name__, message=str(exc))
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
