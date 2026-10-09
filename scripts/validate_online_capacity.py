#!/usr/bin/env python3
"""Gated CUDA memory/shape diagnostic using an explicitly synthetic 32K prefix.

Repeating already saved q002 tokens is not natural 32K generation, a numerical
parity check, a correctness experiment, or a speed measurement. The model stays
BF16/eager; no precision fallback, automatic smaller-chunk retry, or download is
allowed. Existing evidence and run directories are never overwritten.
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
sys.path.insert(0, str(ROOT / "src"))
from online_contract import MODEL_REVISION, RUNNER_PROTOCOL
from online_engine import FINAL_TOKEN_CAP
from online_protocol import MAX_PROBE_TOKENS
from online_source_manifest import METHOD_IDENTITY_FILES
from run_online_diagnostic import (
    EventLog, assert_gpu_idle, atomic_new, gpu_inventory, gpu_lock, sha256,
)
from validate_online_gpu import first_difference, state_fingerprint
from diagnose_probe_parity import ENVIRONMENT_FIELDS, verify_environment

SCOPE = "synthetic_capacity_only_not_natural_generation_accuracy_or_speed"
DATASET_SCOPE = "single_question_development_gpu_acceptance_not_benchmark"
NATURAL_SYNTHETIC_QUESTION_SCOPE = "real_model_natural_boundary_validation_on_synthetic_question"
SAMPLE_ID = "math/train/geometry/428"
SOURCE_SHA256 = "542987a6a291243ac13178f746d94bb879c06ba4f96332b121e4c7a6cf35eccf"
MAIN_TOKEN_BUDGET = 32768
REFERENCE_CONTRACT = "incremental_main_KV_pinned_original_probe_exact_v1"
GATE_CASES = ("q001-probe-0", "q001-probe-1", "q002-probe-0", "q002-probe-1", "q002-probe-2")
BRANCHES = ("actual_early_stop", "actual_natural_eos", "actual_answer_phase_budget")
CRITICAL_SOURCES = ("src/online_contract.py", "src/online_protocol.py", "src/torch_online_backend.py",
                    *METHOD_IDENTITY_FILES)
SOURCE_FILES = ("scripts/validate_online_capacity.py", "scripts/validate_online_gpu.py",
                "scripts/diagnose_probe_parity.py",
                "scripts/run_online_diagnostic.py", "src/online_engine.py", *CRITICAL_SOURCES)


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    def reject(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=unique, parse_constant=reject)


def token_ids(value, label):
    if not isinstance(value, list) or not value or any(type(x) is not int or x < 0 for x in value):
        raise ValueError(f"{label} must be nonempty nonnegative integer IDs")
    return list(value)


def load_source(path, *, expected_sha256=SOURCE_SHA256):
    path = Path(path).resolve()
    if sha256(path) != expected_sha256:
        raise ValueError("Saved q002 Vanilla source does not match the frozen SHA-256")
    record = read_json(path)
    if (record.get("sample_id") != SAMPLE_ID or record.get("status") != "completed"
            or record.get("method") != "vanilla" or record.get("max_new_tokens") != 1024
            or record.get("backend_metadata", {}).get("model_revision") != MODEL_REVISION):
        raise ValueError("Expected the frozen completed q002 1024-token Vanilla request")
    prompt = token_ids(record["prompt_token_ids"], "prompt")
    samples = record["main_samples"]
    if len(samples) != 1024 or any(item.get("phase") != "reason" for item in samples):
        raise ValueError("The capacity pattern must contain exactly 1024 saved thinking tokens")
    pattern = token_ids([item["token_id"] for item in samples], "saved thinking pattern")
    if record["output_token_ids"][:1024] != pattern:
        raise ValueError("Saved main samples differ from the accepted output prefix")
    return {"path": str(path), "sha256": expected_sha256, "sample_id": SAMPLE_ID,
            "prompt_ids": prompt, "pattern_ids": pattern,
            "pattern_origin": "saved accepted q002 thinking tokens, repeated synthetically"}


def check_source_hashes(values, names=CRITICAL_SOURCES):
    if not isinstance(values, dict):
        raise ValueError("Missing source identity map")
    for name in names:
        if values.get(name) != sha256(ROOT / name):
            raise ValueError(f"Prerequisite source differs from current execution source: {name}")


def validate_gate(path):
    path = Path(path).resolve()
    gate = read_json(path)
    if (gate.get("gate_passed") is not True or gate.get("schema_version") != 1
            or gate.get("status") != "passed_same_KV_contract_bounded_validation"
            or gate.get("reference_contract") != REFERENCE_CONTRACT
            or gate.get("source_input_identity_integrity") != "unchanged"):
        raise ValueError("A passed continuation gate under the explicit same-KV contract is required")
    rows = gate.get("completed_prefixes", [])
    if ([item.get("case") for item in rows] != list(GATE_CASES)
            or any(item.get("same_KV_gate", {}).get("status") != "passed" for item in rows)):
        raise ValueError("Continuation gate must pass all five fixed saved prefixes")
    isolation = gate.get("isolation", {})
    variants = isolation.get("variants", [])
    lengths = [item.get("continuation_tokens") for item in variants]
    if (isolation.get("status") != "passed" or isolation.get("case") != "q002-probe-0"
            or [item.get("inserted_probes") for item in variants] != [0, 1, 2]
            or any(item.get("exact") is not True for item in variants)
            or any(type(length) is not int or not 1 <= length <= 64 for length in lengths)
            or len(set(lengths)) != 1):
        raise ValueError("Continuation gate must include exact q002 0/1/2-probe bounded isolation")
    source_path = path.parent / "source-hashes.json"
    check_source_hashes(read_json(source_path), (*CRITICAL_SOURCES, "src/online_engine.py"))
    environment_path = path.parent / "environment.json"
    environment = read_json(environment_path)
    if any(field not in environment for field in ENVIRONMENT_FIELDS):
        raise ValueError("Continuation gate lacks complete frozen backend environment metadata")
    before_path, after_path = path.parent / "integrity-before.json", path.parent / "integrity-after.json"
    if read_json(before_path) != read_json(after_path):
        raise ValueError("Continuation gate input/source integrity records differ")
    evidence_paths = (path, source_path, environment_path, before_path, after_path)
    return {"path": str(path), "sha256": sha256(path), "reference_contract": REFERENCE_CONTRACT,
            "source_hashes_sha256": sha256(source_path), "gate_passed": True,
            "isolation_continuation_tokens": lengths[0], "isolation_cap": 64,
            "input_sha256": {str(item): sha256(item) for item in evidence_paths},
            "environment": {field: environment[field] for field in ENVIRONMENT_FIELDS},
            "old_full_prefix_gate": "not replaced by this same-KV contract"}


def validate_natural_runs(roots, *, independent=False):
    covered, evidence, input_hashes = set(), [], {}
    coverage_by_origin = {"frozen_dataset": set(), "synthetic_question_real_model": set()}
    for root in roots:
        root = Path(root).resolve()
        summary_path, manifest_path = root / "summary.json", root / "manifest.json"
        summary, manifest = read_json(summary_path), read_json(manifest_path)
        input_hashes.update({str(path): sha256(path) for path in (summary_path, manifest_path)})
        if (summary.get("status") != "completed" or summary.get("unexecuted_count") != 0
                or summary.get("executed_count", 0) < 1
                or summary.get("code_unchanged_during_run") is not True):
            raise ValueError("Natural branch evidence requires a complete, unchanged-source run")
        check_source_hashes(manifest.get("code_sha256"), (*CRITICAL_SOURCES, "src/online_engine.py"))
        rows = summary.get("requests", [])
        if len(rows) != summary["executed_count"]:
            raise ValueError("Natural run request count is inconsistent")
        for row in rows:
            label = row.get("configuration")
            if not isinstance(label, str) or not label or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in label):
                raise ValueError("Unsafe natural-run configuration label")
            source_path = root / f"{label}.json"
            record = read_json(source_path)
            if sha256(source_path) != row.get("source_sha256"):
                raise ValueError("Natural request source hash differs from its summary")
            input_hashes[str(source_path)] = row["source_sha256"]
            metadata = record.get("backend_metadata", {})
            scope = record.get("scope")
            synthetic_question = scope == NATURAL_SYNTHETIC_QUESTION_SCOPE
            if (record.get("status") != "completed" or record.get("configuration") != label
                    or scope not in (DATASET_SCOPE, NATURAL_SYNTHETIC_QUESTION_SCOPE)
                    or ("forced_tokens" in record and record["forced_tokens"] is not False)
                    or (synthetic_question and (record.get("forced_tokens") is not False
                                               or record.get("synthetic_question") is not True))
                    or record.get("validation_only") or metadata.get("validation_model") is not False
                    or metadata.get("model_revision") != MODEL_REVISION
                    or metadata.get("attention_implementation") != "eager"
                    or metadata.get("model_parameter_dtype") != "torch.bfloat16"):
                raise ValueError("Natural coverage cannot be established by synthetic/incompatible requests")
            matched = []
            stop = record.get("stop_reason")
            boundary = record.get("answer_boundary", {})
            if (scope == DATASET_SCOPE and record.get("sample_id") == SAMPLE_ID
                    and stop in ("confidence", "degeneration", "confidence_and_degeneration") and any(
                    item.get("stop_applied") is True for item in record.get("probes", []))):
                matched.append("actual_early_stop")
            if (stop == "natural_eos" and record.get("finalization_tokens") == 0
                    and record.get("output_token_ids", [])
                    and record["output_token_ids"][-1] in metadata.get("markers", {}).get("eos_ids", [])):
                matched.append("actual_natural_eos")
            if (stop == "budget" and boundary.get("source") == "natural_first_end_think"
                    and record.get("injected_prompt_tokens") == 0
                    and 0 < record.get("finalization_tokens", 0) <= FINAL_TOKEN_CAP):
                matched.append("actual_answer_phase_budget")
            covered.update(matched)
            origin = "synthetic_question_real_model" if synthetic_question else "frozen_dataset"
            coverage_by_origin[origin].update(matched)
            evidence.append({"path": str(source_path), "sha256": row["source_sha256"],
                             "covered_branches": matched, "stop_reason": stop,
                             "question_origin": origin, "scope": scope, "sample_id": record.get("sample_id")})
    missing = [name for name in BRANCHES if name not in covered]
    if missing and not independent:
        raise ValueError("Missing actual natural-branch coverage: " + ", ".join(missing)
                         + "; independent capacity diagnostics must be explicitly selected")
    return {"covered": sorted(covered), "missing": missing, "gate_passed": not missing,
            "independent_capacity_diagnostic": independent, "evidence": evidence,
            "input_sha256": input_hashes,
            "coverage_by_origin": {origin: sorted(branches) for origin, branches in coverage_by_origin.items()},
            "interpretation": "Real-model branch coverage can combine dataset and synthetic questions; synthetic-question EOS does not establish dataset EOS coverage."}


def memory_snapshot(backend, state=None):
    backend.synchronize()
    cuda, device = backend.torch.cuda, backend.device
    free, total = cuda.mem_get_info(device)
    result = {"allocated_bytes": int(cuda.memory_allocated(device)),
              "reserved_bytes": int(cuda.memory_reserved(device)),
              "peak_allocated_bytes": int(cuda.max_memory_allocated(device)),
              "peak_reserved_bytes": int(cuda.max_memory_reserved(device)),
              "device_free_bytes": int(free), "device_total_bytes": int(total)}
    if state is not None:
        result.update(kv_length=int(state.cache.get_seq_length()), committed_state_length=state.length,
                      kv_tensor_bytes=sum(value.numel() * value.element_size()
                                          for pair in state.cache for value in pair),
                      kv_layer_shapes=[[list(value.shape) for value in pair] for pair in state.cache],
                      kv_lengths_consistent=all(value.shape[-2] == state.length
                                                for pair in state.cache for value in pair))
    return result


def run_capacity(args, *, backend_factory=None):
    if args.chunk_size not in (128, 256) or not 1 <= args.wall_time_seconds <= 3600 or not 0 <= args.seed < 2**63:
        raise ValueError("Require chunk-size 128/256, wall-time 1..3600 seconds, and seed in [0,2**63)")
    source = load_source(args.source_record)
    gate = validate_gate(args.gate_summary)
    coverage = validate_natural_runs(args.natural_run_root,
                                     independent=args.independent_capacity_diagnostic)
    plan = {"schema_version": 1, "scope": SCOPE, "runner_protocol": RUNNER_PROTOCOL,
            "model_revision": MODEL_REVISION, "attention_implementation": "eager", "dtype": "BF16",
            "main_token_budget": MAIN_TOKEN_BUDGET, "chunk_size": args.chunk_size,
            "wall_time_seconds": args.wall_time_seconds, "seed": args.seed,
            "prompt_tokens": len(source["prompt_ids"]), "saved_pattern_tokens": len(source["pattern_ids"]),
            "source": {k: v for k, v in source.items() if k not in ("prompt_ids", "pattern_ids")},
            "prerequisite_gate": gate, "natural_branch_coverage": coverage,
            "probe_cap": MAX_PROBE_TOKENS, "greedy_answer_cap": FINAL_TOKEN_CAP,
            "protocol_limits": "reserve=max(final prefix+30,trial prefix+21,30), checked after tokenizer load",
            "interpretation": "chunk filling validates memory and shapes; no single-token numerical parity claim"}
    if args.inspect_only:
        print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
        return 0
    if args.model_dir is None or args.run_root is None:
        raise ValueError("GPU execution requires --model-dir and a new --run-root")
    run_root = args.run_root.expanduser().resolve()
    protected = [args.source_record.parent, args.gate_summary.parent, args.model_dir, *args.natural_run_root]
    if run_root.exists() or any(run_root.is_relative_to(path.expanduser().resolve()) for path in protected):
        raise ValueError("Run root must be new and outside model, source, gate, and natural-run evidence")
    run_root.mkdir(parents=True, exist_ok=False)
    atomic_new(run_root / "plan.json", plan)
    code_hashes = {name: sha256(ROOT / name) for name in SOURCE_FILES}
    atomic_new(run_root / "source-hashes.json", code_hashes)
    integrity_before = {str(ROOT / name): digest for name, digest in code_hashes.items()}
    integrity_before.update({source["path"]: source["sha256"], **gate["input_sha256"], **coverage["input_sha256"]})
    atomic_new(run_root / "integrity-before.json", integrity_before)
    atomic_new(run_root / "source-token-pattern.json", source)
    log = EventLog(run_root / "events.jsonl")
    progress, answer_samples, state, backend = [], [], None, None
    phase, filled = "gpu_preflight", 0
    start = time.perf_counter()

    def deadline():
        if time.perf_counter() - start > args.wall_time_seconds:
            raise TimeoutError("Capacity wall-time budget reached; no retry or fallback")

    def snapshot(label):
        deadline()
        item = {"phase": label, "filled_main_tokens": filled,
                "elapsed_seconds": time.perf_counter() - start, **memory_snapshot(backend, state)}
        progress.append(item)
        atomic_new(run_root / f"memory-{len(progress):03d}.json", item)
        log.emit("capacity_progress", **item)
        if item.get("kv_lengths_consistent") is False:
            raise RuntimeError("At least one KV layer differs from the committed context length")

    try:
        gpu = gpu_inventory()
        with gpu_lock(gpu["uuid"]) as lock_path:
            idle = assert_gpu_idle(gpu["uuid"])
            atomic_new(run_root / "gpu-preflight.json", {"gpu": gpu, "idle": idle, "lock": lock_path})
            phase = "model_load"
            log.emit("model_load_start", scope=SCOPE)
            if backend_factory is None:
                from torch_online_backend import TorchOnlineBackend
                backend_factory = TorchOnlineBackend
            backend = backend_factory(args.model_dir, attention_implementation="eager")
            atomic_new(run_root / "environment.json", backend.metadata)
            verify_environment(gate["environment"], backend.metadata)
            reserve = max(len(backend.markers.final_prefix) + FINAL_TOKEN_CAP,
                          len(backend.markers.trial_prefix) + MAX_PROBE_TOKENS, FINAL_TOKEN_CAP)
            context_required = len(source["prompt_ids"]) + MAIN_TOKEN_BUDGET + reserve
            if context_required > backend.context_limit:
                raise ValueError(f"Context reserve insufficient: require {context_required}, have {backend.context_limit}")
            if any(token in (*backend.markers.eos_ids, backend.markers.end_think) for token in source["pattern_ids"]):
                raise ValueError("Saved synthetic fill pattern unexpectedly includes EOS/end-think")
            full_pattern = (source["pattern_ids"] * (MAIN_TOKEN_BUDGET // len(source["pattern_ids"]) + 1))[:MAIN_TOKEN_BUDGET]
            atomic_new(run_root / "capacity-input.json", {
                "prompt_token_ids": source["prompt_ids"], "synthetic_main_token_ids": full_pattern,
                "synthetic_filling": True, "context_required": context_required,
                "context_limit": backend.context_limit, "reserved_tokens": reserve})
            phase = "prompt_prefill"
            deadline()
            backend.start_request(args.seed)
            state = backend.prefill(source["prompt_ids"])
            snapshot("prompt_prefill")
            phase = "synthetic_chunk_fill"
            while filled < MAIN_TOKEN_BUDGET:
                deadline()
                end = min(filled + args.chunk_size, MAIN_TOKEN_BUDGET)
                state = backend.extend(state, full_pattern[filled:end])
                filled = end
                if state.length != len(source["prompt_ids"]) + filled or state.cache.get_seq_length() != state.length:
                    raise RuntimeError("Synthetic main KV shape/length conservation failed")
                if filled % 1024 == 0 or filled == MAIN_TOKEN_BUDGET:
                    snapshot("synthetic_chunk_fill")
            phase = "maximum_prefix_probe"
            before = state_fingerprint(backend, state)
            atomic_new(run_root / "before-probe-state.json", before)
            deadline()
            observation = backend.probe(state)
            after = state_fingerprint(backend, state)
            atomic_new(run_root / "probe.json", {"observation": asdict(observation), "state_after": after,
                                                "scope": "real probe on synthetic maximum-length KV",
                                                "numerical_parity_assessed": False})
            difference = first_difference(before, after)
            if difference:
                atomic_new(run_root / "first-difference.json", difference)
                raise RuntimeError("Maximum-prefix probe changed main KV/logits/RNG")
            if not 1 <= len(observation.token_ids) <= MAX_PROBE_TOKENS:
                raise RuntimeError("Maximum-prefix probe violated the token cap")
            snapshot("maximum_prefix_probe")
            phase = "controlled_final_prefix"
            deadline()
            state = backend.extend(state, backend.markers.final_prefix)
            snapshot("controlled_final_prefix")
            phase, answer_end = "greedy_finalization", "answer_budget"
            for _ in range(FINAL_TOKEN_CAP):
                deadline()
                sample = backend.greedy(state)
                answer_samples.append(asdict(sample))
                if sample.token_id in backend.markers.eos_ids:
                    answer_end = "eos"
                    break
                state = backend.extend(state, (sample.token_id,))
            atomic_new(run_root / "finalization.json", {
                "injected_prefix_ids": list(backend.markers.final_prefix), "samples": answer_samples,
                "end": answer_end, "generated_tokens": len(answer_samples),
                "eos_is_not_accepted_to_KV": True, "graded": False})
            snapshot("greedy_finalization")
            phase = "source_input_identity_integrity"
            integrity_after = {}
            for path in integrity_before:
                try:
                    integrity_after[path] = sha256(Path(path))
                except OSError as read_error:
                    integrity_after[path] = {"read_error": type(read_error).__name__}
            atomic_new(run_root / "integrity-after.json", integrity_after)
            difference = first_difference(integrity_before, integrity_after)
            if difference:
                atomic_new(run_root / "first-difference.json", difference)
                raise RuntimeError("Capacity diagnostic source/input/prerequisite identity changed during execution")
            summary = {"status": "capacity_shape_memory_passed", "scope": SCOPE,
                       "main_token_budget": MAIN_TOKEN_BUDGET, "filled_main_tokens": filled,
                       "context_required": context_required, "context_limit": backend.context_limit,
                       "probe_state_isolation": "passed", "probe_tokens": len(observation.token_ids),
                       "probe_observation_valid": observation.confidence_valid,
                       "finalization_tokens": len(answer_samples), "finalization_end": answer_end,
                       "final_KV_length": state.length, "memory_progress": progress,
                       "elapsed_seconds_including_validation_IO": time.perf_counter() - start,
                       "prerequisite_gate": gate, "natural_branch_coverage": coverage,
                       "natural_32k_generation_validated": False, "numerical_parity_assessed": False,
                       "eligible_for_accuracy_or_speed_comparison": False,
                       "source_input_identity_integrity": "unchanged",
                       "overall_GPU_acceptance": "not_established_by_capacity_diagnostic"}
            atomic_new(run_root / "summary.json", summary)
            log.emit("capacity_completed", status=summary["status"], scope=SCOPE)
            return 0
    except BaseException as error:
        failure = {"status": "failed", "scope": SCOPE, "phase": phase,
                   "filled_main_tokens": filled, "last_committed_state_length": getattr(state, "length", None),
                   "answer_samples": answer_samples, "memory_progress": progress,
                   "error_type": type(error).__name__, "message": str(error),
                   "traceback": traceback.format_exc(), "retry_performed": False,
                   "elapsed_seconds": time.perf_counter() - start}
        if backend is not None:
            try:
                failure["memory_at_failure"] = memory_snapshot(backend, state)
            except BaseException as memory_error:
                failure["memory_capture_error"] = type(memory_error).__name__
        atomic_new(run_root / "failure.json", failure)
        log.emit("capacity_failed", phase=phase, error_type=type(error).__name__, message=str(error))
        return 1
    finally:
        log.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-record", type=Path, required=True, help="Frozen a01 q002 Vanilla JSON; SHA anchored")
    result.add_argument("--gate-summary", type=Path, required=True, help="New passed five-prefix same-KV continuation summary")
    result.add_argument("--natural-run-root", type=Path, action="append", default=[], help="Completed natural online smoke run; repeatable")
    result.add_argument("--independent-capacity-diagnostic", action="store_true",
                        help="Explicitly allow missing natural branches, recording them as NOT validated")
    result.add_argument("--model-dir", type=Path, help="Existing pinned model snapshot; required for GPU execution")
    result.add_argument("--run-root", type=Path, help="New evidence directory; required for GPU execution")
    result.add_argument("--chunk-size", type=int, choices=(128, 256), default=128)
    result.add_argument("--wall-time-seconds", type=int, default=900, help="1..3600; checked between CUDA operations, not a hard kernel preemption")
    result.add_argument("--seed", type=int, default=872)
    result.add_argument("--inspect-only", action="store_true", help="Validate saved prerequisites without loading model or creating outputs")
    return result


def main(argv=None):
    try:
        return run_capacity(parser().parse_args(argv))
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(f"Capacity diagnostic refused: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
