#!/usr/bin/env python3
"""One gated q002 Vanilla request with a fixed 32768-token main budget.

This is development GPU acceptance, not a benchmark. All prerequisite evidence
must remain available on this host under its recorded paths. Inspect-only reads
and verifies those files without loading Torch, contacting a GPU, or writing.
No automatic retry, shorter budget, precision change, or attention fallback.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from online_contract import MODEL_ID, MODEL_REVISION, RUNNER_PROTOCOL
from online_engine import FINAL_TOKEN_CAP, RequestError, run_request
from run_online_diagnostic import (
    CODE_FILES, EventLog, ProgressBackend, answer_record, assert_gpu_idle,
    atomic_new, configuration_record, configurations, derive_seed,
    gpu_inventory, gpu_lock, sha256, verify_data,
)
from validate_online_capacity import (
    BRANCHES, CRITICAL_SOURCES, MAIN_TOKEN_BUDGET, SAMPLE_ID, SOURCE_SHA256,
    SOURCE_FILES as CAPACITY_SOURCE_FILES, SCOPE as CAPACITY_SCOPE,
    check_source_hashes, load_source, memory_snapshot, read_json,
    validate_gate, validate_natural_runs,
)
from diagnose_probe_parity import ENVIRONMENT_FIELDS, verify_environment
from validate_online_gpu import first_difference, fresh_run

SCOPE = "fixed_q002_32k_vanilla_gpu_acceptance_not_benchmark"
MASTER_SEED, ROLLOUT_ID = 42, 0
SOURCE_FILES = tuple(dict.fromkeys(("scripts/run_online_long_context.py", *CODE_FILES,
                                   *CAPACITY_SOURCE_FILES)))
CRITICAL = (*CRITICAL_SOURCES, "src/online_engine.py")


def current_hashes(expected):
    values = {}
    for name in expected:
        try:
            values[name] = sha256(Path(name))
        except OSError as error:
            values[name] = {"read_error": type(error).__name__}
    return values


def require_integrity(expected, label):
    difference = first_difference(expected, current_hashes(expected))
    if difference:
        raise ValueError(f"{label} changed or is unavailable: {difference}")


def validate_capacity(path, gate):
    """Bind actual saved capacity shapes, inputs, source identities and model."""
    root = Path(path).expanduser().resolve()
    names = ("summary.json", "plan.json", "source-hashes.json", "environment.json",
             "capacity-input.json", "source-token-pattern.json", "integrity-before.json",
             "integrity-after.json", "before-probe-state.json", "probe.json", "finalization.json")
    evidence = {name: read_json(root / name) for name in names}
    inputs = {str(root / name): sha256(root / name) for name in names}
    summary, plan, environment = (evidence[name] for name in ("summary.json", "plan.json", "environment.json"))
    if (summary.get("status") != "capacity_shape_memory_passed" or summary.get("scope") != CAPACITY_SCOPE
            or summary.get("source_input_identity_integrity") != "unchanged"
            or summary.get("filled_main_tokens") != MAIN_TOKEN_BUDGET
            or summary.get("main_token_budget") != MAIN_TOKEN_BUDGET
            or summary.get("probe_state_isolation") != "passed"
            or (root / "failure.json").exists() or (root / "first-difference.json").exists()):
        raise ValueError("A completed, unchanged-source 32768-token capacity diagnostic is required")
    before, after = evidence["integrity-before.json"], evidence["integrity-after.json"]
    if not before or before != after or any(not Path(name).is_absolute() for name in before):
        raise ValueError("Capacity integrity records must be identical absolute-path identities")
    require_integrity(before, "Recorded capacity source/input")
    inputs.update(before)
    hashes = evidence["source-hashes.json"]
    check_source_hashes(hashes, CRITICAL)
    if set(hashes) != set(CAPACITY_SOURCE_FILES) or any(
            not any(name.endswith("/" + relative) and digest == value for name, digest in before.items())
            for relative, value in hashes.items()):
        raise ValueError("Capacity source map is not bound to its integrity records")
    if (summary.get("prerequisite_gate") != plan.get("prerequisite_gate")
            or summary["prerequisite_gate"].get("sha256") != gate["sha256"]):
        raise ValueError("Capacity was not validated against the selected same-KV gate")
    verify_environment(gate["environment"], environment)
    if (environment.get("validation_model") is not False
            or environment.get("model_id") != MODEL_ID or environment.get("model_revision") != MODEL_REVISION
            or environment.get("model_parameter_dtype") != "torch.bfloat16"
            or environment.get("attention_implementation") != "eager"):
        raise ValueError("Capacity evidence must use the real pinned BF16/eager model")
    pattern = evidence["source-token-pattern.json"]
    source = load_source(pattern["path"], expected_sha256=SOURCE_SHA256)
    if source != pattern or plan.get("source", {}).get("sha256") != SOURCE_SHA256:
        raise ValueError("Capacity pattern differs from the frozen q002 source")
    if before.get(source["path"]) != SOURCE_SHA256:
        raise ValueError("Capacity source is absent from its integrity chain")
    inp, markers = evidence["capacity-input.json"], environment["markers"]
    prompt, final_prefix, trial_prefix = source["prompt_ids"], markers["final_prefix"], markers["trial_prefix"]
    reserve = max(len(final_prefix) + FINAL_TOKEN_CAP, len(trial_prefix) + 21, FINAL_TOKEN_CAP)
    required = len(prompt) + MAIN_TOKEN_BUDGET + reserve
    if (inp.get("synthetic_filling") is not True or inp.get("prompt_token_ids") != prompt
            or inp.get("synthetic_main_token_ids") != (source["pattern_ids"] * 32)
            or inp.get("reserved_tokens") != reserve or inp.get("context_required") != required
            or summary.get("context_required") != required
            or inp.get("context_limit") != summary.get("context_limit")
            or inp["context_limit"] != environment.get("context_limit")
            or inp["context_limit"] != environment["model_config"]["max_position_embeddings"]
            or required > inp["context_limit"]):
        raise ValueError("Capacity synthetic input or actual context reserve is inconsistent")
    probe, state, final = (evidence[name] for name in ("probe.json", "before-probe-state.json", "finalization.json"))
    observation = probe["observation"]
    if (set(state) != {"length", "kv", "logits", "private_rng", "global_cpu_rng", "global_cuda_rng"}
            or len(state["kv"]) != environment["model_config"]["num_hidden_layers"]
            or state != probe.get("state_after") or state.get("length") != len(prompt) + MAIN_TOKEN_BUDGET
            or not 1 <= len(observation["token_ids"]) <= 21
            or summary.get("probe_tokens") != len(observation["token_ids"])):
        raise ValueError("Capacity probe state-isolation evidence is inconsistent")
    samples = final["samples"]
    if (final.get("injected_prefix_ids") != final_prefix or len(samples) != FINAL_TOKEN_CAP
            or final.get("generated_tokens") != FINAL_TOKEN_CAP or final.get("end") != "answer_budget"
            or any(item["token_id"] in markers["eos_ids"] for item in samples)
            or summary.get("final_KV_length") != required or summary.get("finalization_tokens") != FINAL_TOKEN_CAP):
        raise ValueError("Capacity must actually reach the full final-prefix and 30-token answer reserve")
    progress = summary.get("memory_progress", [])
    if len(progress) != 36 or [row["filled_main_tokens"] for row in progress[:33]] != list(range(0, 32769, 1024)):
        raise ValueError("Capacity requires all 1024-token progress snapshots")
    config = environment["model_config"]
    layers, heads, width = config["num_hidden_layers"], config["num_key_value_heads"], config["head_dim"]
    for index, row in enumerate(progress, 1):
        memory_path = root / f"memory-{index:03d}.json"
        if read_json(memory_path) != row:
            raise ValueError("Capacity memory snapshot differs from its summary")
        inputs[str(memory_path)] = sha256(memory_path)
        length = len(prompt) + row["filled_main_tokens"]
        if index >= 35:
            length += len(final_prefix)
        if index == 36:
            length += FINAL_TOKEN_CAP
        shape = [1, heads, length, width]
        if (row.get("kv_lengths_consistent") is not True or row["kv_length"] != length
                or row["committed_state_length"] != length
                or row["kv_layer_shapes"] != [[shape, shape] for _ in range(layers)]
                or row["kv_tensor_bytes"] != layers * 2 * heads * length * width * 2
                or not 0 <= row["allocated_bytes"] <= row["reserved_bytes"] <= row["device_total_bytes"]
                or not row["allocated_bytes"] <= row["peak_allocated_bytes"] <= row["peak_reserved_bytes"] <= row["device_total_bytes"]):
            raise ValueError("Capacity KV shapes, bytes or memory bounds are inconsistent")
    return {"path": str(root), "summary_sha256": sha256(root / "summary.json"), "input_sha256": inputs,
            "prompt_token_ids": prompt, "context_required": required, "context_limit": inp["context_limit"],
            "environment": {key: environment[key] for key in ENVIRONMENT_FIELDS},
            "gpu_total_memory_bytes": environment["gpu_total_memory_bytes"],
            "probe_actual_tokens": len(observation["token_ids"]), "probe_cap": 21,
            "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in progress),
            "peak_reserved_bytes": max(row["peak_reserved_bytes"] for row in progress),
            "minimum_sampled_device_free_bytes": min(row["device_free_bytes"] for row in progress),
            "interpretation": "Synthetic capacity only; actual 21-token probe tail and natural 32K generation were not established."}


def prepare(args):
    if args.max_new_tokens != MAIN_TOKEN_BUDGET:
        raise ValueError("This entry permits exactly 32768 main tokens")
    row, data = verify_data(args.data_dir, args.data_manifest_sha256, SAMPLE_ID)
    gate = validate_gate(args.gate_summary)
    if gate["isolation_continuation_tokens"] != 64:
        raise ValueError("The selected same-KV gate must cover all 64 continuation tokens")
    natural = validate_natural_runs(args.natural_run_root, independent=False)
    if not natural["gate_passed"] or natural["missing"] or set(natural["covered"]) != set(BRANCHES):
        raise ValueError("All three actual branch gates are required; no independent override")
    capacity = validate_capacity(args.capacity_root, gate)
    inputs = {**gate["input_sha256"], **natural["input_sha256"], **capacity["input_sha256"]}
    inputs[str(args.data_dir.resolve() / "manifest.json")] = data["manifest_sha256"]
    inputs.update({str(args.data_dir.resolve() / name): digest for name, digest in data["files_sha256"].items()})
    for evidence in natural["evidence"]:
        verify_environment(gate["environment"], read_json(evidence["path"])["backend_metadata"])
    # Include the natural validation's derived plans/pair checks, as well as its
    # request records. No assumption that every run uses the same pair schema.
    for directory in args.natural_run_root:
        directory = Path(directory)
        summary = read_json(directory / "summary.json")
        for flag, name in (("boundary_pair_passed", "boundary-pair-validation.json"),
                           ("answer_body_pair_passed", "answer-body-pair-validation.json")):
            if flag in summary and (summary[flag] is not True
                                    or read_json(directory / name).get("passed") is not True):
                raise ValueError("Natural boundary pair evidence did not pass")
        if "coverage_complete" in summary and summary["coverage_complete"] is not True:
            raise ValueError("Natural boundary coverage is incomplete")
        after = directory / "source-hashes-after.json"
        if after.exists() and read_json(after) != read_json(directory / "manifest.json")["code_sha256"]:
            raise ValueError("Natural boundary source hashes changed")
        inputs.update({str(path.resolve()): sha256(path) for path in directory.glob("*.json")})
    code = {name: sha256(ROOT / name) for name in SOURCE_FILES}
    inputs.update({str(ROOT / name): digest for name, digest in code.items()})
    require_integrity(inputs, "Preflight source/input")
    seed = derive_seed(MASTER_SEED, SAMPLE_ID, ROLLOUT_ID, "reason")
    config = configurations(["vanilla"], 1)[0]
    config_record = configuration_record(config, seed_reason=seed, seed_schedule=None,
                                        max_new_tokens=MAIN_TOKEN_BUDGET, attention_implementation="eager")
    manifest = {"schema_version": 1, "scope": SCOPE, "runner_protocol": RUNNER_PROTOCOL,
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "sample_id": SAMPLE_ID,
        "problem_sha256": row["problem_sha256"], "question_count": 1, "planned_count": 1,
        "method": "vanilla", "rollout_id": ROLLOUT_ID, "master_seed": MASTER_SEED,
        "seed_reason": seed, "max_new_tokens": MAIN_TOKEN_BUDGET, "max_final_tokens": FINAL_TOKEN_CAP,
        "attention_implementation": "eager", "dtype": "BF16", "configurations": [config_record],
        "data_identity": data, "code_sha256": code, "source_input_sha256": inputs,
        "prerequisite_gate": gate, "natural_branch_coverage": natural, "capacity_evidence": capacity,
        "eligible_for_primary_speed_comparison": False, "forced_tokens": False,
        "warmup": {"main_tokens": 16, "max_final_tokens": FINAL_TOKEN_CAP, "method": "vanilla",
                   "included_in_measured_requests": False, "includes_probe_warmup": False},
        "logging": {"main_progress_every": 64, "synchronous_logging_overhead_in_request_timing": True},
        "grading": "separate strict grade_math_answers.py invocation; not inside request timer",
        "failure_policy": "one attempt; stop and save; no budget, dtype or attention fallback"}
    return row, manifest


def invoke(backend, log, *, question, sample_id, label, seed, budget):
    error, started = None, time.perf_counter()
    try:
        record = run_request(ProgressBackend(backend, log, label, every=64), question=question,
            sample_id=sample_id, rollout_id=ROLLOUT_ID, method="vanilla", seed_reason=seed,
            max_new_tokens=budget)
    except BaseException as exc:
        error = exc
        record = exc.partial if isinstance(exc, RequestError) else {
            "status": "failed", "sample_id": sample_id, "rollout_id": ROLLOUT_ID,
            "method": "vanilla", "seed_reason": seed, "max_new_tokens": budget,
            "partial_evidence_available": False, "error_type": type(exc).__name__, "error": str(exc),
            "external_elapsed_ms_before_error_report": (time.perf_counter() - started) * 1000,
            "timing_scope": "external failed invocation elapsed; engine phase timings unavailable"}
    return record, error


def run_long_context(args, *, backend_factory=None):
    row, manifest = prepare(args)
    if args.inspect_only:
        print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
        return 0
    if args.model_dir is None or args.run_root is None:
        raise ValueError("Execution requires --model-dir and a new --run-root")
    model_dir, run_root = args.model_dir.expanduser().resolve(), args.run_root.expanduser().resolve()
    protected = (model_dir, args.data_dir, args.capacity_root, args.gate_summary.parent, *args.natural_run_root)
    if (run_root.exists() or args.run_root.is_symlink()
            or any(run_root.is_relative_to(Path(path).resolve()) for path in protected)):
        raise ValueError("Run root must be new and outside model/data/prerequisite evidence")
    if not model_dir.is_dir() or model_dir.name != MODEL_REVISION:
        raise ValueError("Expected an existing pinned local model snapshot")
    with fresh_run(run_root):
        atomic_new(run_root / "manifest.json", manifest)
        atomic_new(run_root / "integrity-before.json", manifest["source_input_sha256"])
        log = EventLog(run_root / "events.jsonl")
        answers, completed, failure = [], [], None
        backend, phase = None, "gpu_preflight"
        try:
            require_integrity(manifest["source_input_sha256"], "Before GPU startup source/input")
            gpu = gpu_inventory()
            with gpu_lock(gpu["uuid"]) as lock_path:
                idle = assert_gpu_idle(gpu["uuid"])
                atomic_new(run_root / "gpu-preflight.json", {"gpu": gpu, "idle": idle, "lock": lock_path})
                phase, started = "model_load", time.perf_counter()
                log.emit("model_load_start", model_revision=MODEL_REVISION)
                if backend_factory is None:
                    from torch_online_backend import TorchOnlineBackend
                    backend_factory = TorchOnlineBackend
                backend = backend_factory(model_dir, attention_implementation="eager")
                atomic_new(run_root / "environment.json", {"backend_metadata": backend.metadata, "gpu": gpu,
                    "model_load_seconds_external": time.perf_counter() - started})
                for expected in (manifest["prerequisite_gate"]["environment"], manifest["capacity_evidence"]["environment"]):
                    verify_environment(expected, backend.metadata)
                if (backend.metadata.get("validation_model") is not False or backend.metadata.get("device") != "cuda:0"
                        or backend.metadata.get("gpu_total_memory_bytes") != manifest["capacity_evidence"]["gpu_total_memory_bytes"]):
                    raise ValueError("Actual backend is not the validated production CUDA device")
                prompt = backend.prompt_ids(row["problem"])
                if prompt != manifest["capacity_evidence"]["prompt_token_ids"]:
                    raise ValueError("Actual frozen q002 prompt differs from capacity/source evidence")
                reserve = max(len(backend.markers.final_prefix) + FINAL_TOKEN_CAP,
                              len(backend.markers.trial_prefix) + 21, FINAL_TOKEN_CAP)
                if (len(prompt) + MAIN_TOKEN_BUDGET + reserve != manifest["capacity_evidence"]["context_required"]
                        or backend.context_limit != manifest["capacity_evidence"]["context_limit"]):
                    raise ValueError("Actual context capacity or reserve differs from validated capacity")
                log.emit("model_load_completed", elapsed_seconds=time.perf_counter() - started)
                phase = "warmup"
                warmup, error = invoke(backend, log, question="Compute 1 + 1.", sample_id="synthetic-warmup-only",
                    label="warmup", seed=derive_seed(MASTER_SEED, SAMPLE_ID, ROLLOUT_ID, "warmup"), budget=16)
                atomic_new(run_root / "warmup.json", {**warmup, "scope": "excluded_warmup"})
                if error:
                    raise error
                log.emit("warmup_completed", elapsed_ms=warmup["time_total_ms"], excluded_from_results=True)
                atomic_new(run_root / "memory-before-main.json", memory_snapshot(backend))
                require_integrity(manifest["source_input_sha256"], "Before main request source/input")
                phase = "main_request"
                log.emit("request_start", configuration="vanilla", max_new_tokens=MAIN_TOKEN_BUDGET,
                         seed_reason=manifest["seed_reason"], config_hash=manifest["configurations"][0]["config_hash"])
                result, error = invoke(backend, log, question=row["problem"], sample_id=SAMPLE_ID,
                    label="vanilla", seed=manifest["seed_reason"], budget=MAIN_TOKEN_BUDGET)
                result.update(configuration="vanilla", scope=SCOPE, eligible_for_primary_speed_comparison=False,
                              backend_metadata=backend.metadata, request_configuration=manifest["configurations"][0],
                              config_hash=manifest["configurations"][0]["config_hash"])
                source_path = run_root / "vanilla.json"
                atomic_new(source_path, result)
                try:
                    answer = answer_record(result, source_path, row["answer"], backend, "vanilla")
                except BaseException as export_error:
                    answer = answer_record({**result, "status": "failed", "answer_token_ids": []},
                                           source_path, row["answer"], backend, "vanilla")
                    answer.update(source_execution_status=result["status"], export_error=str(export_error))
                    error = error or export_error
                atomic_new(run_root / "vanilla.final-answer.json", answer)
                answers.append(answer)
                completed.append({"configuration": "vanilla", "status": answer["execution_status"],
                                  "source_execution_status": result["status"], "source_sha256": sha256(source_path)})
                log.emit("request_saved", configuration="vanilla", status=answer["execution_status"],
                         main_tokens=result.get("main_generated_tokens"), stop_reason=result.get("stop_reason"),
                         elapsed_ms=result.get("time_total_ms"))
                if error:
                    raise error
        except BaseException as error:
            failure = {"status": "failed", "phase": phase, "error_type": type(error).__name__,
                       "message": str(error), "traceback": traceback.format_exc(), "retry_performed": False}
            atomic_new(run_root / "failure.json", failure)
            log.emit("run_failed", phase=phase, error_type=type(error).__name__, message=str(error))
        finally:
            if backend is not None:
                try:
                    atomic_new(run_root / "memory-after-run.json", memory_snapshot(backend))
                except BaseException as error:
                    atomic_new(run_root / "memory-capture-error.json", {"error_type": type(error).__name__, "message": str(error)})
            after = current_hashes(manifest["source_input_sha256"])
            atomic_new(run_root / "integrity-after.json", after)
            difference = first_difference(manifest["source_input_sha256"], after)
            if difference:
                integrity_failure = {"error_type": "SourceInputChanged", "message": str(difference), "phase": "final_integrity"}
                atomic_new(run_root / "integrity-failure.json", integrity_failure)
                failure = failure or integrity_failure
            payload = "".join(json.dumps(answer, ensure_ascii=False, allow_nan=False) + "\n"
                              for answer in answers).encode("utf-8")
            atomic_new(run_root / "final_answers.jsonl", payload, raw=True)
            summary = {"schema_version": 1, "scope": SCOPE, "status": "failed" if failure else "completed",
                "planned_count": 1, "executed_count": len(completed), "unexecuted_count": 1 - len(completed),
                "requests": completed, "failure": failure, "warmup_excluded": True, "grading_status": "not_run",
                "source_input_identity_integrity": "changed" if difference else "unchanged",
                "eligible_for_primary_speed_comparison": False,
                "final_answers_sha256": sha256(run_root / "final_answers.jsonl")}
            atomic_new(run_root / "summary.json", summary)
            log.emit("run_finished", status=summary["status"], executed_count=len(completed))
            log.close()
        return 1 if failure else 0


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-dir", type=Path, default=ROOT / "data/benchmarks")
    result.add_argument("--data-manifest-sha256", required=True)
    result.add_argument("--gate-summary", type=Path, required=True)
    result.add_argument("--capacity-root", type=Path, required=True)
    result.add_argument("--natural-run-root", type=Path, action="append", required=True)
    result.add_argument("--max-new-tokens", type=int, choices=(32768,), required=True)
    result.add_argument("--model-dir", type=Path)
    result.add_argument("--run-root", type=Path)
    result.add_argument("--inspect-only", action="store_true")
    return result


def main(argv=None):
    try:
        return run_long_context(parser().parse_args(argv))
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(f"Long-context request refused: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
