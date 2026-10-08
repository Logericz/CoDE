#!/usr/bin/env python3
"""Bounded, fail-closed CUDA acceptance using existing q001/q002 prefixes.

Inputs are read-only. This script never searches for a saved question's answer,
downloads weights, modifies the upstream checkout, or resumes a prior run.
Numerical acceptance is exact (atol=rtol=0), including raw BF16 probabilities.
The full-prefix reference is extracted from the pinned original Git object, or
from a byte-identical exported source on deployments without a Git repository.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from online_contract import MODEL_REVISION
from online_protocol import MAX_PROBE_TOKENS, UPSTREAM_COMMIT
from run_online_diagnostic import assert_gpu_idle, atomic_new, gpu_inventory, gpu_lock

REFERENCE_FILE = "method_codestop.py"
REFERENCE_SHA256 = "d020fde481bf1ad2d4ee53ea43a3ae31560e05be1d6cf6f172489b54179cb1ff"
DEFAULT_PREFIXES = ("q001:0", "q001:1", "q002:0")
TOLERANCE = {"atol": 0.0, "rtol": 0.0, "equal_nan": True,
             "policy": "exact; preserve first difference and stop; no dtype/attention fallback"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path, value):
    # Publish complete files atomically without replacing an existing filename.
    atomic_new(Path(path), json_safe(value))


def event(run_root, kind, **fields):
    record = {"time_utc": datetime.now(timezone.utc).isoformat(), "event": kind, **fields}
    line = json.dumps(json_safe(record), ensure_ascii=False, allow_nan=False)
    with (run_root / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(line, flush=True)


@contextmanager
def fresh_run(run_root):
    """Atomic creation plus a process lock; stale/finished directories are refused."""
    import fcntl
    run_root = Path(run_root)
    run_root.parent.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(exist_ok=False)
    with (run_root / ".run.lock").open("x", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.write(str(os.getpid()) + "\n")
        lock.flush()
        try:
            yield run_root
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def ids(value, label):
    if not isinstance(value, list) or not value or any(type(x) is not int or x < 0 for x in value):
        raise ValueError(f"{label} requires a nonempty integer token list")
    return value


def one_batch(value, label):
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError(f"{label} requires exactly one saved batch item")
    return ids(value[0], label)


def parse_prefixes(selectors):
    if not 1 <= len(selectors) <= 3 or len(set(selectors)) != len(selectors):
        raise ValueError("Select 1-3 unique saved prefixes")
    result = []
    for selector in selectors:
        parts = selector.split(":")
        if len(parts) != 2 or parts[0] not in ("q001", "q002") or not parts[1].isdigit():
            raise ValueError("Prefixes must be q001:INDEX or q002:INDEX (zero-based)")
        result.append((parts[0], int(parts[1])))
    return result


def load_saved_prefixes(evidence_root, selectors):
    """Recover the exact saved pre-probe prefix from the recorded final input.

    Earlier checks are prefixes of the last checked input. Tokenizer validation
    below independently checks this against upstream's sample_response encoding.
    """
    evidence_root = Path(evidence_root).resolve()
    cases = []
    for question, index in parse_prefixes(selectors):
        documents, hashes = {}, {}
        for name in ("base.json", "codestop.json"):
            path = evidence_root / "items" / question / name
            raw = path.read_bytes()
            documents[name], hashes[str(path)] = json.loads(raw), sha(raw)
            if documents[name].get("status") != "completed":
                raise ValueError(f"Saved {question}/{name} did not complete")
            if documents[name].get("metrics", {}).get("model_revision") != MODEL_REVISION:
                raise ValueError(f"Saved {question}/{name} has a different model revision")
        base, stage = documents["base.json"], documents["codestop.json"]
        checks = stage["method_output"]["prob_checks"]
        if not checks or index >= len(checks):
            raise ValueError(f"No saved probe at {question}:{index}")
        positions = [check["stop_token_idx"] for check in checks]
        if any(type(x) is not int or x < 0 for x in positions) or positions != sorted(set(positions)):
            raise ValueError("Saved checkpoint positions must be strictly increasing integers")
        prompt = one_batch(base["generation_calls"][0]["input_token_ids"], "saved prompt")
        calls = stage["generation_calls"]
        if len(calls) != 1 or calls[0].get("status") != "ok":
            raise ValueError("Expected exactly one successful saved final-answer call")
        final_input = one_batch(calls[0]["input_token_ids"], "saved final input")
        last_length = len(prompt) + positions[-1]
        if final_input[:len(prompt)] != prompt or len(final_input) <= last_length:
            raise ValueError("Saved final input does not preserve its prompt and final prefix")
        prefix = final_input[:len(prompt) + positions[index]]
        cases.append({"name": f"{question}-probe-{index}", "question": question,
                      "probe_index": index, "stop_token_idx": positions[index],
                      "prompt_ids": prompt, "prefix_ids": prefix,
                      "saved_last_prefix_ids": final_input[:last_length],
                      "saved_final_prefix_ids": final_input[last_length:],
                      "sample_response": stage["method_output"]["sample_response"],
                      "saved_observation": checks[index], "input_sha256": hashes,
                      "saved_attention": stage.get("metadata", {}).get("attention_implementation"),
                      "prefix_ids_sha256": sha(json.dumps(prefix, separators=(",", ":")).encode())})
    return cases


def validate_tokenizer_prefix(case, backend):
    response = backend.tokenizer(case["sample_response"])["input_ids"]
    saved_prefix = case["saved_last_prefix_ids"]
    expected = case["prompt_ids"] + response[:len(saved_prefix) - len(case["prompt_ids"])]
    if saved_prefix != expected:
        raise ValueError("Original tokenizer does not reproduce the saved upstream reasoning prefix")
    if case["saved_final_prefix_ids"] != list(backend.markers.final_prefix):
        raise ValueError("Saved final-answer prefix differs from the frozen online prefix")
    backend._tokens(case["prefix_ids"])
    response_position = case["stop_token_idx"]
    if response_position >= len(response) or response[response_position] not in (
            backend.markers.wait, backend.markers.eos):
        raise ValueError("Selected saved position is not a Wait/EOS checkpoint")


def reference_source(upstream_source_dir=None):
    if upstream_source_dir is None:
        command = ["git", "-C", str(ROOT / "upstream" / "CoDE-Stop"), "show",
                   f"{UPSTREAM_COMMIT}:{REFERENCE_FILE}"]
        raw = subprocess.run(command, check=True, capture_output=True).stdout
        source_kind = "pinned_git_object"
    else:
        raw = (Path(upstream_source_dir) / REFERENCE_FILE).read_bytes()
        source_kind = "sha256_verified_git_object_export"
    if sha(raw) != REFERENCE_SHA256:
        raise ValueError("Reference source differs from the frozen upstream Git object")
    functions = [node for node in ast.parse(raw).body
                 if isinstance(node, ast.FunctionDef) and node.name == "calcu_max_probs_w_kv"]
    if len(functions) != 1:
        raise ValueError("Expected the original calcu_max_probs_w_kv function exactly once")
    return raw, functions[0], {"source_kind": source_kind, "upstream_commit": UPSTREAM_COMMIT,
                               "file": REFERENCE_FILE, "sha256": sha(raw)}


def load_reference(function, torch):
    namespace = {"torch": torch, "F": torch.nn.functional}
    exec(compile(ast.Module(body=[function], type_ignores=[]),
                 f"{UPSTREAM_COMMIT}:{REFERENCE_FILE}", "exec"), namespace)
    return namespace["calcu_max_probs_w_kv"]


def first_difference(expected, actual, path=""):
    """Exact recursive comparison; matching nonfinite raw diagnostics stay raw."""
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            return {"field": path + ".length", "expected": len(expected), "actual": len(actual)}
        for index, (left, right) in enumerate(zip(expected, actual)):
            difference = first_difference(left, right, f"{path}[{index}]")
            if difference:
                return difference
        return None
    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            return {"field": path + ".keys", "expected": sorted(expected), "actual": sorted(actual)}
        for key in expected:
            difference = first_difference(expected[key], actual[key], f"{path}.{key}".strip("."))
            if difference:
                return difference
        return None
    if isinstance(expected, float) and isinstance(actual, float) and math.isnan(expected) and math.isnan(actual):
        return None
    if type(expected) is not type(actual) or expected != actual:
        return {"field": path, "expected": expected, "actual": actual}
    return None


def tensor_fingerprint(tensor):
    # view(uint8) preserves BF16 bits without a float32 conversion.
    import torch
    raw = tensor.detach().contiguous().cpu()
    return {"dtype": str(raw.dtype), "shape": list(raw.shape),
            "sha256": sha(raw.view(torch.uint8).numpy().tobytes())}


def state_fingerprint(backend, state):
    result = {"length": state.length, "logits": tensor_fingerprint(state.logits),
              "kv": [[tensor_fingerprint(key), tensor_fingerprint(value)] for key, value in state.cache],
              "private_rng": tensor_fingerprint(backend._generator.get_state()),
              "global_cpu_rng": tensor_fingerprint(backend.torch.random.get_rng_state())}
    if backend.device.type == "cuda":
        result["global_cuda_rng"] = tensor_fingerprint(backend.torch.cuda.get_rng_state(backend.device))
    return result


def bf16_bits(torch, values):
    tensor = torch.tensor(values, dtype=torch.bfloat16)
    return [int(value) & 0xffff for value in tensor.view(torch.int16).tolist()]


class AcceptanceFailure(RuntimeError):
    pass


def fail_on_difference(run_root, expected, actual, *, phase, case):
    difference = first_difference(expected, actual)
    if difference:
        write_json(run_root / "first-difference.json", {"phase": phase, "case": case,
                                                       "tolerance": TOLERANCE, **difference})
        raise AcceptanceFailure(f"{phase} exact comparison failed for {case}: {difference['field']}")


def rebuild_online_prefix(backend, case, run_root):
    """Teacher-force saved IDs through production's prompt + one-token KV path.

    No sampling or answer search occurs here. A full-prefix prefill alone would
    miss numerical differences caused by the actual incremental main KV path.
    """
    prompt, prefix = case["prompt_ids"], case["prefix_ids"]
    if prefix[:len(prompt)] != prompt:
        raise ValueError("Saved online prefix does not start with its original prompt")
    state = backend.prefill(prompt)
    for index, token in enumerate(prefix[len(prompt):], start=1):
        state = backend.extend(state, (token,))
        if index % 256 == 0:
            event(run_root, "saved_prefix_replay_progress", case=case["name"], accepted_saved_tokens=index)
    return state


def validate_probe(backend, reference, case, run_root, seed):
    name = case["name"]
    backend.start_request(seed)
    backend.synchronize()
    start = time.perf_counter()
    state = rebuild_online_prefix(backend, case, run_root)
    backend.synchronize()
    prefix_build_seconds = time.perf_counter() - start
    before = state_fingerprint(backend, state)
    event(run_root, "probe_start", case=name, prefix_tokens=state.length)
    backend.synchronize()
    start = time.perf_counter()
    observation = backend.probe(state)
    backend.synchronize()
    actual_seconds = time.perf_counter() - start
    after = state_fingerprint(backend, state)
    actual = {"token_ids": list(observation.token_ids), "token_probs": list(observation.token_probs),
              "confidence": observation.confidence_raw, "ended": observation.ended_with_think}
    write_json(run_root / f"{name}-online-probe.json", {
        **actual, "confidence_valid": observation.confidence_valid,
        "invalid_reasons": observation.invalid_reasons, "source": observation.confidence_source,
        "prefix_build_seconds": prefix_build_seconds, "probe_seconds": actual_seconds,
        "prefix_build_method": "original_prompt_prefill_then_saved_token_by_token_extend",
        "state_before": before, "state_after": after,
        "probability_bf16_bits": bf16_bits(backend.torch, actual["token_probs"]),
        "confidence_bf16_bits": bf16_bits(backend.torch, [actual["confidence"]])})
    fail_on_difference(run_root, before, after, phase="probe_state_isolation", case=name)
    if len(actual["token_ids"]) > MAX_PROBE_TOKENS:
        raise AcceptanceFailure("Probe exceeded its fixed 21-token bound")
    full = case["prefix_ids"] + list(backend.markers.trial_prefix)
    event(run_root, "reference_start", case=name, full_prefix_tokens=len(full))
    backend.synchronize()
    start = time.perf_counter()
    original = reference(backend.model, backend.torch.tensor([full], device=backend.device),
                         None, backend.tokenizer, method=0)
    backend.synchronize()
    reference_seconds = time.perf_counter() - start
    expected = {"token_ids": original["token_ids"], "token_probs": original["token_probs"],
                "confidence": original["total_prob_max"], "ended": bool(original["ended_with_think"])}
    write_json(run_root / f"{name}-reference-probe.json", {
        **expected, "reference_seconds": reference_seconds,
        "probability_bf16_bits": bf16_bits(backend.torch, expected["token_probs"]),
        "confidence_bf16_bits": bf16_bits(backend.torch, [expected["confidence"]]),
        "saved_historical_observation": case["saved_observation"],
        "historical_comparison_scope": "context only; historical attention/environment may differ"})
    fail_on_difference(run_root, expected, actual, phase="full_prefix_probe_numerics", case=name)
    event(run_root, "probe_passed", case=name, generated_tokens=len(actual["token_ids"]),
          confidence=actual["confidence"], ended=actual["ended"],
          probe_seconds=actual_seconds, reference_seconds=reference_seconds)
    return {"case": name, "probe_tokens": len(actual["token_ids"]),
            "probe_seconds": actual_seconds, "reference_seconds": reference_seconds,
            "prefix_build_seconds": prefix_build_seconds, "exact": True}


def validate_isolation(backend, case, run_root, seed, continuation_tokens):
    baseline, results = None, []
    for probes in (0, 1, 2):
        backend.start_request(seed)
        backend.synchronize()
        start = time.perf_counter()
        state = rebuild_online_prefix(backend, case, run_root)
        initial = state_fingerprint(backend, state)
        write_json(run_root / f"isolation-{probes}-initial-state.json", initial)
        event(run_root, "isolation_start", case=case["name"], inserted_probes=probes,
              continuation_cap=continuation_tokens)
        observations = []
        for index in range(probes):
            observation = backend.probe(state)
            observations.append({"token_ids": list(observation.token_ids),
                                 "token_probs": list(observation.token_probs),
                                 "confidence": observation.confidence_raw,
                                 "ended": observation.ended_with_think})
            current = state_fingerprint(backend, state)
            write_json(run_root / f"isolation-{probes}-after-probe-{index}.json",
                       {"state": current, "observation": observations[-1]})
            fail_on_difference(run_root, initial, current, phase="inserted_probe_state", case=case["name"])
        trace = []
        for index in range(continuation_tokens):
            sample = backend.sample(state)
            trace.append({"token_id": sample.token_id, "sample_probability": sample.sample_probability,
                          "raw_probability": sample.raw_probability})
            if (index + 1) % 16 == 0 or sample.token_id in backend.markers.eos_ids:
                event(run_root, "continuation_progress", inserted_probes=probes,
                      generated_tokens=index + 1, token_id=sample.token_id)
            if sample.token_id in backend.markers.eos_ids:
                break
            state = backend.extend(state, (sample.token_id,))
        backend.synchronize()
        elapsed = time.perf_counter() - start
        comparison = {"initial_state": initial, "continuation": trace,
                      "final_state": state_fingerprint(backend, state)}
        write_json(run_root / f"isolation-{probes}.json", {
            **comparison, "inserted_probes": observations, "elapsed_seconds": elapsed,
            "timing_scope": "validation includes state hashing and log I/O; not online latency"})
        if baseline is None:
            baseline = comparison
        else:
            fail_on_difference(run_root, baseline, comparison, phase="main_continuation_isolation", case=case["name"])
        event(run_root, "isolation_passed", inserted_probes=probes,
              continuation_tokens=len(trace), elapsed_seconds=elapsed)
        results.append({"inserted_probes": probes, "continuation_tokens": len(trace), "exact": True})
    return results


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model-dir", type=Path, help="Existing pinned HF snapshot; never downloaded")
    result.add_argument("--evidence-root", type=Path, required=True,
                        help="Read-only pilot20-diagnostic-8192 root containing items/q001,q002")
    result.add_argument("--run-root", type=Path, help="Must not exist; required for GPU execution")
    result.add_argument("--upstream-source-dir", type=Path,
                        help="Export of pinned original method_codestop.py; hash checked, no .git required")
    result.add_argument("--prefix", action="append", dest="prefixes", help="q001:0 etc.; 1-3 unique selectors")
    result.add_argument("--attention-implementation", choices=("eager", "sdpa"), default="eager")
    result.add_argument("--continuation-tokens", type=int, default=64)
    result.add_argument("--seed", type=int, default=872)
    result.add_argument("--inspect-only", action="store_true", help="No CUDA/model load and no output writes")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if not 1 <= args.continuation_tokens <= 64 or not 0 <= args.seed < 2**63:
        raise ValueError("Require continuation tokens in [1,64] and seed in [0,2**63)")
    cases = load_saved_prefixes(args.evidence_root, args.prefixes or DEFAULT_PREFIXES)
    raw_reference, function, source = reference_source(args.upstream_source_dir)
    plan = {"scope": "bounded_online_gpu_acceptance", "tolerance": TOLERANCE,
            "reference": source, "prefixes": [{key: case[key] for key in (
                "name", "stop_token_idx", "prefix_ids_sha256", "input_sha256", "saved_attention")}
                | {"prefix_tokens": len(case["prefix_ids"])} for case in cases],
            "probe_max_new_tokens": MAX_PROBE_TOKENS, "continuation_tokens": args.continuation_tokens,
            "isolation_inserted_probe_counts": [0, 1, 2], "seed": args.seed,
            "prefix_build_method": "original_prompt_prefill_then_saved_token_by_token_extend",
            "attention_implementation": args.attention_implementation,
            "tokenizer_prefix_validation": "pending until pinned tokenizer is loaded",
            "answer_boundary_coverage": "not performed; separate production runner acceptance required",
            "capacity_32k": "not performed"}
    if args.inspect_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.model_dir is None or args.run_root is None:
        raise ValueError("GPU execution requires --model-dir and a new --run-root")
    for protected in (args.evidence_root, args.model_dir):
        if args.run_root.resolve().is_relative_to(protected.resolve()):
            raise ValueError("The new run root must be outside read-only evidence and model directories")
    with fresh_run(args.run_root) as run_root:
        write_json(run_root / "plan.json", plan)
        write_json(run_root / "inputs.json", cases)
        source_paths = [Path(__file__), ROOT / "src" / "torch_online_backend.py",
                        ROOT / "src" / "online_contract.py", ROOT / "src" / "online_protocol.py",
                        ROOT / "scripts" / "run_online_diagnostic.py"]
        write_json(run_root / "source-hashes.json", {str(path.relative_to(ROOT)): sha(path.read_bytes())
                                                   for path in source_paths})
        (run_root / "pinned-reference.py").write_bytes(raw_reference)
        try:
            inventory = gpu_inventory()
            with gpu_lock(inventory["uuid"]) as lock_path:
                idle = assert_gpu_idle(inventory["uuid"])
                write_json(run_root / "gpu-preflight.json", {"gpu": inventory, "idle": idle, "lock": lock_path})
                event(run_root, "model_load_start", python=sys.version, host=platform.node(),
                      attention_implementation=args.attention_implementation, gpu_uuid=inventory["uuid"])
                from torch_online_backend import TorchOnlineBackend
                backend = TorchOnlineBackend(args.model_dir, attention_implementation=args.attention_implementation)
                write_json(run_root / "environment.json", backend.metadata)
                reference = load_reference(function, backend.torch)
                for case in cases:
                    validate_tokenizer_prefix(case, backend)
                event(run_root, "saved_prefixes_verified", count=len(cases), dtype=backend.metadata["model_parameter_dtype"])
                probes = [validate_probe(backend, reference, case, run_root, args.seed) for case in cases]
                isolation = validate_isolation(backend, cases[0], run_root, args.seed, args.continuation_tokens)
                write_json(run_root / "summary.json", {"status": "passed", "scope": plan["scope"],
                    "probe_validation": probes, "isolation_validation": isolation,
                    "answer_boundary_coverage": plan["answer_boundary_coverage"],
                    "capacity_32k": plan["capacity_32k"], "peak_memory_bytes_last_request": backend.peak_memory_bytes()})
                event(run_root, "acceptance_passed", scope=plan["scope"])
                return 0
        except Exception as exc:
            write_json(run_root / "failure.json", {"status": "failed", "error_type": type(exc).__name__,
                                                   "message": str(exc), "traceback": traceback.format_exc()})
            event(run_root, "acceptance_failed", error_type=type(exc).__name__, message=str(exc))
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
