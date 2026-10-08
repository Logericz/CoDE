#!/usr/bin/env python3
"""Diagnose one saved exact-parity failure without relaxing its acceptance gate.

A: incrementally rebuilt main KV + common probe.
B: bulk-prefix main KV + common probe.
C: immutable full-prefix original-probe evidence from the failed run (reused).
D: original probe with precisely A's main KV, using pinned cache_to_device and
   deepcopy. A/D share the same state; their probes must leave it unchanged.

At most three new comparison probes are executed. A separate 0/1/2-probe
isolation diagnostic follows on the same loaded model, with at most 64 main
tokens per variant. All numerical comparisons remain exact. Completing this
diagnostic never means the overall GPU acceptance or latency benchmark passed.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import validate_online_gpu as validation
from run_online_diagnostic import assert_gpu_idle, atomic_new, gpu_inventory, gpu_lock

CACHE_FILE = "cache_utils.py"
CACHE_SHA256 = "e4bcea4a8c9bb5f76a4e2cacf7070c4c302de4272645d974e1de008636f73d04"
SCOPE = "single_failed_prefix_parity_diagnosis_not_gpu_acceptance"
COMPARABLE_FIELDS = ("token_ids", "token_probs", "confidence", "ended")
ENVIRONMENT_FIELDS = (
    "model_id", "model_revision", "tokenizer_revision", "model_parameter_dtype",
    "attention_implementation", "torch_version", "transformers_version", "cuda_version",
    "gpu_name", "driver_version", "snapshot_config_sha256", "tokenizer_sha256", "weight_shards",
    "markers", "sampling", "probe", "main_logits_to_keep", "context_limit",
)
CRITICAL_SOURCE_FILES = ("src/torch_online_backend.py", "src/online_contract.py", "src/online_protocol.py")


def cache_source(upstream_source_dir=None):
    if upstream_source_dir is None:
        raw = subprocess.run(
            ["git", "-C", str(ROOT / "upstream" / "CoDE-Stop"), "show",
             f"{validation.UPSTREAM_COMMIT}:{CACHE_FILE}"],
            check=True, capture_output=True).stdout
        kind = "pinned_git_object"
    else:
        raw = (Path(upstream_source_dir) / CACHE_FILE).read_bytes()
        kind = "sha256_verified_git_object_export"
    if validation.sha(raw) != CACHE_SHA256:
        raise ValueError("cache_utils.py differs from the frozen upstream Git object")
    functions = [node for node in ast.parse(raw).body
                 if isinstance(node, ast.FunctionDef) and node.name == "cache_to_device"]
    if len(functions) != 1:
        raise ValueError("Expected the pinned cache_to_device function exactly once")
    return raw, functions[0], {"file": CACHE_FILE, "sha256": CACHE_SHA256,
                              "upstream_commit": validation.UPSTREAM_COMMIT, "source_kind": kind}


def load_cached_reference(probe_function, cache_function, torch):
    """Execute verified original ASTs; do not replace their cache semantics."""
    import transformers
    from transformers.cache_utils import DynamicCache
    namespace = {"torch": torch, "F": torch.nn.functional, "deepcopy": deepcopy,
                 "transformers": transformers, "DynamicCache": DynamicCache}
    exec(compile(ast.Module(body=[cache_function], type_ignores=[]),
                 f"{validation.UPSTREAM_COMMIT}:{CACHE_FILE}", "exec"), namespace)
    exec(compile(ast.Module(body=[probe_function], type_ignores=[]),
                 f"{validation.UPSTREAM_COMMIT}:{validation.REFERENCE_FILE}", "exec"), namespace)
    return namespace["calcu_max_probs_w_kv"]


def comparable(record):
    result = {key: record[key] for key in COMPARABLE_FIELDS}
    tokens, probabilities = result["token_ids"], result["token_probs"]
    if not isinstance(tokens, list) or not 1 <= len(tokens) <= validation.MAX_PROBE_TOKENS:
        raise ValueError("Every comparison path must contain 1-21 probe tokens")
    if not isinstance(probabilities, list) or len(probabilities) != len(tokens):
        raise ValueError("Probe token/probability lengths disagree")
    if any(type(token) is not int or token < 0 for token in tokens) or type(result["ended"]) is not bool:
        raise ValueError("Invalid saved probe token IDs or ended flag")
    return result


def load_failed_evidence(failure_root, case):
    """Validate links before reusing C; no former output is changed or resumed."""
    root = Path(failure_root).resolve()
    filenames = ("plan.json", "inputs.json", "environment.json", "source-hashes.json",
                 "first-difference.json", f"{case['name']}-online-probe.json",
                 f"{case['name']}-reference-probe.json")
    records, hashes, raw_files = {}, {}, {}
    for filename in filenames:
        raw = (root / filename).read_bytes()
        records[filename], hashes[filename], raw_files[filename] = json.loads(raw), validation.sha(raw), raw
    plan, first = records["plan.json"], records["first-difference.json"]
    if (plan.get("scope") != "bounded_online_gpu_acceptance"
            or plan.get("reference", {}).get("sha256") != validation.REFERENCE_SHA256
            or plan.get("reference", {}).get("upstream_commit") != validation.UPSTREAM_COMMIT
            or plan.get("tolerance") != validation.TOLERANCE):
        raise ValueError("Reused evidence must be from the unchanged pinned exact-parity gate")
    if first.get("phase") != "full_prefix_probe_numerics" or first.get("case") != case["name"]:
        raise ValueError("Selected prefix is not the saved full-prefix numerical failure")
    selected = [item for item in records["inputs.json"] if item.get("name") == case["name"]]
    if len(selected) != 1:
        raise ValueError("Failed-run inputs do not identify exactly one selected prefix")
    for key in ("prefix_ids", "prompt_ids", "prefix_ids_sha256", "stop_token_idx"):
        if selected[0].get(key) != case.get(key):
            raise ValueError(f"Failed-run and frozen source inputs differ: {key}")
    # Source locations can differ after evidence transfer; content hashes must not.
    if sorted(selected[0]["input_sha256"].values()) != sorted(case["input_sha256"].values()):
        raise ValueError("Failed-run raw stage hashes do not match the current saved evidence")
    for filename in CRITICAL_SOURCE_FILES:
        if records["source-hashes.json"].get(filename) != validation.sha((ROOT / filename).read_bytes()):
            raise ValueError(f"Backend/protocol source changed since the failed run: {filename}")
    original = comparable(records[f"{case['name']}-reference-probe.json"])
    online = comparable(records[f"{case['name']}-online-probe.json"])
    actual_difference = validation.first_difference(original, online)
    if actual_difference is None or any(first.get(key) != actual_difference[key]
                                        for key in ("field", "expected", "actual")):
        raise ValueError("The saved first difference does not agree with its raw probe records")
    attention = plan.get("attention_implementation")
    seed = plan.get("seed")
    if attention not in ("eager", "sdpa") or type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Failed run lacks a valid frozen attention implementation or seed")
    return {"root": str(root), "hashes": hashes, "records": records, "raw_files": raw_files,
            "C": original, "attention": attention, "seed": seed}


def verify_environment(previous, current):
    for field in ENVIRONMENT_FIELDS:
        if field not in previous or field not in current or previous[field] != current[field]:
            raise ValueError(f"Cannot reuse full-prefix reference across changed environment: {field}")
    if current["model_parameter_dtype"] != "torch.bfloat16":
        raise ValueError("The diagnostic requires unchanged BF16 model arithmetic")


def pack_common(observation):
    return {"token_ids": list(observation.token_ids), "token_probs": list(observation.token_probs),
            "confidence": observation.confidence_raw, "ended": observation.ended_with_think,
            "confidence_valid": observation.confidence_valid,
            "invalid_reasons": observation.invalid_reasons, "source": observation.confidence_source}


def record_path(backend, run_root, label, record, *, seconds, before, after, **extra):
    values = comparable(record)
    report = {**record, "label": label, "elapsed_seconds": seconds, "state_before": before,
              "state_after": after, "probability_bf16_bits": validation.bf16_bits(backend.torch, values["token_probs"]),
              "confidence_bf16_bits": validation.bf16_bits(backend.torch, [values["confidence"]]), **extra}
    validation.write_json(run_root / f"path-{label}.json", report)
    validation.fail_on_difference(run_root, before, after, phase=f"path_{label}_state_isolation", case=label)
    validation.event(run_root, "path_finished", path=label, probe_tokens=len(values["token_ids"]),
                     confidence=values["confidence"], ended=values["ended"], elapsed_seconds=seconds)
    return values


def compare_paths(expected, actual, expected_label, actual_label):
    difference = validation.first_difference(expected, actual)
    return {"expected_path": expected_label, "actual_path": actual_label,
            "status": "passed" if difference is None else "failed", "exact": difference is None,
            "first_difference": difference,
            "token_ids_identical": expected["token_ids"] == actual["token_ids"],
            "confidence_identical": validation.first_difference(expected["confidence"], actual["confidence"]) is None,
            "ended_identical": expected["ended"] == actual["ended"],
            "probability_differences": [
                {"index": index, "expected": left, "actual": right, "absolute_difference": abs(left - right)}
                for index, (left, right) in enumerate(zip(expected["token_probs"], actual["token_probs"]))
                if validation.first_difference(left, right) is not None]}


def run_paths(backend, reference, case, run_root, reused, seed, previous_online=None):
    """A then D on the very same KV; B uses a separately initialized request."""
    paths = {"C": reused}
    backend.start_request(seed)
    validation.event(run_root, "path_start", path="A", mode="incremental_saved_token_KV_common_probe")
    backend.synchronize()
    start = time.perf_counter()
    state = validation.rebuild_online_prefix(backend, case, run_root)
    backend.synchronize()
    build_seconds = time.perf_counter() - start
    before = validation.state_fingerprint(backend, state)
    incremental_before = before
    backend.synchronize()
    start = time.perf_counter()
    observation = backend.probe(state)
    backend.synchronize()
    elapsed = time.perf_counter() - start
    paths["A"] = record_path(backend, run_root, "A", pack_common(observation), seconds=elapsed,
        before=before, after=validation.state_fingerprint(backend, state),
        prefix_build_seconds=build_seconds, prefix_build_method="prompt_prefill_then_saved_token_by_token_extend")

    validation.event(run_root, "path_start", path="D", mode="original_probe_on_exact_A_main_KV")
    # Passing this actual state.cache is intentional. The original function itself
    # deep-copies it and invokes the verified original cache_to_device helper.
    same_kv_before = validation.state_fingerprint(backend, state)
    validation.fail_on_difference(run_root, before, same_kv_before, phase="same_KV_precondition", case=case["name"])
    backend.synchronize()
    start = time.perf_counter()
    result = reference(backend.model,
        backend.torch.tensor([list(backend.markers.trial_prefix)], device=backend.device),
        state.cache, backend.tokenizer, method=0)
    backend.synchronize()
    elapsed = time.perf_counter() - start
    original = {"token_ids": result["token_ids"], "token_probs": result["token_probs"],
                "confidence": result["total_prob_max"], "ended": bool(result["ended_with_think"])}
    paths["D"] = record_path(backend, run_root, "D", original, seconds=elapsed,
        before=same_kv_before, after=validation.state_fingerprint(backend, state),
        shares_main_KV_with="A", reference_cache_policy="original deepcopy plus original cache_to_device")
    del state

    backend.start_request(seed)
    validation.event(run_root, "path_start", path="B", mode="bulk_saved_prefix_KV_common_probe")
    backend.synchronize()
    start = time.perf_counter()
    state = backend.prefill(case["prefix_ids"])
    backend.synchronize()
    build_seconds = time.perf_counter() - start
    before = validation.state_fingerprint(backend, state)
    backend.synchronize()
    start = time.perf_counter()
    observation = backend.probe(state)
    backend.synchronize()
    elapsed = time.perf_counter() - start
    paths["B"] = record_path(backend, run_root, "B", pack_common(observation), seconds=elapsed,
        before=before, after=validation.state_fingerprint(backend, state),
        prefix_build_seconds=build_seconds, prefix_build_method="bulk_saved_prefix_prefill")
    del state
    reports = {"full_prefix_parity": compare_paths(paths["C"], paths["A"], "C", "A"),
               "same_KV_parity": compare_paths(paths["D"], paths["A"], "D", "A"),
               "bulk_prefix_parity": compare_paths(paths["C"], paths["B"], "C", "B"),
               "incremental_vs_bulk": compare_paths(paths["B"], paths["A"], "B", "A")}
    reports["incremental_vs_bulk"]["prefix_state"] = {
        "KV_identical": incremental_before["kv"] == before["kv"],
        "logits_identical": incremental_before["logits"] == before["logits"],
        "first_KV_fingerprint_difference": validation.first_difference(before["kv"], incremental_before["kv"]),
        "first_logits_fingerprint_difference": validation.first_difference(before["logits"], incremental_before["logits"]),
        "differing_KV_layers": [index for index, (bulk, incremental) in
            enumerate(zip(before["kv"], incremental_before["kv"])) if bulk != incremental],
        "interpretation": "fingerprint comparison localizes a path difference; it does not establish its numerical cause"}
    if previous_online is not None:
        reports["online_replay_reproducibility"] = compare_paths(previous_online, paths["A"], "a01_A", "A")
    validation.write_json(run_root / "path-comparisons.json", reports)
    for comparison, result in reports.items():
        validation.event(run_root, "path_comparison", comparison=comparison,
                         status=result["status"], first_difference=result["first_difference"])
    return reports


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model-dir", type=Path)
    result.add_argument("--evidence-root", type=Path, required=True)
    result.add_argument("--failure-root", type=Path, required=True)
    result.add_argument("--run-root", type=Path)
    result.add_argument("--upstream-source-dir", type=Path)
    result.add_argument("--prefix", default="q001:0", help="Exactly one already failed saved prefix")
    result.add_argument("--inspect-only", action="store_true", help="Read and hash only; no GPU or output writes")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    cases = validation.load_saved_prefixes(args.evidence_root, [args.prefix])
    case = cases[0]
    reused = load_failed_evidence(args.failure_root, case)
    raw_probe, probe_function, probe_metadata = validation.reference_source(args.upstream_source_dir)
    raw_cache, cache_function, cache_metadata = cache_source(args.upstream_source_dir)
    plan = {"scope": SCOPE, "overall_GPU_acceptance": "not_established_by_this_diagnostic",
            "prefix": case["name"], "prefix_ids_sha256": case["prefix_ids_sha256"],
            "prefix_tokens": len(case["prefix_ids"]), "new_comparison_paths": ["A", "D", "B"],
            "reused_path": "C", "reused_evidence_root": reused["root"],
            "reused_file_sha256": reused["hashes"], "reference_sources": [probe_metadata, cache_metadata],
            "tolerance": validation.TOLERANCE, "seed": reused["seed"],
            "attention_implementation": reused["attention"], "probe_max_new_tokens": validation.MAX_PROBE_TOKENS,
            "isolation_inserted_probes": [0, 1, 2], "isolation_continuation_cap": 64,
            "maximum_new_probe_calls": 6, "maximum_new_probe_tokens": 126,
            "maximum_main_continuation_tokens": 192, "timing_scope": "diagnostic only; no speed claims"}
    if args.inspect_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.model_dir is None or args.run_root is None:
        raise ValueError("Execution requires --model-dir and a new --run-root")
    for protected in (args.evidence_root, args.failure_root, args.model_dir):
        if args.run_root.resolve().is_relative_to(protected.resolve()):
            raise ValueError("The new run root must be outside read-only source and model directories")
    with validation.fresh_run(args.run_root) as run_root:
        validation.write_json(run_root / "plan.json", plan)
        validation.write_json(run_root / "input.json", case)
        validation.write_json(run_root / "reused-evidence.json", reused["records"])
        source_files = ("scripts/diagnose_probe_parity.py", "scripts/validate_online_gpu.py",
                        "scripts/run_online_diagnostic.py", *CRITICAL_SOURCE_FILES)
        validation.write_json(run_root / "source-hashes.json", {
            filename: validation.sha((ROOT / filename).read_bytes()) for filename in source_files})
        atomic_new(run_root / "pinned-probe.py", raw_probe, raw=True)
        atomic_new(run_root / "pinned-cache-utils.py", raw_cache, raw=True)
        atomic_new(run_root / "path-C-reused-reference.json",
                   reused["raw_files"][f"{case['name']}-reference-probe.json"], raw=True)
        try:
            inventory = gpu_inventory()
            with gpu_lock(inventory["uuid"]) as lock_path:
                idle = assert_gpu_idle(inventory["uuid"])
                validation.write_json(run_root / "gpu-preflight.json", {"gpu": inventory, "lock": lock_path, "idle": idle})
                validation.event(run_root, "model_load_start", gpu_uuid=inventory["uuid"], attention=reused["attention"])
                from torch_online_backend import TorchOnlineBackend
                backend = TorchOnlineBackend(args.model_dir, attention_implementation=reused["attention"])
                validation.write_json(run_root / "environment.json", backend.metadata)
                verify_environment(reused["records"]["environment.json"], backend.metadata)
                validation.validate_tokenizer_prefix(case, backend)
                reference = load_cached_reference(probe_function, cache_function, backend.torch)
                previous_online = comparable(reused["records"][f"{case['name']}-online-probe.json"])
                reports = run_paths(backend, reference, case, run_root, reused["C"], reused["seed"], previous_online)
                validation.event(run_root, "independent_isolation_start", continuation_cap=64)
                try:
                    isolation_rows = validation.validate_isolation(backend, cases[0], run_root, reused["seed"], 64)
                    isolation = {"status": "passed", "variants": isolation_rows}
                except validation.AcceptanceFailure as exc:
                    isolation = {"status": "failed", "error": str(exc), "evidence": "first-difference.json"}
                summary = {"status": "diagnostic_completed", "scope": SCOPE, **reports,
                           "isolation": isolation, "overall_GPU_acceptance": "not_established",
                           "strict_full_prefix_gate": reports["full_prefix_parity"]["status"],
                           "timing_interpretation": "validation overhead included; not online speed evidence",
                           "answer_boundary_coverage": "not performed", "capacity_32k": "not performed"}
                validation.write_json(run_root / "summary.json", summary)
                validation.event(run_root, "diagnostic_completed", full_prefix=summary["strict_full_prefix_gate"],
                    same_KV=reports["same_KV_parity"]["status"], isolation=isolation["status"],
                    overall_GPU_acceptance="not_established")
                # Exit 2 retains the strict failed gate; exit 0 only means the
                # requested diagnostic completed with these local comparisons equal.
                return 2 if any(row["status"] == "failed" for row in (*reports.values(), isolation)) else 0
        except Exception as exc:
            validation.write_json(run_root / "failure.json", {"status": "diagnostic_interrupted", "scope": SCOPE,
                "error_type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(),
                "overall_GPU_acceptance": "not_established"})
            validation.event(run_root, "diagnostic_interrupted", error_type=type(exc).__name__, message=str(exc))
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
