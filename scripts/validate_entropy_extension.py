#!/usr/bin/env python3
"""只重放已保存首个真实21→42追加案例；不生成新主轨迹或E/T答案。"""
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import re
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from online_contract import MODEL_ID, MODEL_REVISION
from run_online_diagnostic import (atomic_new, assert_gpu_idle, derive_seed, encoded,
                                   gpu_inventory, gpu_lock, read_json, sha256)
from run_entropy_value_pilot import gpu_acceptance

MAX_SECONDS, MAX_MAIN_TOKENS = 900, 8192
SHORT_CAP, LONG_CAP = 21, 42
MATCH_FIELDS = ("token_ids", "token_probs", "confidence_raw", "confidence_nonfinite",
                "confidence_valid", "entropy_raw_fp32_nats", "entropy_valid",
                "ended_with_think", "ended_with_eos", "invalid_reason", "actual_length")
REQUIRED_SOURCES = {"src/entropy_probe.py", "src/torch_online_backend.py",
                    "src/online_protocol.py", "src/online_contract.py",
                    "scripts/run_entropy_value_pilot.py", "scripts/run_online_diagnostic.py"}


def check_budget(started, seconds=MAX_SECONDS):
    if type(seconds) not in (int, float) or not 0 < seconds <= MAX_SECONDS:
        raise ValueError("seconds must be positive and at most 900")
    if time.monotonic() - started >= seconds:
        raise TimeoutError("Bounded extension validation exceeded its wall-clock budget")


def verify_sources(manifest, code_root=ROOT):
    hashes = manifest.get("code_sha256", {})
    if not isinstance(hashes, dict) or not REQUIRED_SOURCES <= set(hashes):
        raise ValueError("Manifest lacks required frozen implementation hashes")
    root = Path(code_root).resolve()
    for relative, expected in hashes.items():
        path = Path(relative)
        if (path.is_absolute() or ".." in path.parts or path.parts[0] not in ("src", "scripts")
                or path.suffix != ".py" or not re.fullmatch(r"[0-9a-f]{64}", expected)):
            raise ValueError("Unsafe or invalid manifest source identity")
        actual = (root / path).resolve()
        if not actual.is_relative_to(root) or sha256(actual) != expected:
            raise ValueError(f"Frozen source mismatch: {relative}")
    return dict(hashes)


def valid_snapshot(value):
    if not isinstance(value, dict):
        return False
    ids, probabilities, entropies = (value.get(key, []) for key in
                                     ("token_ids", "token_probs", "entropy_raw_fp32_nats"))
    n, confidence = value.get("actual_length"), value.get("confidence_raw")
    finite = lambda x: type(x) in (int, float) and math.isfinite(x)
    return (type(n) is int and 3 <= n <= LONG_CAP and len(ids) == len(probabilities) == len(entropies) == n
            and all(type(token) is int and token >= 0 for token in ids)
            and all(finite(p) and 0 <= p <= 1 for p in probabilities)
            and all(finite(h) and h >= 0 for h in entropies)
            and finite(confidence) and 0 <= confidence <= 1
            and value.get("invalid_reason") is None and value.get("confidence_valid") is True
            and value.get("entropy_valid") is True and value.get("ended_with_eos") is False)


def substantive_extension(short, long):
    return (valid_snapshot(short) and valid_snapshot(long)
            and short["actual_length"] == SHORT_CAP and long["actual_length"] > SHORT_CAP)


def first_eligible(run_root):
    """只按数字顺序选第一个，禁止依据答案或验证结果换题。"""
    paths = []
    for path in Path(run_root).glob("question-*/candidate-*.json"):
        question = re.fullmatch(r"question-(\d+)", path.parent.name)
        candidate = re.fullmatch(r"candidate-(\d+)\.json", path.name)
        if question and candidate:
            paths.append((int(question[1]), int(candidate[1]), path))
    for question, candidate, path in sorted(paths):
        row = read_json(path)
        if substantive_extension(row.get("short_raw"), row.get("long_raw")):
            additional = row["long_raw"]["actual_length"] - row["short_raw"]["actual_length"]
            if type(row.get("long_additional_tokens")) is not int or row["long_additional_tokens"] != additional:
                raise ValueError("Saved additional-token count contradicts snapshots")
            return question, candidate, path, row
    return None


def load_prefix(selection, manifest):
    question, candidate, path, row = selection
    ids = manifest.get("question_ids", [])
    if not 1 <= question <= len(ids) or candidate < 1 or row.get("candidate_index") != candidate:
        raise ValueError("Candidate numeric identity differs from manifest/path")
    if row.get("question_id") != ids[question - 1]:
        raise ValueError("Candidate question ID differs from frozen order")
    prefix_path = path.with_suffix("").with_name(path.stem + "-partial") / "prefix.json"
    prefix = read_json(prefix_path)
    if prefix.get("question_id") != row["question_id"] or prefix.get("candidate_index") != candidate:
        raise ValueError("Saved prefix identity differs from candidate")
    prompt, main = prefix.get("prompt_token_ids"), prefix.get("main_token_ids")
    if (not isinstance(prompt, list) or not prompt or not isinstance(main, list)
            or not 1 <= len(main) <= MAX_MAIN_TOKENS
            or any(type(token) is not int or token < 0 for token in prompt + main)
            or type(prefix.get("pending_wait")) is not int):
        raise ValueError("Invalid or over-budget saved prefix")
    if row.get("prefix_tokens") != len(main):
        raise ValueError("Saved prefix token position mismatch")
    if hashlib.sha256(encoded(prompt + main)).hexdigest() != row.get("prefix_sha256"):
        raise ValueError("Saved prefix SHA256 mismatch")
    return prefix_path, prefix


def compare_saved(short, long, saved):
    checks = {"substantive_extension": substantive_extension(short, long)}
    for name, current in (("short", short), ("long", long)):
        checks[f"saved_{name}_snapshot_exact"] = current == saved[f"{name}_raw"]
        for field in MATCH_FIELDS:
            checks[f"saved_{name}_{field}"] = current.get(field) == saved[f"{name}_raw"].get(field)
    return checks


def replay_and_validate(backend, prefix, row, manifest, destination, deadline):
    from entropy_probe import start_probe, extend_probe, close_probe
    if prefix["pending_wait"] != backend.markers.wait:
        raise ValueError("Saved pending token is not this backend's Wait")
    forbidden = set(backend.markers.eos_ids) | {backend.markers.end_think}
    if any(token in forbidden for token in prefix["main_token_ids"]):
        raise ValueError("Saved reasoning prefix crosses a natural terminal boundary")
    seed = manifest["seed"]
    backend.start_request(derive_seed(seed["master"], row["question_id"], seed["rollout"], seed["domain"]))
    deadline()
    backend.synchronize()
    started = time.perf_counter()
    state = backend.prefill(prefix["prompt_token_ids"])
    for token in prefix["main_token_ids"]:
        deadline()
        state = backend.extend(state, (token,))
    backend.synchronize()
    replay_ms = (time.perf_counter() - started) * 1000
    deadline()
    rng = backend._generator.get_state().clone()
    expected = backend.sample(state)  # 仅隔离验收抽样，不接入或续写主轨迹。
    backend._generator.set_state(rng)
    before = ([(k.clone(), v.clone()) for k, v in state.cache], state.logits.clone(), rng, expected)
    backend.synchronize()
    probe_started = time.perf_counter()
    session = start_probe(backend, state, max_tokens=LONG_CAP)
    try:
        short = extend_probe(session, SHORT_CAP)
        deadline()
        long = extend_probe(session, LONG_CAP)
    finally:
        close_probe(session)
    backend.synchronize()
    probe_ms = (time.perf_counter() - probe_started) * 1000
    deadline()
    checks = compare_saved(short, long, row)
    # 原验收包含旧21、分段42、直生成42和主分支隔离；单独保存而不覆盖旧gate。
    gpu_acceptance(backend, state, short, long, before, destination)
    deadline()
    gate = read_json(destination)
    checks["same_KV_acceptance"] = gate.get("passed") is True
    return {"checks": checks, "passed": all(checks.values()), "short": short, "long": long,
            "replay_prefix_ms": replay_ms, "split_probe_ms_including_close": probe_ms,
            "rng_note": "Replay accepts saved tokens without replaying main sampling draws; next-sample check tests isolation within this replay."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=MAX_SECONDS)
    args = parser.parse_args(argv)
    started = time.monotonic()
    check_budget(started, args.seconds)
    output = args.output.resolve()
    gate_output = output.with_name(output.stem + ".same-kv.json")
    if output.exists() or gate_output.exists():
        raise FileExistsError("Both validation output and same-KV output must be new")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"scope": "one_saved_prefix_real_extension_GPU_validation_not_new_experiment",
              "status": "failed", "passed": False, "max_seconds": args.seconds,
              "selection_policy": "first eligible candidate in numeric question/candidate order; no retry",
              "validator_sha256": sha256(Path(__file__)), "same_KV_output": str(gate_output)}
    deadline = lambda: check_budget(started, args.seconds)
    hashes = {}
    try:
        root = args.run_root.resolve()
        manifest_path, summary_path = root / "manifest.json", root / "summary.json"
        manifest, summary = read_json(manifest_path), read_json(summary_path)
        hashes.update({str(path): sha256(path) for path in (manifest_path, summary_path)})
        if summary.get("status") not in ("completed", "failed") or summary.get("source_data_integrity") is not True:
            raise ValueError("Collection must be finished with verified source/data identity")
        if (manifest.get("model_id") != MODEL_ID or manifest.get("model_revision") != MODEL_REVISION
                or manifest.get("protocol") != "entropy-value-paired-v1"):
            raise ValueError("Unsupported saved collection/model protocol")
        report["frozen_sources"] = verify_sources(manifest)
        deadline()
        selected = first_eligible(root)
        if selected is None:
            report.update(status="not_covered", reason="No valid saved 21-to-long extension; terminated short probes do not cover continuation")
        else:
            _, _, candidate_path, row = selected
            prefix_path, prefix = load_prefix(selected, manifest)
            hashes.update({str(path): sha256(path) for path in (candidate_path, prefix_path)})
            report["selected"] = {"question_id": row["question_id"], "candidate_index": row["candidate_index"],
                                  "candidate_path": str(candidate_path), "prefix_tokens": len(prefix["main_token_ids"])}
            gpu = gpu_inventory()
            if "4090" not in gpu["name"]:
                raise RuntimeError("Extension validation requires the checked RTX4090")
            with gpu_lock(gpu["uuid"]):
                idle = assert_gpu_idle(gpu["uuid"])
                deadline()
                from torch_online_backend import TorchOnlineBackend
                backend = TorchOnlineBackend(args.model_dir, attention_implementation="eager")
                try:
                    report["environment"] = {"gpu": gpu, "idle_before": idle, "backend": backend.metadata}
                    deadline()
                    report["validation"] = replay_and_validate(backend, prefix, row, manifest, gate_output, deadline)
                finally:
                    del backend
            deadline()
            report.update(status="completed" if report["validation"]["passed"] else "failed",
                          passed=report["validation"]["passed"])
        verify_sources(manifest)
        if any(sha256(Path(path)) != expected for path, expected in hashes.items()):
            raise ValueError("Archived validation input changed during execution")
        if sha256(Path(__file__)) != report["validator_sha256"]:
            raise ValueError("Validator changed during execution")
        deadline()
    except BaseException as exc:
        report.update(status="failed", passed=False,
                      failure={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
    report.update(input_sha256=hashes, elapsed_seconds=time.monotonic() - started)
    atomic_new(output, report)
    print(encoded({"status": report["status"], "passed": report["passed"], "output": str(output)}).decode(), end="")
    return 0 if report["passed"] else 2 if report["status"] == "not_covered" else 1


if __name__ == "__main__":
    raise SystemExit(main())
