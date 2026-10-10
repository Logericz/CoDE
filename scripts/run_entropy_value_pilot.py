#!/usr/bin/env python3
"""有界机制采集：20开发题，最多3个Wait/题，试答21→42。

这是新协议的配对采集，不是 CoDE/EntroCut 全方法比较或线上加速结果。
主轨迹不早停；每个候选的 E/T 分支隔离 KV 和 RNG。未知评分不当成错误。
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from online_contract import MODEL_ID, MODEL_REVISION
from online_protocol import degeneration_score
from run_online_diagnostic import (CODE_FILES, atomic_new, assert_gpu_idle, derive_seed,
    encoded, EventLog, gpu_inventory, gpu_lock, read_json, sha256, verify_data)

PROTOCOL = "entropy-value-paired-v1"
DATA_HASH = "ae236d14cfa3b5cfbf434163386fea878016355051313ccdbd11d0a309eebca7"
QUESTION_COUNT, CANDIDATE_CAP = 20, 3
MAIN_CAP, THINK_CHUNK, FINAL_CAP = 8192, 256, 128
SHORT_CAP, LONG_CAP = 21, 42
TOTAL_SECONDS, QUESTION_SECONDS = 7200, 900
SOURCE_FILES = tuple(dict.fromkeys((*CODE_FILES,
    "scripts/run_entropy_value_pilot.py", "scripts/analyze_entropy_value.py",
    "src/entropy_probe.py", "src/entropy_value_analysis.py")))


def source_hashes():
    """绑定本release实际执行代码；不导入/执行学习注释版上游。"""
    # 显式清单让漏部署的依赖直接报错，而非被glob悄悄忽略。
    return {name: sha256(ROOT / name) for name in SOURCE_FILES}


def select_questions(data_dir):
    """固定selection120中非pilot20的前20题；不得按结果/长度筛换。"""
    data_dir = Path(data_dir).resolve()
    pilot = [json.loads(line) for line in (data_dir / "pilot20.jsonl").read_text().splitlines()]
    _, identity = verify_data(data_dir, DATA_HASH, pilot[0]["id"])
    selection = [json.loads(line) for line in (data_dir / "selection120.jsonl").read_text().splitlines()]
    excluded = {row["id"] for row in pilot}
    selected = [row for row in selection if row["id"] not in excluded][:QUESTION_COUNT]
    if len(selected) != QUESTION_COUNT:
        raise ValueError("Need the fixed twenty non-pilot development questions")
    for row in selected:
        if not row["problem"].strip() or not row["answer"].strip():
            raise ValueError("Missing question/gold; do not silently replace")
    return selected, identity


def make_plan(data_dir):
    rows, identity = select_questions(data_dir)
    return rows, {"schema_version": 1, "protocol": PROTOCOL,
        "scope": "small_development_paired_mechanism_not_online_policy_or_benchmark",
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "attention": "eager", "dtype": "BF16", "data_identity": identity,
        "question_ids": [row["id"] for row in rows],
        "questions": [{"id": row["id"], "problem_sha256": row["problem_sha256"],
                       "row_sha256": hashlib.sha256(encoded(row)).hexdigest(),
                       "source_needs_review": row.get("needs_review", False)} for row in rows],
        "selection": "selection120 frozen order excluding every pilot20 ID, first twenty, no replacement",
        "limits": {"questions": QUESTION_COUNT, "candidates_per_question": CANDIDATE_CAP,
            "main_tokens": MAIN_CAP, "short_probe_tokens": SHORT_CAP,
            "long_probe_tokens": LONG_CAP, "T_main_tokens": THINK_CHUNK,
            "final_tokens": FINAL_CAP, "total_seconds": TOTAL_SECONDS,
            "question_seconds": QUESTION_SECONDS},
        "seed": {"master": 42, "rollout": 0, "domain": "entropy-value-main-v1"},
        "candidate": "first three Wait tokens before natural thinking close/EOS; prefix excludes pending Wait",
        "E": "clone current main prefix, fixed final prefix, greedy up to128 or EOS",
        "T": "clone main prefix and RNG, accept pending Wait, sample at most256 main tokens including Wait; stop before think/EOS; then same F",
        "O": "same isolated probe session from21 to42 or termination; main prefix and RNG untouched",
        "finalizer": "original FINAL_PREFIX, same for E/T, retain full output; no probe text reuse",
        "grading": "frozen strict Math-Verify; unknown is null, no answer repair",
        "history": "D from valid fixed21 observations only; long observations never update history",
        "analysis": {"lambda_error_ms": 60000, "sensitivity_ms": [30000, 120000],
            "outer_folds_by_question": 4, "inner_folds_by_question": 4,
            "predictor": "fixed alpha1 ridge; fold-local preprocessing",
            "target": "L(short-selector)-L(long-selector)-extra_probe_ms",
            "comparison": "same nested-fold target, B confidence/history/probability curve vs H adds entropy",
            "status": "exploratory; no expansion or primary benchmark claim"},
        "failure_policy": "first execution error/OOM/deadline aborts; no retries; no-candidate/invalid/unknown retained",
        "code_sha256": source_hashes()}


def check_deadline(total_start, question_start=None):
    now = time.monotonic()
    if now - total_start >= TOTAL_SECONDS or (question_start is not None and now - question_start >= QUESTION_SECONDS):
        raise TimeoutError("Fixed pilot wall-clock budget exhausted")


def timed(backend, function, *args, **kwargs):
    backend.synchronize()
    start = time.perf_counter()
    value = function(*args, **kwargs)
    backend.synchronize()
    return value, (time.perf_counter() - start) * 1000


def final_answer(backend, state, deadline):
    """同一个F用于全部端点；输出保留注入boxed前缀及后续全部内容。"""
    from entropy_probe import clone_state
    branch = clone_state(backend, state)
    branch = backend.extend(branch, backend.markers.final_prefix)
    ids, stopped = [], "cap"
    for _ in range(FINAL_CAP):
        deadline()
        token = backend.greedy(branch).token_id
        ids.append(token)
        if token in backend.markers.eos_ids:
            stopped = "eos"
            break
        if len(ids) < FINAL_CAP:
            branch = backend.extend(branch, (token,))
    # FINAL_PREFIX begins with </think>; it is outside the answer span.
    prefix = list(backend.markers.final_prefix)
    closing = prefix.index(backend.markers.end_think)
    answer_ids = prefix[closing + 1:] + ids
    content_ids = answer_ids[:-1] if stopped == "eos" else answer_ids
    return {"token_ids": ids, "answer_token_ids": answer_ids,
        "answer_text": backend.decode(content_ids, skip_special_tokens=False),
        "generated_tokens": len(ids), "termination": stopped,
        "boundary_policy": "known_injected_final_prefix_v1", "answer_boundary_confirmed": True}


def continue_then_answer(backend, state, pending_wait, deadline):
    """T消耗独立KV与临时RNG，函数结束后恢复主轨迹的随机状态。"""
    from entropy_probe import clone_state
    rng = backend._generator.get_state().clone()
    try:
        branch = clone_state(backend, state)
        branch = backend.extend(branch, (pending_wait,))
        ids, end = [pending_wait], "chunk"
        for _ in range(THINK_CHUNK - 1):
            deadline()
            token = backend.sample(branch).token_id
            ids.append(token)
            if token in backend.markers.eos_ids or token == backend.markers.end_think:
                end = "natural_boundary_pending"
                break
            branch = backend.extend(branch, (token,))
        result = final_answer(backend, branch, deadline)
        result.update(main_token_ids=ids, main_generated_tokens=len(ids), main_termination=end)
        return result
    finally:
        backend._generator.set_state(rng)


def gpu_acceptance(backend, state, short, long, before, destination):
    """首个真实候选上的唯一验收：同KV数值、分段续写、主状态隔离。"""
    from entropy_probe import start_probe, extend_probe, close_probe
    torch = backend.torch
    cache_before, logits_before, rng_before, expected = before
    unchanged_after_extension = (
        len(cache_before) == len(state.cache)
        and all(torch.equal(a, c) and torch.equal(b, d)
                for (a, b), (c, d) in zip(cache_before, state.cache))
        and torch.equal(logits_before, state.logits)
        and torch.equal(rng_before, backend._generator.get_state()))
    reference = backend.probe(state)
    direct_session = start_probe(backend, state, max_tokens=LONG_CAP)
    try:
        direct = extend_probe(direct_session, LONG_CAP)
    finally:
        close_probe(direct_session)
    actual = backend.sample(state)
    backend._generator.set_state(rng_before)
    confidence_same = reference.confidence_raw == short["confidence_raw"]
    if reference.confidence_raw is not None and math.isnan(reference.confidence_raw):
        confidence_same = short["confidence_raw"] is None
    checks = {
        "extension_preserves_main_state": unchanged_after_extension,
        "legacy_short_no_eos": not short["ended_with_eos"],
        "legacy_short_token_ids": list(reference.token_ids) == short["token_ids"],
        "legacy_short_probabilities": list(reference.token_probs) == short["token_probs"],
        "legacy_short_confidence": confidence_same,
        "long_token_ids": direct["token_ids"] == long["token_ids"],
        "long_probabilities": direct["token_probs"] == long["token_probs"],
        "long_confidence": direct["confidence_raw"] == long["confidence_raw"],
        "long_entropy": direct["entropy_raw_fp32_nats"] == long["entropy_raw_fp32_nats"],
        "main_cache": all(torch.equal(a, c) and torch.equal(b, d)
                          for (a, b), (c, d) in zip(cache_before, state.cache)),
        "main_logits": torch.equal(logits_before, state.logits),
        "main_rng": torch.equal(rng_before, backend._generator.get_state()),
        "next_sample": asdict(expected) == asdict(actual),
    }
    report = {"scope": "same_KV_changed_path_GPU_acceptance_not_method_result",
              "checks": checks, "passed": all(checks.values()),
              "reference": asdict(reference), "short": short, "split_long": long,
              "direct_long": direct}
    atomic_new(destination, report)
    if not report["passed"]:
        raise RuntimeError("GPU acceptance failed; see immutable gpu-acceptance.json")


def observation(snapshot, D):
    return {"confidence": snapshot["confidence_raw"], "D": D,
        "ended_with_think": snapshot["ended_with_think"],
        "max_probabilities": snapshot["token_probs"],
        "entropies_nats": snapshot["entropy_raw_fp32_nats"]}


def collect_question(backend, row, number, root, log, total_start, need_gate):
    from entropy_probe import start_probe, extend_probe, close_probe
    question_start = time.monotonic()
    deadline = lambda: check_deadline(total_start, question_start)
    deadline()
    backend.start_request(derive_seed(42, row["id"], 0, "entropy-value-main-v1"))
    prompt = backend.prompt_ids(row["problem"])
    if len(prompt) + MAIN_CAP + THINK_CHUNK + len(backend.markers.final_prefix) + FINAL_CAP > backend.context_limit:
        raise ValueError("Actual prompt exceeds fixed context reserve")
    state = backend.prefill(prompt)
    generated, records, valid_c, valid_t = [], [], [], []
    previous_D, previous_position, terminal = 0.0, None, "main_cap"
    folder = root / f"question-{number + 1:02d}"
    folder.mkdir()
    log.emit("question_start", question=number + 1, planned_questions=QUESTION_COUNT, sample_id=row["id"])
    for index in range(MAIN_CAP):
        deadline()
        sample = backend.sample(state)
        token = sample.token_id
        generated.append(token)
        if token in backend.markers.eos_ids or token == backend.markers.end_think:
            terminal = "natural_boundary"
            break
        if token == backend.markers.wait:
            if index == 0:
                raise ValueError("First-token Wait has zero prefix length; do not silently skip or shift it")
            candidate = len(records) + 1
            log.emit("candidate_start", question=number + 1, candidate=candidate, prefix_tokens=index)
            evidence = folder / f"candidate-{candidate:02d}-partial"
            evidence.mkdir()
            atomic_new(evidence / "prefix.json", {"question_id": row["id"],
                "candidate_index": candidate, "prompt_token_ids": prompt,
                "main_token_ids": generated[:-1], "pending_wait": token})
            acceptance_before = None
            if need_gate:
                rng = backend._generator.get_state().clone()
                expected = backend.sample(state)
                backend._generator.set_state(rng)
                acceptance_before = ([(k.clone(), v.clone()) for k, v in state.cache],
                                     state.logits.clone(), rng, expected)
            def short_probe():
                session = start_probe(backend, state, max_tokens=LONG_CAP)
                try:
                    return session, extend_probe(session, SHORT_CAP)
                except BaseException:
                    close_probe(session)
                    raise
            (session, short), short_ms = timed(backend, short_probe)
            atomic_new(evidence / "short.json", {"observation": short, "elapsed_ms": short_ms})
            try:
                long, extra_ms = timed(backend, extend_probe, session, LONG_CAP)
            finally:
                _, close_ms = timed(backend, close_probe, session)
            extra_ms += close_ms
            atomic_new(evidence / "long.json", {"observation": long, "extra_ms": extra_ms})
            if need_gate:
                gpu_acceptance(backend, state, short, long, acceptance_before, root / "gpu-acceptance.json")
                del acceptance_before
                need_gate = False
                log.emit("gpu_acceptance_passed", question=number + 1, candidate=candidate)
            history = {"valid_count": len(valid_c), "previous_confidence": valid_c[-1] if valid_c else None,
                "previous_D": previous_D, "candidate_gap": 1 if records else None,
                "token_gap": index - previous_position if previous_position is not None else None}
            c = short["confidence_raw"]
            if short["invalid_reason"] is None and c is not None and math.isfinite(c) and 0 <= c <= 1:
                valid_c.append(c)
                valid_t.append(index)
                D = degeneration_score(valid_t, valid_c)
            else:
                D = None
            # 两个端点交替换序，降低固定先后顺序的计时偏差。
            action_results = {}
            for action in (("E", "T") if (number + candidate) % 2 == 0 else ("T", "E")):
                deadline()
                fn = (lambda: final_answer(backend, state, deadline)) if action == "E" else (
                    lambda: continue_then_answer(backend, state, token, deadline))
                result, elapsed = timed(backend, fn)
                action_results[action] = {**result, "remaining_ms": elapsed, "error": None}
                atomic_new(evidence / f"{action}.json", action_results[action])
            record = {"question_id": row["id"], "candidate_index": candidate,
                "status": "ok" if short["invalid_reason"] is None and long["invalid_reason"] is None and D is not None else "invalid_probe",
                "prefix_tokens": index, "prefix_sha256": hashlib.sha256(encoded(prompt + generated[:-1])).hexdigest(),
                "history": history, "short": observation(short, D), "long": observation(long, D),
                "short_raw": short, "long_raw": long, "actions": action_results,
                "short_probe_ms": short_ms, "extra_probe_ms": extra_ms,
                "source_needs_review": row.get("needs_review", False),
                "long_additional_tokens": long["actual_length"] - short["actual_length"]}
            atomic_new(folder / f"candidate-{candidate:02d}.json", record)
            records.append(record)
            previous_D, previous_position = D, index
            log.emit("candidate_saved", question=number + 1, candidate=candidate,
                prefix_tokens=index, short_length=short["actual_length"], long_length=long["actual_length"],
                short_ms=round(short_ms, 2), extra_ms=round(extra_ms, 2),
                E_ms=round(action_results["E"]["remaining_ms"], 2), T_ms=round(action_results["T"]["remaining_ms"], 2))
            if len(records) == CANDIDATE_CAP:
                terminal = "candidate_cap"
                break
        state = backend.extend(state, (token,))
        if (index + 1) % 256 == 0:
            log.emit("main_progress", question=number + 1, generated=index + 1,
                     candidates=len(records), elapsed_seconds=round(time.monotonic() - question_start, 1))
    summary = {"question_id": row["id"], "status": "completed", "candidates": len(records),
        "terminal": terminal, "main_generated_tokens": len(generated),
        "main_token_ids": generated, "prompt_token_ids": prompt,
        "elapsed_seconds": time.monotonic() - question_start,
        "peak_memory_bytes": backend.peak_memory_bytes()}
    atomic_new(folder / "summary.json", summary)
    log.emit("question_completed", **{k: v for k, v in summary.items() if not k.endswith("token_ids")})
    return records, summary, need_gate


def grade_saved(root, rows, records, deadline=lambda: None):
    """评分另存，原candidate.json不回写；unknown保留为null。"""
    from math_grading import _check_dependencies, grade_record
    versions = _check_dependencies()
    by_id = {row["id"]: row for row in rows}
    grades = []
    completed = False
    try:
        for record in records:
            row = by_id[record["question_id"]]
            for action, result in record["actions"].items():
                deadline()
                grade = grade_record({"id": f"{row['id']}:{record['candidate_index']}:{action}",
                    "gold": row["answer"], "answer_text": result["answer_text"],
                    "execution_status": "completed", "answer_boundary_confirmed": True,
                    "boundary_policy": result["boundary_policy"]})
                if row.get("needs_review"):
                    grade.update(status="needs_review", reason="source_gold_requires_review", grade=None)
                result["error"] = 0 if grade.get("grade") is True else 1 if grade.get("grade") is False else None
                result["grading_status"] = grade["status"]
                grades.append(grade)
        completed = True
    finally:
        atomic_new(root / "grading.json", {"dependencies": versions, "results": grades,
                                          "completed": completed})
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args(argv)
    started = time.monotonic()
    rows, plan = make_plan(args.data_dir)
    if args.inspect_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if args.model_dir is None or args.run_root is None:
        parser.error("GPU execution requires --model-dir and a new --run-root")
    root = args.run_root.resolve()
    if root.exists() or any(root.is_relative_to(p.resolve()) for p in (args.model_dir, args.data_dir)):
        raise ValueError("Run root must be new and outside model/data")
    root.mkdir(parents=True)
    atomic_new(root / "manifest.json", plan)
    log = EventLog(root / "events.jsonl")
    records, summaries, failure, grading_failure = [], [], None, None
    need_gate = True
    try:
        from math_grading import _check_dependencies
        _check_dependencies()
        gpu = gpu_inventory()
        if "4090" not in gpu["name"]:
            raise RuntimeError("This bounded pilot is specified for the checked RTX4090")
        with gpu_lock(gpu["uuid"]):
            idle = assert_gpu_idle(gpu["uuid"])
            from torch_online_backend import TorchOnlineBackend
            backend = TorchOnlineBackend(args.model_dir, attention_implementation="eager")
            try:
                atomic_new(root / "environment.json", {"gpu": gpu, "idle_before": idle,
                                                     "backend": backend.metadata})
                log.emit("model_loaded", gpu=gpu["name"], seconds=round(time.monotonic() - started, 1))
                for number, row in enumerate(rows):
                    collected, summary, need_gate = collect_question(backend, row, number, root, log, started, need_gate)
                    records.extend(collected)
                    summaries.append(summary)
            finally:
                del backend
        if need_gate:
            raise RuntimeError("No candidate reached; changed GPU path has not been validated")
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        atomic_new(root / "failure.json", failure)
        log.emit("generation_failed", **failure)
    # Recover every durable candidate, including those before a partial-question failure.
    records = [read_json(path) for path in sorted(root.glob("question-*/candidate-*.json"))]
    try:
        records = grade_saved(root, rows, records, deadline=lambda: check_deadline(started))
    except BaseException as exc:
        grading_failure = {"type": type(exc).__name__, "message": str(exc)}
        atomic_new(root / "grading-failure.json", grading_failure)
    try:
        integrity = source_hashes() == plan["code_sha256"]
        _, after = select_questions(args.data_dir)
        integrity = integrity and after == plan["data_identity"]
    except Exception as exc:
        integrity = False
        atomic_new(root / "integrity-failure.json", {"type": type(exc).__name__, "message": str(exc)})
    analysis = {"schema_version": "entropy-value-v1", "question_ids": plan["question_ids"], "records": records}
    atomic_new(root / "analysis-input.json", analysis)
    summary = {"protocol": PROTOCOL, "status": "completed" if failure is None and grading_failure is None and integrity else "failed",
        "planned_questions": QUESTION_COUNT, "completed_questions": len(summaries),
        "max_planned_candidates": QUESTION_COUNT * CANDIDATE_CAP, "saved_candidates": len(records),
        "question_summaries": summaries, "failure": failure, "grading_failure": grading_failure,
        "source_data_integrity": integrity,
        "gpu_acceptance_passed": (root / "gpu-acceptance.json").exists() and read_json(root / "gpu-acceptance.json")["passed"],
        "elapsed_seconds": time.monotonic() - started,
        "analysis_input_sha256": sha256(root / "analysis-input.json")}
    atomic_new(root / "summary.json", summary)
    log.emit("pilot_finished", status=summary["status"], saved_candidates=len(records),
             completed_questions=len(summaries), elapsed_seconds=round(summary["elapsed_seconds"], 1))
    log.close()
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
