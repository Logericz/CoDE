#!/usr/bin/env python3
"""冻结短融合预测器后，小规模采集异学科 E/T 配对结果；不训练、不选优。

真正生成、KV/RNG 隔离和 GPU 验收均复用旧采集器。这里只改变选题与总时限，
长试答仍为协议兼容而采集，不能进入本次短融合预测。所有输出只写新目录。
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_entropy_value_pilot as old

PROTOCOL = "entropy-frozen-short-fusion-explore-v1"
QUESTION_CAP, PER_SUBJECT = 14, 2
TOTAL_SECONDS, QUESTION_SECONDS, GRADING_SECONDS = 3000, 900, 90
SOURCE_FILES = tuple(dict.fromkeys((*old.SOURCE_FILES,
    "scripts/run_entropy_fusion_explore.py", "src/entropy_frozen_fusion.py",
    "src/entropy_short_features.py", "src/entropy_short_fusion.py")))


def source_hashes():
    return {name: old.sha256(ROOT / name) for name in SOURCE_FILES}


def check_gpu_model(name):
    """允许已核验租赁型号；运行环境记录真实型号，不沿用旧4090标签。"""
    normalized = " ".join(name.upper().split())
    if normalized not in {"NVIDIA GEFORCE RTX 4090", "NVIDIA GEFORCE RTX 5090",
                          "GEFORCE RTX 4090", "GEFORCE RTX 5090", "RTX 4090", "RTX 5090"}:
        raise RuntimeError("This bounded exploration requires a checked RTX 4090 or RTX 5090")


def select_questions(data_dir):
    """排除两批已暴露题；类别字母序轮转，每类冻结顺序前两题，不补齐。"""
    data_dir = Path(data_dir).resolve()
    previous, identity = old.select_questions(data_dir)
    pilot = [json.loads(line) for line in (data_dir / "pilot20.jsonl").read_text().splitlines()]
    selection = [json.loads(line) for line in (data_dir / "selection120.jsonl").read_text().splitlines()]
    excluded = {row["id"] for row in (*pilot, *previous)}
    if len({row["id"] for row in selection}) != len(selection):
        raise ValueError("Duplicate IDs in frozen selection")
    subjects = sorted({row["subject"] for row in selection})
    if len(subjects) != 7:
        raise ValueError("Expected the frozen seven subjects")
    by_subject = {subject: [row for row in selection
        if row["subject"] == subject and row["id"] not in excluded][:PER_SUBJECT]
        for subject in subjects}
    selected = [by_subject[subject][index] for index in range(PER_SUBJECT)
        for subject in subjects if index < len(by_subject[subject])]
    if not selected or len(selected) > QUESTION_CAP:
        raise ValueError("No eligible questions or question cap exceeded")
    for row in selected:
        if not row["problem"].strip() or not row["answer"].strip():
            raise ValueError("Missing question/gold; do not silently replace")
    return selected, identity


def bounded_seconds(value):
    value = int(value)
    if not 60 <= value <= TOTAL_SECONDS:
        raise argparse.ArgumentTypeError("max-seconds must be between 60 and 3000")
    return value


def make_plan(data_dir, frozen_selector, max_seconds=TOTAL_SECONDS):
    max_seconds = bounded_seconds(max_seconds)
    rows, identity = select_questions(data_dir)
    frozen_selector = Path(frozen_selector).resolve()
    if not isinstance(old.read_json(frozen_selector), dict):
        raise ValueError("Frozen selector must be a pre-existing JSON object")
    _, original_plan = old.make_plan(data_dir)
    plan = dict(original_plan)
    plan.update(protocol=PROTOCOL,
        scope="prospective_cross_subject_exploration_not_full_paper_or_online_policy",
        gpu_policy={"allowed_models": ["RTX 4090", "RTX 5090"],
            "actual_gpu_from_environment": True,
            "comparison_scope": "cross_hardware_exploration; cached 4090 action times and new actual GPU times are not an online speedup comparison"},
        question_ids=[row["id"] for row in rows],
        questions=[{"id": row["id"], "subject": row["subject"],
            "problem_sha256": row["problem_sha256"],
            "row_sha256": hashlib.sha256(old.encoded(row)).hexdigest(),
            "source_needs_review": row.get("needs_review", False)} for row in rows],
        selection="selection120 excludes all pilot20 and prior entropy20; alphabetical subject round-robin, first two per subject in frozen row order; no replacement",
        frozen_selector={"path": str(frozen_selector), "sha256": old.sha256(frozen_selector)},
        code_sha256=source_hashes(),
        analysis={"training": "none; selectors frozen before collection",
            "prediction": "offline frozen short-only C/H/CH and fixed fusion rules",
            "long_observation": "collected for original protocol compatibility; never a short-fusion feature",
            "lambda_error_ms": 60000, "sensitivity_ms": [30000, 120000],
            "status": "bounded exploratory prospective questions; not an online policy timing result"},
        failure_policy="first execution error/OOM or total/per-question deadline stops; durable candidates retained; no retries or replacement",
        cpu_grading={"seconds": GRADING_SECONDS,
            "clock": "fresh monotonic clock after GPU collection and backend release",
            "unknown_policy": "null; strict original scoring preserved; no boundary repairs here"},
        module_overrides={"QUESTION_COUNT": len(rows), "TOTAL_SECONDS": max_seconds,
            "QUESTION_SECONDS": QUESTION_SECONDS})
    plan["limits"] = dict(original_plan["limits"], questions=len(rows), question_cap=QUESTION_CAP,
        total_seconds=max_seconds, question_seconds=QUESTION_SECONDS)
    return rows, plan


@contextmanager
def collection_limits(question_count, max_seconds):
    """旧函数从其模块全局读取预算；限定作用域并在异常后还原。"""
    names = {"QUESTION_COUNT": question_count, "TOTAL_SECONDS": max_seconds,
             "QUESTION_SECONDS": QUESTION_SECONDS}
    before = {name: getattr(old, name) for name in names}
    try:
        for name, value in names.items():
            setattr(old, name, value)
        yield
    finally:
        for name, value in before.items():
            setattr(old, name, value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--frozen-selector", type=Path, required=True)
    parser.add_argument("--max-seconds", type=bounded_seconds, default=TOTAL_SECONDS)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args(argv)
    rows, plan = make_plan(args.data_dir, args.frozen_selector, args.max_seconds)
    if args.inspect_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if args.model_dir is None or args.run_root is None:
        parser.error("GPU execution requires --model-dir and a new --run-root")
    root = args.run_root.resolve()
    if root.exists() or any(root.is_relative_to(p.resolve()) for p in (args.model_dir, args.data_dir)):
        raise ValueError("Run root must be new and outside model/data")
    root.mkdir(parents=True)
    old.atomic_new(root / "manifest.json", plan)
    log = old.EventLog(root / "events.jsonl")
    started = time.monotonic()
    summaries, failure, budget_stop, grading_failure = [], None, None, None
    need_gate = True
    try:
        from math_grading import _check_dependencies
        _check_dependencies()
        gpu = old.gpu_inventory()
        check_gpu_model(gpu["name"])
        with old.gpu_lock(gpu["uuid"]), collection_limits(len(rows), args.max_seconds):
            idle = old.assert_gpu_idle(gpu["uuid"])
            from torch_online_backend import TorchOnlineBackend
            backend = TorchOnlineBackend(args.model_dir, attention_implementation="eager")
            try:
                old.atomic_new(root / "environment.json", {"gpu": gpu, "idle_before": idle,
                    "backend": backend.metadata})
                log.emit("model_loaded", gpu=gpu["name"], seconds=round(time.monotonic() - started, 1))
                for number, row in enumerate(rows):
                    _, summary, need_gate = old.collect_question(
                        backend, row, number, root, log, started, need_gate)
                    summaries.append(summary)
            finally:
                del backend
    except TimeoutError as exc:
        budget_stop = {"type": type(exc).__name__, "message": str(exc),
            "elapsed_seconds": time.monotonic() - started,
            "policy": "stop entire roster; retain durable candidates; no replacement"}
        old.atomic_new(root / "budget-stop.json", budget_stop)
        log.emit("budget_exhausted", **budget_stop)
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        old.atomic_new(root / "failure.json", failure)
        log.emit("generation_failed", **failure)
    records = [old.read_json(path) for path in sorted(root.glob("question-*/candidate-*.json"))]
    # 不把已耗尽的 GPU 时钟传给评分，否则全部端点会无故成为 unknown。
    grading_start = time.monotonic()
    def grading_deadline():
        if time.monotonic() - grading_start >= GRADING_SECONDS:
            raise TimeoutError("Independent CPU grading budget exhausted; remaining outcomes stay unknown")
    log.emit("cpu_grading_started", saved_candidates=len(records), independent_seconds=GRADING_SECONDS)
    try:
        records = old.grade_saved(root, rows, records, deadline=grading_deadline)
    except BaseException as exc:
        grading_failure = {"type": type(exc).__name__, "message": str(exc)}
        old.atomic_new(root / "grading-failure.json", grading_failure)
    try:
        selected_after, identity_after = select_questions(args.data_dir)
        integrity = (source_hashes() == plan["code_sha256"]
            and identity_after == plan["data_identity"]
            and [row["id"] for row in selected_after] == plan["question_ids"]
            and old.sha256(args.frozen_selector) == plan["frozen_selector"]["sha256"])
    except Exception as exc:
        integrity = False
        old.atomic_new(root / "integrity-failure.json", {"type": type(exc).__name__, "message": str(exc)})
    acceptance = (root / "gpu-acceptance.json").exists() and old.read_json(root / "gpu-acceptance.json")["passed"]
    analysis = {"schema_version": "entropy-value-v1", "question_ids": plan["question_ids"], "records": records}
    old.atomic_new(root / "analysis-input.json", analysis)
    status = "failed" if failure or grading_failure or not integrity else (
        "budget_exhausted" if budget_stop else "completed")
    if not acceptance and status == "completed":
        status = "no_gpu_acceptance"
    summary = {"protocol": PROTOCOL, "status": status,
        "planned_questions": len(rows), "completed_questions": len(summaries),
        "max_planned_candidates": len(rows) * old.CANDIDATE_CAP, "saved_candidates": len(records),
        "question_summaries": summaries, "failure": failure, "budget_stop": budget_stop,
        "grading_failure": grading_failure, "source_data_selector_integrity": integrity,
        "gpu_acceptance_passed": acceptance, "elapsed_seconds": time.monotonic() - started,
        "analysis_input_sha256": old.sha256(root / "analysis-input.json")}
    old.atomic_new(root / "summary.json", summary)
    log.emit("exploration_finished", status=status, saved_candidates=len(records),
        completed_questions=len(summaries), elapsed_seconds=round(summary["elapsed_seconds"], 1))
    log.close()
    return 0 if status in {"completed", "budget_exhausted"} and acceptance else 1


if __name__ == "__main__":
    raise SystemExit(main())
