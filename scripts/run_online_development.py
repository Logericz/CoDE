#!/usr/bin/env python3
"""Fixed ten-question, four-configuration development cost diagnostic.

No question, method, seed, budget or retry override is exposed. Inspect-only is
file verification only. Tokenizer-only preflight runs without Torch or GPU.
Timeouts are checked between backend operations; a CUDA kernel cannot be
preempted. Per-request JSON files remain the authoritative recovery evidence.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_online_long_context as long_run
from online_contract import FINAL_PREFIX, MODEL_ID, MODEL_REVISION, RUNNER_PROTOCOL, TASK_SUFFIX, TRIAL_PREFIX
from online_engine import FINAL_TOKEN_CAP, RequestError, run_request
from online_protocol import MAX_PROBE_TOKENS
from run_online_diagnostic import (
    EventLog, ProgressBackend, answer_record, assert_gpu_idle, atomic_new,
    configuration_record, configurations, derive_seed, encoded, gpu_inventory,
    gpu_lock, sha256, verify_data,
)
from validate_online_capacity import check_source_hashes, read_json
from diagnose_probe_parity import verify_environment
from validate_online_gpu import first_difference, fresh_run

SCOPE = "fixed_pilot20_ten_question_development_cost_diagnostic_not_primary_benchmark"
EXCLUDED_IDS = ("math/train/algebra/566", "math/train/geometry/428")
LABELS = ("vanilla", "codestop-dense", "codestop-fixed", "codestop-adaptive")
MAIN_CAP, MASTER_SEED, ROLLOUT_ID = 32768, 42, 0
TOTAL_SECONDS, REQUEST_SECONDS = 7200, 1800
A07_MANIFEST_SHA256 = "61b3f2d4c022ad129b9284f7a1614bbffbcc7848982ea79921dec160ffe224d2"
A07_REQUEST_SHA256 = "fed49a30ede55b57ca157657bb28319e4dab1bc08ea93c4bf8ee5456b3658549"
SOURCE_FILES = ("scripts/run_online_development.py", *long_run.SOURCE_FILES)


def validate_long_run(path, prerequisites):
    root = Path(path).expanduser().resolve()
    if (sha256(root / "manifest.json") != A07_MANIFEST_SHA256
            or sha256(root / "vanilla.json") != A07_REQUEST_SHA256):
        raise ValueError("Require the frozen completed a07 manifest and q002 request")
    manifest, summary, request = (read_json(root / name) for name in ("manifest.json", "summary.json", "vanilla.json"))
    before, after = (read_json(root / name) for name in ("integrity-before.json", "integrity-after.json"))
    if (summary.get("status") != "completed" or summary.get("executed_count") != 1
            or summary.get("unexecuted_count") != 0 or summary.get("planned_count") != 1
            or summary.get("source_input_identity_integrity") != "unchanged"
            or before != after or before != manifest["source_input_sha256"]
            or summary.get("requests", [{}])[0].get("source_sha256") != A07_REQUEST_SHA256):
        raise ValueError("a07 completion/source-input integrity evidence is inconsistent")
    long_run.require_integrity(before, "a07 historical source/input")
    check_source_hashes(manifest["code_sha256"], long_run.CRITICAL)
    if (manifest.get("scope") != long_run.SCOPE or manifest["data_identity"] != prerequisites["data_identity"]
            or manifest["prerequisite_gate"]["sha256"] != prerequisites["prerequisite_gate"]["sha256"]
            or manifest["capacity_evidence"]["summary_sha256"] != prerequisites["capacity_evidence"]["summary_sha256"]):
        raise ValueError("a07 is not bound to this frozen data and prerequisite chain")
    metadata = request["backend_metadata"]
    verify_environment(prerequisites["prerequisite_gate"]["environment"], metadata)
    if (request.get("status") != "completed" or request.get("sample_id") != EXCLUDED_IDS[1]
            or request.get("method") != "vanilla" or request.get("max_new_tokens") != MAIN_CAP
            or request.get("seed_reason") != derive_seed(MASTER_SEED, EXCLUDED_IDS[1], ROLLOUT_ID, "reason")
            or request.get("stop_reason") != "natural_eos" or request.get("reason_tokens", 0) < 8192
            or request.get("injected_prompt_tokens") != 0 or request.get("finalization_tokens") != 0
            or request.get("answer_boundary", {}).get("source") != "natural_first_end_think"
            or request["output_token_ids"][-1] not in metadata["markers"]["eos_ids"]
            or [item["token_id"] for item in request["main_samples"]] != request["output_token_ids"]):
        raise ValueError("a07 must contain the actual long q002 natural trajectory")
    inputs = {**before, **{str(p.resolve()): sha256(p) for p in root.glob("*.json")},
              **{str(p.resolve()): sha256(p) for p in root.glob("*.jsonl")}}
    return {"path": str(root), "manifest_sha256": A07_MANIFEST_SHA256,
            "request_sha256": A07_REQUEST_SHA256, "reason_tokens": request["reason_tokens"],
            "main_generated_tokens": request["main_generated_tokens"], "input_sha256": inputs}


def prepare(args):
    # 复用既有三门槛；这里不改旧 q002 工具，也不把新题伪装成旧 prompt。
    _, prerequisites = long_run.prepare(SimpleNamespace(**vars(args), max_new_tokens=MAIN_CAP))
    previous = validate_long_run(args.long_context_root, prerequisites)
    pilot = [json.loads(line) for line in (args.data_dir / "pilot20.jsonl").read_text(encoding="utf-8").splitlines()]
    if not set(EXCLUDED_IDS) <= {row["id"] for row in pilot}:
        raise ValueError("Both explicitly excluded historical IDs must exist in frozen pilot20")
    selected = [(index, row) for index, row in enumerate(pilot) if row["id"] not in EXCLUDED_IDS][:10]
    if len(selected) != 10:
        raise ValueError("Frozen pilot20 does not contain ten eligible questions")
    questions, plan = [], []
    configs = {item["label"]: item for item in configurations(LABELS, 4)}
    for number, (source_index, row) in enumerate(selected):
        checked, _ = verify_data(args.data_dir, args.data_manifest_sha256, row["id"])
        if checked != row:
            raise ValueError("Selected row differs from the frozen data verifier")
        label = f"dev{number + 1:02d}"
        order = LABELS[number % 4:] + LABELS[:number % 4]
        questions.append({"development_id": label, "source_row_index": source_index,
            "source_row_index_base": 0, "original_id": row["id"], "problem_sha256": row["problem_sha256"],
            "row_sha256": hashlib.sha256(encoded(row)).hexdigest(), "configuration_order": list(order)})
        seed = derive_seed(MASTER_SEED, row["id"], ROLLOUT_ID, "reason")
        for config_label in order:
            config = configuration_record(configs[config_label], seed_reason=seed, seed_schedule=None,
                                          max_new_tokens=MAIN_CAP, attention_implementation="eager")
            plan.append({"request_index": len(plan), "development_id": label, "sample_id": row["id"],
                         "configuration": config_label, "request_configuration": config,
                         "relative_directory": f"requests/{label}/{config_label}"})
    code = {name: sha256(ROOT / name) for name in SOURCE_FILES}
    inputs = {**prerequisites["source_input_sha256"], **previous["input_sha256"],
              **{str(ROOT / name): digest for name, digest in code.items()}}
    long_run.require_integrity(inputs, "Development preflight source/input")
    manifest = {"schema_version": 1, "scope": SCOPE, "runner_protocol": RUNNER_PROTOCOL,
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "dtype": "BF16", "attention_implementation": "eager",
        "question_count": 10, "planned_count": 40, "master_seed": MASTER_SEED, "rollout_id": ROLLOUT_ID,
        "max_new_tokens": MAIN_CAP, "max_final_tokens": FINAL_TOKEN_CAP, "max_probe_tokens": MAX_PROBE_TOKENS,
        "selection_policy": "frozen pilot20 order, exclude the two exact historical IDs, take first ten; never skip or replace",
        "excluded_sample_ids": list(EXCLUDED_IDS), "questions": questions, "requests": plan,
        "order_policy": "question-major; rotate the four fixed configurations left by question index modulo four",
        "data_identity": prerequisites["data_identity"], "prerequisite_gate": prerequisites["prerequisite_gate"],
        "natural_branch_coverage": prerequisites["natural_branch_coverage"],
        "capacity_evidence": prerequisites["capacity_evidence"], "long_context_evidence": previous,
        "capacity_scope": "previous synthetic capacity covered the q002 prompt only; longer new prompts are a real-run test, not previously measured capacity",
        "context_policy": "check every actual prompt+32768+max(final-prefix+30,trial-prefix+21,30) against actual model context; no question filtering",
        "code_sha256": code, "source_input_sha256": inputs, "eligible_for_primary_speed_comparison": False,
        "limits": {"total_wall_seconds": TOTAL_SECONDS, "per_request_wall_seconds": REQUEST_SECONDS,
                   "total_scope": "run startup, model load, warmup, requests and intervening IO",
                   "enforcement": "before backend operations and between requests; CUDA kernels cannot be preempted"},
        "warmup": {"main_tokens": 16, "max_final_tokens": 30, "method": "vanilla",
                   "includes_probe_warmup": False, "included_in_measured_requests": False},
        "logging": {"main_progress_every": 64, "synchronous_logging_and_deadline_overhead_in_request_timing": True},
        "grading": "separate strict grading of all executed rows, including failures; never used to select questions or configurations",
        "answer_durability": "atomic per-request final-answer.json is canonical; answers.jsonl append+fsync; final_answers.jsonl is an atomic final snapshot",
        "failure_policy": "stop entire matrix on first error, OOM or timeout; preserve unexecuted denominator; no retry or fallback"}
    return [row for _, row in selected], configs, manifest


def prompt_preflight(rows, manifest, prompt_ids, context_limit, final_prefix, trial_prefix):
    reserve = max(len(final_prefix) + FINAL_TOKEN_CAP, len(trial_prefix) + MAX_PROBE_TOKENS, FINAL_TOKEN_CAP)
    old_context = manifest["capacity_evidence"]["context_required"]
    entries = []
    for row, question in zip(rows, manifest["questions"]):
        ids = list(prompt_ids(row["problem"]))
        if not ids or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError("Tokenizer returned invalid prompt IDs")
        required = len(ids) + MAIN_CAP + reserve
        entries.append({"development_id": question["development_id"], "sample_id": row["id"],
            "prompt_token_ids": ids, "prompt_tokens": len(ids),
            "prompt_sha256": hashlib.sha256(encoded(ids)).hexdigest(),
            "context_required": required, "within_model_context": required <= context_limit,
            "exceeds_prior_synthetic_context": required > old_context,
            "extra_context_positions_not_previously_capacity_tested": max(0, required - old_context)})
    return {"schema_version": 1, "scope": "tokenization_and_context_check_only_no_generation",
        "context_limit": context_limit, "reserved_tokens": reserve, "previous_synthetic_context": old_context,
        "all_within_model_context": all(row["within_model_context"] for row in entries),
        "max_prompt_tokens": max(row["prompt_tokens"] for row in entries),
        "max_context_required": max(row["context_required"] for row in entries), "questions": entries,
        "interpretation": "The old synthetic capacity result is not evidence for these new prompt contents or additional context positions."}


def tokenizer_preflight(args, rows, manifest):
    if args.model_dir is None:
        raise ValueError("Tokenizer-only preflight requires --model-dir")
    model_dir = args.model_dir.expanduser().resolve()
    expected = manifest["prerequisite_gate"]["environment"]
    if model_dir.name != MODEL_REVISION or not model_dir.is_dir():
        raise ValueError("Require the existing pinned local tokenizer snapshot")
    hashes = {**expected["snapshot_config_sha256"], "tokenizer.json": expected["tokenizer_sha256"]}
    if any(sha256(model_dir / name) != digest for name, digest in hashes.items()):
        raise ValueError("Local tokenizer/config differs from the validated snapshot")
    if "torch" in sys.modules:
        raise RuntimeError("Tokenizer-only mode requires a fresh process before Torch is imported")
    # 仅此独立 CLI 模式禁用框架探测；正常 GPU 模式不走这里。
    os.environ["USE_TORCH"], os.environ["USE_TF"], os.environ["USE_FLAX"] = "0", "0", "0"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    final = tokenizer.encode(FINAL_PREFIX, add_special_tokens=False)
    trial = tokenizer.encode(TRIAL_PREFIX, add_special_tokens=False)
    if final != expected["markers"]["final_prefix"] or trial != expected["markers"]["trial_prefix"]:
        raise ValueError("Tokenizer prefix IDs differ from the validated protocol")
    capacity = read_json(model_dir / "config.json")["max_position_embeddings"]
    maximum = getattr(tokenizer, "model_max_length", None)
    if type(maximum) is int and 0 < maximum < 10**9:
        capacity = min(capacity, maximum)
    if capacity != expected["context_limit"]:
        raise ValueError("Tokenizer/model context differs from the validated environment")
    result = prompt_preflight(rows, manifest, lambda question: tokenizer.apply_chat_template(
        [{"role": "user", "content": question + TASK_SUFFIX}], tokenize=True,
        add_generation_prompt=True, enable_thinking=True), capacity, final, trial)
    if "torch" in sys.modules:
        raise RuntimeError("Tokenizer-only mode unexpectedly imported Torch")
    result["torch_imported"] = False
    return result


class BudgetExceeded(TimeoutError):
    pass


class WallBudget:
    def __init__(self, now=time.monotonic):
        self.now, self.started, self.request_started = now, now(), None

    def check(self):
        current = self.now()
        if current - self.started >= TOTAL_SECONDS:
            raise BudgetExceeded("total_wall_limit_7200_seconds")
        if self.request_started is not None and current - self.request_started >= REQUEST_SECONDS:
            raise BudgetExceeded("request_wall_limit_1800_seconds")

    def begin_request(self):
        self.check()
        self.request_started = self.now()

    def end_request(self):
        self.request_started = None


class DeadlineBackend:
    """检查放在操作前，避免已完成的 sample 因操作后超时而丢失记录。

    synchronize 不抛超时：engine 会先把刚返回的 token/KV 状态记入 partial，
    下次计算/解码前才拒绝继续。单次正在运行的 kernel 无法被这里中断。
    """
    OPERATIONS = {"prompt_ids", "start_request", "prefill", "sample", "greedy", "extend", "probe", "decode"}

    def __init__(self, backend, budget):
        self.backend, self.budget = backend, budget

    def __getattr__(self, name):
        value = getattr(self.backend, name)
        if name not in self.OPERATIONS:
            return value
        def checked(*args, **kwargs):
            self.budget.check()
            return value(*args, **kwargs)
        return checked


def invoke(backend, log, budget, row, job, config):
    result, error, started = None, None, time.perf_counter()
    try:
        result = run_request(ProgressBackend(DeadlineBackend(backend, budget), log, job["development_id"] + "/" + job["configuration"]),
            question=row["problem"], sample_id=row["id"], rollout_id=ROLLOUT_ID, method=config["method"],
            seed_reason=job["request_configuration"]["seed_reason"], seed_schedule=None,
            max_new_tokens=MAIN_CAP, protocol_config=config["protocol_config"], schedule_config=config["schedule_config"])
        budget.check()
    except BaseException as exc:
        error = exc
        if isinstance(exc, RequestError):
            result = exc.partial
        elif result is not None:
            result = {**result, "source_execution_status": result["status"], "status": "failed",
                      "error_type": type(exc).__name__, "error": str(exc)}
        else:
            result = {"status": "failed", "partial_evidence_available": False,
                      "error_type": type(exc).__name__, "error": str(exc)}
    result.update(sample_id=row["id"], rollout_id=ROLLOUT_ID, method=config["method"],
        development_id=job["development_id"], configuration=job["configuration"], scope=SCOPE,
        max_new_tokens=MAIN_CAP, seed_reason=job["request_configuration"]["seed_reason"], seed_schedule=None,
        protocol_config=asdict(config["protocol_config"]), schedule_config=asdict(config["schedule_config"]),
        request_configuration=job["request_configuration"], config_hash=job["request_configuration"]["config_hash"],
        eligible_for_primary_speed_comparison=False, backend_metadata=backend.metadata,
        external_invocation_elapsed_ms=(time.perf_counter() - started) * 1000)
    return result, error


def run_development(args, *, backend_factory=None, now=time.monotonic):
    rows, configs, manifest = prepare(args)
    if args.inspect_only:
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.preflight_tokenizer_only:
        result = tokenizer_preflight(args, rows, manifest)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0 if result["all_within_model_context"] else 2
    if args.model_dir is None or args.run_root is None:
        raise ValueError("Execution requires --model-dir and a new --run-root")
    run_root, model_dir = args.run_root.expanduser().resolve(), args.model_dir.expanduser().resolve()
    protected = (model_dir, args.data_dir, args.capacity_root, args.long_context_root,
                 args.gate_summary.parent, *args.natural_run_root)
    if (run_root.exists() or args.run_root.is_symlink()
            or any(run_root.is_relative_to(Path(path).resolve()) for path in protected)):
        raise ValueError("Run root must be new and outside all prerequisite evidence/model/data")
    if not model_dir.is_dir() or model_dir.name != MODEL_REVISION:
        raise ValueError("Require the existing pinned model snapshot")
    budget = WallBudget(now)
    with fresh_run(run_root):
        atomic_new(run_root / "manifest.json", manifest)
        atomic_new(run_root / "integrity-before.json", manifest["source_input_sha256"])
        log = EventLog(run_root / "events.jsonl")
        completed, answers, failure, backend, phase = [], [], None, None, "gpu_preflight"
        with (run_root / "answers.jsonl").open("xb") as journal:
            try:
                long_run.require_integrity(manifest["source_input_sha256"], "Before GPU startup source/input")
                budget.check()
                gpu = gpu_inventory()
                with gpu_lock(gpu["uuid"]) as lock:
                    idle = assert_gpu_idle(gpu["uuid"])
                    atomic_new(run_root / "gpu-preflight.json", {"gpu": gpu, "lock": lock, "idle": idle})
                    phase, started = "model_load", time.perf_counter()
                    log.emit("model_load_start", model_revision=MODEL_REVISION)
                    if backend_factory is None:
                        from torch_online_backend import TorchOnlineBackend
                        backend_factory = TorchOnlineBackend
                    backend = backend_factory(model_dir, attention_implementation="eager")
                    atomic_new(run_root / "environment.json", {"backend_metadata": backend.metadata,
                        "gpu": gpu, "model_load_seconds_external": time.perf_counter() - started})
                    verify_environment(manifest["prerequisite_gate"]["environment"], backend.metadata)
                    if (backend.metadata.get("validation_model") is not False or backend.metadata.get("device") != "cuda:0"
                            or backend.metadata.get("gpu_total_memory_bytes") != manifest["capacity_evidence"]["gpu_total_memory_bytes"]):
                        raise ValueError("Actual backend differs from the validated real CUDA model")
                    budget.check()
                    phase = "all_question_prompt_preflight"
                    preflight = prompt_preflight(rows, manifest, DeadlineBackend(backend, budget).prompt_ids,
                        backend.context_limit, backend.markers.final_prefix, backend.markers.trial_prefix)
                    atomic_new(run_root / "prompt-preflight.json", preflight)
                    if not preflight["all_within_model_context"]:
                        raise ValueError("At least one fixed question exceeds model context; no questions were skipped")
                    phase = "warmup"
                    budget.begin_request()
                    warmup, error = long_run.invoke(DeadlineBackend(backend, budget), log,
                        question="Compute 1 + 1.", sample_id="synthetic-warmup-only", label="warmup",
                        seed=derive_seed(MASTER_SEED, "fixed-ten-development", ROLLOUT_ID, "warmup"), budget=16)
                    atomic_new(run_root / "warmup.json", {**warmup, "scope": "excluded_warmup"})
                    if error:
                        raise error
                    budget.check()
                    budget.end_request()
                    log.emit("warmup_completed", elapsed_ms=warmup["time_total_ms"], excluded_from_results=True)
                    long_run.require_integrity(manifest["source_input_sha256"], "Before matrix source/input")
                    by_id = {row["id"]: row for row in rows}
                    for job in manifest["requests"]:
                        phase = "between_requests"
                        budget.begin_request()
                        directory = run_root / job["relative_directory"]
                        directory.mkdir(parents=True, exist_ok=False)
                        atomic_new(directory / "started.json", job)
                        phase = job["relative_directory"]
                        log.emit("request_start", request_index=job["request_index"], development_id=job["development_id"],
                            configuration=job["configuration"], seed_reason=job["request_configuration"]["seed_reason"])
                        row = by_id[job["sample_id"]]
                        result, error = invoke(backend, log, budget, row, job, configs[job["configuration"]])
                        budget.end_request()
                        source_path = directory / "request.json"
                        atomic_new(source_path, result)
                        try:
                            answer = answer_record(result, source_path, row["answer"], backend, job["configuration"])
                        except BaseException as export_error:
                            answer = answer_record({**result, "status": "failed", "answer_token_ids": []},
                                source_path, row["answer"], backend, job["configuration"])
                            answer.update(source_execution_status=result["status"], export_error=str(export_error))
                            error = error or export_error
                        answer.update(development_id=job["development_id"], source_file=str(source_path.relative_to(run_root)))
                        atomic_new(directory / "final-answer.json", answer)
                        answers.append(answer)
                        completed.append({**job, "status": answer["execution_status"],
                            "source_execution_status": result["status"], "source_sha256": sha256(source_path),
                            "final_answer_sha256": sha256(directory / "final-answer.json")})
                        # 先有独立原子文件，再追加并 fsync；中途宕机也可据每请求文件恢复。
                        journal.write((json.dumps(answer, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
                        journal.flush()
                        os.fsync(journal.fileno())
                        log.emit("request_saved", request_index=job["request_index"], status=answer["execution_status"],
                            main_tokens=result.get("main_generated_tokens"), probes=result.get("n_probes"),
                            elapsed_ms=result.get("time_total_ms"), stop_reason=result.get("stop_reason"))
                        if error:
                            raise error
                    budget.check()
            except BaseException as error:
                failure = {"phase": phase, "error_type": type(error).__name__, "message": str(error),
                    "traceback": traceback.format_exc(), "retry_performed": False}
                atomic_new(run_root / "failure.json", failure)
                log.emit("run_failed", phase=phase, error_type=type(error).__name__, message=str(error))
            finally:
                after = long_run.current_hashes(manifest["source_input_sha256"])
                atomic_new(run_root / "integrity-after.json", after)
                difference = first_difference(manifest["source_input_sha256"], after)
                if difference:
                    integrity_error = {"phase": "final_integrity", "error_type": "SourceInputChanged", "message": str(difference)}
                    atomic_new(run_root / "integrity-failure.json", integrity_error)
                    failure = failure or integrity_error
                atomic_new(run_root / "final_answers.jsonl", "".join(
                    json.dumps(answer, ensure_ascii=False, allow_nan=False) + "\n" for answer in answers).encode("utf-8"), raw=True)
                failed = sum(item["status"] != "completed" for item in completed)
                summary = {"schema_version": 1, "scope": SCOPE, "status": "failed" if failure else "completed",
                    "planned_count": 40, "executed_count": len(completed), "failed_count": failed,
                    "completed_count": len(completed) - failed, "unexecuted_count": 40 - len(completed),
                    "requests": completed, "unexecuted_requests": manifest["requests"][len(completed):],
                    "failure": failure, "warmup_excluded": True, "grading_status": "not_run",
                    "source_input_identity_integrity": "changed" if difference else "unchanged",
                    "eligible_for_primary_speed_comparison": False,
                    "elapsed_wall_seconds_including_load_warmup_and_IO": now() - budget.started,
                    "final_answers_sha256": sha256(run_root / "final_answers.jsonl"),
                    "journal_sha256": sha256(run_root / "answers.jsonl")}
                atomic_new(run_root / "summary.json", summary)
                log.emit("run_finished", status=summary["status"], executed_count=len(completed),
                         failed_count=failed, unexecuted_count=40 - len(completed))
                log.close()
        return 1 if failure else 0


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-dir", type=Path, default=ROOT / "data/benchmarks")
    result.add_argument("--data-manifest-sha256", required=True)
    result.add_argument("--gate-summary", type=Path, required=True)
    result.add_argument("--capacity-root", type=Path, required=True)
    result.add_argument("--natural-run-root", type=Path, action="append", required=True)
    result.add_argument("--long-context-root", type=Path, required=True)
    result.add_argument("--model-dir", type=Path)
    result.add_argument("--run-root", type=Path)
    modes = result.add_mutually_exclusive_group()
    modes.add_argument("--inspect-only", action="store_true")
    modes.add_argument("--preflight-tokenizer-only", action="store_true")
    return result


def main(argv=None):
    try:
        return run_development(parser().parse_args(argv))
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(f"Development run refused: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
