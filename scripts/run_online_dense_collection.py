#!/usr/bin/env python3
"""Collect all reasoning-candidate probes for the frozen a08 ten questions.

Stopping is disabled; each trajectory ends naturally or at the fixed 32768
main-token cap. Collection costs are separate from online-method comparisons.
The immutable a08 evidence and every earlier acceptance gate remain mandatory.
No sample, seed, configuration, budget, timeout, or retry overrides are exposed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_online_development as development
from online_contract import MODEL_REVISION
from run_online_diagnostic import (
    EventLog, answer_record, assert_gpu_idle, atomic_new, configuration_record,
    configurations, derive_seed, gpu_inventory, gpu_lock, sha256,
)
from validate_online_capacity import check_source_hashes, read_json
from diagnose_probe_parity import verify_environment
from validate_online_gpu import first_difference, fresh_run

SCOPE = "a08_paired_ten_question_dense_trajectory_collection_not_online_method_or_primary_benchmark"
LABEL = "dense-collect-no-stop"
A08_MANIFEST_SHA256 = "102780d05e8b5adcf91c15468eb0828a898b72fb36e65e7b06f279b5b857ddf1"
A08_SUMMARY_SHA256 = "570fb6bc2c6a7e6367165fd05f32ee0c9236d96124859945d23074d2dfef1bea"
SOURCE_FILES = tuple(dict.fromkeys(("scripts/run_online_dense_collection.py", *development.SOURCE_FILES)))


def validate_development(path, expected):
    """Verify immutable a08 provenance and its complete 40-request hash chain."""
    root = Path(path).expanduser().resolve()
    if (sha256(root / "manifest.json") != A08_MANIFEST_SHA256
            or sha256(root / "summary.json") != A08_SUMMARY_SHA256):
        raise ValueError("Require the frozen completed a08 manifest and summary")
    manifest, summary = (read_json(root / name) for name in ("manifest.json", "summary.json"))
    before, after = (read_json(root / name) for name in ("integrity-before.json", "integrity-after.json"))
    if (summary.get("status") != "completed" or summary.get("planned_count") != 40
            or summary.get("executed_count") != 40 or summary.get("completed_count") != 40
            or summary.get("failed_count") != 0 or summary.get("unexecuted_count") != 0
            or summary.get("failure") is not None or summary.get("unexecuted_requests") != []
            or summary.get("source_input_identity_integrity") != "unchanged"
            or not before or before != after or before != manifest["source_input_sha256"]
            or (root / "failure.json").exists() or (root / "integrity-failure.json").exists()):
        raise ValueError("a08 completion/source-input integrity is inconsistent")
    development.long_run.require_integrity(before, "a08 historical source/input")
    if set(manifest["code_sha256"]) != set(development.SOURCE_FILES):
        raise ValueError("a08 source manifest is incomplete")
    check_source_hashes(manifest["code_sha256"], development.SOURCE_FILES)
    for key in ("scope", "runner_protocol", "model_id", "model_revision", "dtype", "attention_implementation",
                "question_count", "planned_count", "master_seed", "rollout_id", "max_new_tokens",
                "max_final_tokens", "max_probe_tokens", "questions", "requests", "data_identity"):
        if manifest.get(key) != expected[key]:
            raise ValueError(f"a08 differs from the current frozen development plan: {key}")
    for field, key in (("prerequisite_gate", "sha256"), ("capacity_evidence", "summary_sha256"),
                       ("long_context_evidence", "manifest_sha256"), ("long_context_evidence", "request_sha256")):
        if manifest[field][key] != expected[field][key]:
            raise ValueError("a08 is not bound to the required earlier acceptance chain")
    environment = read_json(root / "environment.json")["backend_metadata"]
    verify_environment(expected["prerequisite_gate"]["environment"], environment)
    answers = [json.loads(line) for line in (root / "final_answers.jsonl").read_text(encoding="utf-8").splitlines()]
    if (sha256(root / "final_answers.jsonl") != summary["final_answers_sha256"]
            or sha256(root / "answers.jsonl") != summary["journal_sha256"]
            or (root / "answers.jsonl").read_bytes() != (root / "final_answers.jsonl").read_bytes()
            or len(answers) != 40 or len(summary["requests"]) != 40):
        raise ValueError("a08 final-answer journal/hash/count mismatch")
    paired_sources = {}
    for job, saved, answer in zip(manifest["requests"], summary["requests"], answers):
        if any(saved.get(key) != value for key, value in job.items()) or saved.get("status") != "completed":
            raise ValueError("a08 executed request does not match its fixed plan")
        directory = (root / job["relative_directory"]).resolve()
        if not directory.is_relative_to(root):
            raise ValueError("a08 request path escapes its root")
        source_path, answer_path = directory / "request.json", directory / "final-answer.json"
        source_hash = sha256(source_path)
        if source_hash != saved["source_sha256"] or sha256(answer_path) != saved["final_answer_sha256"]:
            raise ValueError("a08 request/final-answer checksum mismatch")
        record = read_json(source_path)
        if (record.get("status") != "completed" or record.get("sample_id") != job["sample_id"]
                or record.get("development_id") != job["development_id"]
                or record.get("configuration") != job["configuration"]
                or record.get("request_configuration") != job["request_configuration"]
                or record.get("config_hash") != job["request_configuration"]["config_hash"]
                or record.get("seed_reason") != job["request_configuration"]["seed_reason"]
                or record.get("rollout_id") != development.ROLLOUT_ID
                or record.get("max_new_tokens") != development.MAIN_CAP):
            raise ValueError("a08 request identity/configuration mismatch")
        verify_environment(environment, record["backend_metadata"])
        if (read_json(answer_path) != answer or answer.get("source_sha256") != source_hash
                or answer.get("execution_status") != "completed"
                or answer.get("sample_id") != job["sample_id"]
                or answer.get("source_file") != str(source_path.relative_to(root))):
            raise ValueError("a08 answer does not bind the actual saved request")
        if job["configuration"] in ("vanilla", "codestop-dense"):
            paired_sources.setdefault(job["development_id"], {})[job["configuration"]] = {
                "path": str(source_path), "sha256": source_hash}
    inputs = {**before, **{str(path.resolve()): sha256(path) for path in root.rglob("*")
                         if path.is_file() and path.suffix in (".json", ".jsonl")}}
    return {"path": str(root), "manifest_sha256": A08_MANIFEST_SHA256,
            "summary_sha256": A08_SUMMARY_SHA256, "planned_count": 40,
            "paired_sources": paired_sources, "input_sha256": inputs}


def prepare(args):
    rows, _, previous_plan = development.prepare(args)
    previous = validate_development(args.development_run, previous_plan)
    config = configurations([LABEL], 4)[0]
    requests = []
    for number, (row, question) in enumerate(zip(rows, previous_plan["questions"])):
        seed = derive_seed(development.MASTER_SEED, row["id"], development.ROLLOUT_ID, "reason")
        declared = configuration_record(config, seed_reason=seed, seed_schedule=None,
                                       max_new_tokens=development.MAIN_CAP, attention_implementation="eager")
        requests.append({"request_index": number, "development_id": question["development_id"],
            "sample_id": row["id"], "configuration": LABEL, "request_configuration": declared,
            "relative_directory": f"requests/{question['development_id']}/{LABEL}"})
    code = {name: sha256(ROOT / name) for name in SOURCE_FILES}
    inputs = {**previous_plan["source_input_sha256"], **previous["input_sha256"],
              **{str(ROOT / name): digest for name, digest in code.items()}}
    development.long_run.require_integrity(inputs, "Dense collection preflight source/input")
    # Keep the original question records, including a08's configuration_order,
    # as an exact provenance assertion. Only requests describes the new run order.
    manifest = {**previous_plan, "scope": SCOPE, "planned_count": 10, "requests": requests,
        "method": "dense_collect_no_stop", "stopping_enabled": False,
        "order_policy": "one dense_collect_no_stop request per original a08 question in its frozen order",
        "question_metadata_policy": "questions is exactly the a08 question list; configuration_order records that prior run",
        "development_evidence": previous, "code_sha256": code, "source_input_sha256": inputs,
        "collection_contract": {"probe_every_reasoning_candidate": True, "apply_early_stop": False,
            "retain_would_stop_decisions": True, "terminal_policy": "natural_eos_or_32768_main_token_budget",
            "incomplete_or_invalid_probes_are_retained": True,
            "full_trajectory_meaning": "all candidates until natural termination or the explicit cap; never beyond that cap"},
        "limits": {**previous_plan["limits"], "total_scope": "entire invocation including preparation, load, warmup, requests and intervening IO"},
        "cost_accounting": "all collection cost is diagnostic acquisition cost, excluded from online-method and primary result tables",
        "eligible_for_primary_speed_comparison": False,
        "grading": "separate strict grading with planned-count 10; diagnostic final answers only, never an online-method accuracy row"}
    return rows, config, manifest


def invoke_collection(backend, log, budget, row, job, config):
    """Reuse deadline/partial handling, then enforce the no-stop collection contract."""
    record, error = development.invoke(backend, log, budget, row, job, config)
    record.update(scope=SCOPE, eligible_for_primary_speed_comparison=False,
                  collection_cost_only=True, stopping_enabled=False)
    if error is None:
        try:
            candidates, probes = record["candidates"], record["probes"]
            if (record.get("method") != "dense_collect_no_stop"
                    or record.get("stop_reason") not in ("natural_eos", "budget")
                    or record.get("n_candidates") != len(candidates)
                    or record.get("n_probes") != len(probes) or len(probes) != len(candidates)
                    or any(candidate.get("queried") is not True for candidate in candidates)
                    or any(probe.get("stop_applied") is not False or probe.get("should_stop") is not False for probe in probes)
                    or [(item["candidate_j"], item["token_position"]) for item in candidates]
                    != [(item["candidate_j"], item["token_position"]) for item in probes]):
                raise ValueError("Completed request violates the complete dense/no-stop collection contract")
            record["collection_complete_to_termination_or_cap"] = True
        except Exception as exc:
            error = exc
            record.update(source_execution_status=record["status"], status="failed",
                          error_type=type(exc).__name__, error=str(exc))
    if error is not None:
        record["collection_complete_to_termination_or_cap"] = False
    return record, error


def save_request(directory, root, row, job, record, error, backend):
    source_path = directory / "request.json"
    atomic_new(source_path, record)
    try:
        answer = answer_record(record, source_path, row["answer"], backend, LABEL)
    except BaseException as export_error:
        answer = answer_record({**record, "status": "failed", "answer_token_ids": []},
                               source_path, row["answer"], backend, LABEL)
        answer.update(source_execution_status=record["status"], export_error=str(export_error))
        error = error or export_error
    answer.update(development_id=job["development_id"], scope=SCOPE, collection_cost_only=True,
                  eligible_for_primary_speed_comparison=False, source_file=str(source_path.relative_to(root)))
    atomic_new(directory / "final-answer.json", answer)
    completed = {**job, "status": answer["execution_status"], "source_execution_status": record["status"],
                 "source_sha256": sha256(source_path), "final_answer_sha256": sha256(directory / "final-answer.json")}
    return answer, completed, error


def run_collection(args, *, backend_factory=None, now=time.monotonic):
    budget = development.WallBudget(now)
    rows, config, manifest = prepare(args)
    if args.inspect_only:
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.preflight_tokenizer_only:
        report = development.tokenizer_preflight(args, rows, manifest)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0 if report["all_within_model_context"] else 2
    if args.model_dir is None or args.run_root is None:
        raise ValueError("Execution requires --model-dir and a new --run-root")
    root, model_dir = args.run_root.expanduser().resolve(), args.model_dir.expanduser().resolve()
    protected = (model_dir, args.data_dir, args.capacity_root, args.long_context_root,
                 args.gate_summary.parent, args.development_run, *args.natural_run_root)
    if (root.exists() or args.run_root.is_symlink()
            or any(root.is_relative_to(Path(path).resolve()) for path in protected)):
        raise ValueError("Run root must be new and outside all prerequisite evidence/model/data")
    if not model_dir.is_dir() or model_dir.name != MODEL_REVISION:
        raise ValueError("Require the existing pinned model snapshot")
    with fresh_run(root):
        atomic_new(root / "manifest.json", manifest)
        atomic_new(root / "integrity-before.json", manifest["source_input_sha256"])
        log = EventLog(root / "events.jsonl")
        answers, completed, failure, phase = [], [], None, "gpu_preflight"
        with (root / "answers.jsonl").open("xb") as journal:
            try:
                development.long_run.require_integrity(manifest["source_input_sha256"], "Before GPU startup source/input")
                budget.check()
                gpu = gpu_inventory()
                with gpu_lock(gpu["uuid"]) as lock:
                    idle = assert_gpu_idle(gpu["uuid"])
                    atomic_new(root / "gpu-preflight.json", {"gpu": gpu, "lock": lock, "idle": idle})
                    phase, started = "model_load", time.perf_counter()
                    log.emit("model_load_start", model_revision=MODEL_REVISION)
                    if backend_factory is None:
                        from torch_online_backend import TorchOnlineBackend
                        backend_factory = TorchOnlineBackend
                    backend = backend_factory(model_dir, attention_implementation="eager")
                    atomic_new(root / "environment.json", {"backend_metadata": backend.metadata, "gpu": gpu,
                                                          "model_load_seconds_external": time.perf_counter() - started})
                    verify_environment(manifest["prerequisite_gate"]["environment"], backend.metadata)
                    if (backend.metadata.get("validation_model") is not False or backend.metadata.get("device") != "cuda:0"
                            or backend.metadata.get("gpu_total_memory_bytes") != manifest["capacity_evidence"]["gpu_total_memory_bytes"]):
                        raise ValueError("Actual backend differs from the validated real CUDA model")
                    budget.check()
                    phase = "all_question_prompt_preflight"
                    preflight = development.prompt_preflight(rows, manifest,
                        development.DeadlineBackend(backend, budget).prompt_ids, backend.context_limit,
                        backend.markers.final_prefix, backend.markers.trial_prefix)
                    atomic_new(root / "prompt-preflight.json", preflight)
                    if not preflight["all_within_model_context"]:
                        raise ValueError("A fixed question exceeds model context; no questions were skipped")
                    phase = "warmup"
                    budget.begin_request()
                    warmup, error = development.long_run.invoke(development.DeadlineBackend(backend, budget), log,
                        question="Compute 1 + 1.", sample_id="synthetic-warmup-only", label="warmup",
                        seed=derive_seed(development.MASTER_SEED, "a08-dense-collection", development.ROLLOUT_ID, "warmup"), budget=16)
                    atomic_new(root / "warmup.json", {**warmup, "scope": "excluded_warmup"})
                    if error:
                        raise error
                    budget.check()
                    budget.end_request()
                    log.emit("warmup_completed", elapsed_ms=warmup["time_total_ms"], excluded_from_results=True)
                    by_id = {row["id"]: row for row in rows}
                    for job in manifest["requests"]:
                        phase = "between_requests"
                        development.long_run.require_integrity(manifest["source_input_sha256"], "Before request source/input")
                        budget.begin_request()
                        directory = root / job["relative_directory"]
                        directory.mkdir(parents=True, exist_ok=False)
                        atomic_new(directory / "started.json", job)
                        phase = job["relative_directory"]
                        log.emit("request_start", request_index=job["request_index"], development_id=job["development_id"],
                                 configuration=LABEL, seed_reason=job["request_configuration"]["seed_reason"])
                        row = by_id[job["sample_id"]]
                        record, error = invoke_collection(backend, log, budget, row, job, config)
                        budget.end_request()
                        answer, saved, error = save_request(directory, root, row, job, record, error, backend)
                        answers.append(answer)
                        completed.append(saved)
                        journal.write((json.dumps(answer, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
                        journal.flush()
                        os.fsync(journal.fileno())
                        log.emit("request_saved", request_index=job["request_index"], status=answer["execution_status"],
                                 main_tokens=record.get("main_generated_tokens"), probes=record.get("n_probes"),
                                 elapsed_ms=record.get("time_total_ms"), stop_reason=record.get("stop_reason"))
                        if error:
                            raise error
                    budget.check()
            except BaseException as error:
                failure = {"phase": phase, "error_type": type(error).__name__, "message": str(error),
                           "traceback": traceback.format_exc(), "retry_performed": False}
                atomic_new(root / "failure.json", failure)
                log.emit("run_failed", phase=phase, error_type=type(error).__name__, message=str(error))
            finally:
                after = development.long_run.current_hashes(manifest["source_input_sha256"])
                atomic_new(root / "integrity-after.json", after)
                difference = first_difference(manifest["source_input_sha256"], after)
                if difference:
                    integrity_error = {"phase": "final_integrity", "error_type": "SourceInputChanged", "message": str(difference)}
                    atomic_new(root / "integrity-failure.json", integrity_error)
                    failure = failure or integrity_error
                atomic_new(root / "final_answers.jsonl", "".join(
                    json.dumps(answer, ensure_ascii=False, allow_nan=False) + "\n" for answer in answers).encode("utf-8"), raw=True)
                failed = sum(item["status"] != "completed" for item in completed)
                summary = {"schema_version": 1, "scope": SCOPE, "status": "failed" if failure else "completed",
                    "planned_count": 10, "executed_count": len(completed), "failed_count": failed,
                    "completed_count": len(completed) - failed, "unexecuted_count": 10 - len(completed),
                    "requests": completed, "unexecuted_requests": manifest["requests"][len(completed):],
                    "failure": failure, "warmup_excluded": True, "grading_status": "not_run",
                    "source_input_identity_integrity": "changed" if difference else "unchanged",
                    "eligible_for_primary_speed_comparison": False, "collection_cost_only": True,
                    "elapsed_wall_seconds_including_load_warmup_and_IO": now() - budget.started,
                    "final_answers_sha256": sha256(root / "final_answers.jsonl"),
                    "journal_sha256": sha256(root / "answers.jsonl")}
                atomic_new(root / "summary.json", summary)
                log.emit("run_finished", status=summary["status"], executed_count=len(completed),
                         failed_count=failed, unexecuted_count=10 - len(completed))
                log.close()
        return 1 if failure else 0


def parser():
    result = development.parser()
    result.description = __doc__
    result.add_argument("--development-run", type=Path, required=True, help="the immutable completed a08 development10-001 directory")
    return result


def main(argv=None):
    try:
        return run_collection(parser().parse_args(argv))
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(f"Dense collection refused: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
