#!/usr/bin/env python3
"""One frozen pilot question, one rollout, sequential common-online GPU requests.

This is a bounded engineering acceptance run, not a benchmark or speed claim.
Only existing local data/model files are used. Run directories are never resumed
or overwritten; warmup, measured requests, and grading inputs remain separate.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from online_contract import MODEL_ID, MODEL_REVISION, RUNNER_PROTOCOL
from online_engine import RequestError, json_safe, run_request
from online_protocol import ProtocolConfig, ScheduleConfig
from online_source_manifest import METHOD_IDENTITY_FILES

SCOPE = "single_question_development_gpu_acceptance_not_benchmark"
DEFAULT_SAMPLE = "math/train/geometry/428"  # Already exposed pilot20 q002.
COUNTS = {"calibration80": 80, "selection120": 120, "pilot20": 20,
          "math500": 500, "analysis100": 100}
DATA_SOURCES = {"EleutherAI/hendrycks_math", "HuggingFaceH4/MATH-500"}
DEFAULT_CONFIGS = ("vanilla", "codestop-dense", "codestop-fixed")
SUPPORTED_CONFIGS = DEFAULT_CONFIGS + (
    "deer-dense", "codestop-log", "codestop-random", "codestop-backoff",
    "codestop-adaptive", "dense-collect-no-stop")
CODE_FILES = ("scripts/run_online_diagnostic.py", "src/online_contract.py",
              "src/online_engine.py", "src/online_protocol.py", "src/torch_online_backend.py",
              "src/math_grading.py", "scripts/grade_math_answers.py", *METHOD_IDENTITY_FILES)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encoded(value):
    return (json.dumps(json_safe(value), ensure_ascii=False, sort_keys=True,
                       indent=2, allow_nan=False) + "\n").encode("utf-8")


def atomic_new(path, value, *, raw=False):
    """Publish a complete file with link(2), which never replaces an old path."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    created = False
    try:
        with temporary.open("xb") as stream:
            created = True
            stream.write(value if raw else encoded(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if created:
            temporary.unlink(missing_ok=True)


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)


def problem_hash(problem):
    text = " ".join(unicodedata.normalize("NFKC", problem).split())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_data(data_dir, expected_manifest_sha256, sample_id):
    """Independently check the frozen prepare_gpu_data schema, without imports/downloads."""
    data_dir = Path(data_dir).resolve()
    manifest_path = data_dir / "manifest.json"
    if not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256 or ""):
        raise ValueError("--data-manifest-sha256 requires a previously checked SHA-256")
    if sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("Frozen data manifest SHA-256 mismatch")
    manifest = read_json(manifest_path)
    files = manifest["files"]
    required = {f"{name}.jsonl" for name in COUNTS} | {"sources.lock.json", "answer_review.jsonl"}
    if manifest.get("schema_version") != 1 or not required <= set(files):
        raise ValueError("Incomplete or unsupported frozen data manifest")
    for name, info in files.items():
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("Unsafe filename in frozen manifest")
        path = data_dir / name
        if path.resolve().parent != data_dir:
            raise ValueError("Frozen file escapes data directory")
        if sha256(path) != info["sha256"]:
            raise ValueError(f"Frozen file checksum mismatch: {name}")
    lock = read_json(data_dir / "sources.lock.json")
    if lock != manifest.get("sources") or set(lock) != DATA_SOURCES or any(
            not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value)
            for value in lock.values()):
        raise ValueError("Dataset revision lock differs from frozen manifest")
    splits = {}
    for name, count in COUNTS.items():
        rows = [json.loads(line) for line in (data_dir / f"{name}.jsonl").read_text(
            encoding="utf-8").splitlines()]
        info = files[f"{name}.jsonl"]
        if len(rows) != count or len({row["id"] for row in rows}) != count or info["count"] != count:
            raise ValueError(f"Invalid frozen count or duplicate ID: {name}")
        if [row["id"] for row in rows] != info["ids"]:
            raise ValueError(f"Frozen ID order mismatch: {name}")
        if [row["problem_sha256"] for row in rows] != info["problem_sha256"]:
            raise ValueError(f"Frozen problem hash order mismatch: {name}")
        if any(problem_hash(row["problem"]) != row["problem_sha256"] for row in rows):
            raise ValueError(f"Frozen problem checksum mismatch: {name}")
        if any(row.get("source") not in lock for row in rows):
            raise ValueError(f"Unlocked source: {name}")
        splits[name] = rows
    for child, parent in (("pilot20", "selection120"), ("analysis100", "math500")):
        parent_rows = {row["id"]: row for row in splits[parent]}
        if any(parent_rows.get(row["id"]) != row for row in splits[child]):
            raise ValueError(f"{child} is not an exact frozen subset of {parent}")
    main = splits["calibration80"] + splits["selection120"] + splits["math500"]
    if any(len({row[key] for row in main}) != len(main) for key in ("id", "problem_sha256")):
        raise ValueError("Frozen calibration, selection, test overlap")
    selected = next((row for row in splits["pilot20"] if row["id"] == sample_id), None)
    if selected is None:
        raise ValueError("--sample-id must be one of the frozen pilot20 IDs")
    if selected.get("needs_review") or any(not isinstance(selected.get(key), str) or not selected[key].strip()
                                           for key in ("id", "problem", "answer")):
        raise ValueError("Selected question has no reviewed nonempty problem/answer")
    if sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("Frozen manifest changed during verification")
    return selected, {"manifest_sha256": expected_manifest_sha256, "sources": lock,
                      "files_sha256": {name: info["sha256"] for name, info in files.items()},
                      "selected_row_sha256": hashlib.sha256(encoded(selected)).hexdigest()}


def derive_seed(master_seed, sample_id, rollout_id, domain):
    payload = json.dumps(["online-diagnostic-seed-v1", master_seed, sample_id, rollout_id, domain],
                         ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)


def configurations(labels, fixed_interval, *, h_max=ScheduleConfig().h_max,
                   beta=ScheduleConfig().beta, log_a=ScheduleConfig().log_a,
                   random_p=ScheduleConfig().random_p, margin_m0=ScheduleConfig().margin_m0,
                   protocol_config=ProtocolConfig()):
    if not labels or len(set(labels)) != len(labels) or set(labels) - set(SUPPORTED_CONFIGS):
        raise ValueError("Configurations must be distinct supported labels")
    # Validate all supplied options before any GPU or filesystem mutation. Each
    # family's recorded configuration below contains only its applicable overrides.
    ScheduleConfig(fixed_interval=fixed_interval, h_max=h_max, beta=beta, log_a=log_a,
                   random_p=random_p, margin_m0=margin_m0)
    if not isinstance(protocol_config, ProtocolConfig):
        raise ValueError("protocol_config must be a ProtocolConfig")
    result = []
    for label in labels:
        method = {"vanilla": "vanilla", "deer-dense": "deer",
                  "dense-collect-no-stop": "dense_collect_no_stop"}.get(label, "codestop")
        kind = label.removeprefix("codestop-") if label.startswith("codestop-") else "dense"
        options = {"fixed": {"fixed_interval": fixed_interval},
                   "log": {"log_a": log_a, "h_max": h_max},
                   "random": {"random_p": random_p},  # The declared random cap stays eight.
                   "backoff": {"margin_m0": margin_m0, "h_max": h_max},
                   "adaptive": {"beta": beta, "h_max": h_max}}.get(kind, {})
        result.append({"label": label, "method": method,
                       "schedule_config": ScheduleConfig(kind=kind, **options),
                       "protocol_config": replace(protocol_config, rule="deer" if method == "deer" else "codestop")})
    return result


def configuration_record(config, *, seed_reason, seed_schedule, max_new_tokens, attention_implementation):
    """Hash the exact declared request settings, including the independent RNGs."""
    record = {"label": config["label"], "method": config["method"],
              "protocol_config": asdict(config["protocol_config"]),
              "schedule_config": asdict(config["schedule_config"]),
              "seed_reason": seed_reason,
              "seed_schedule": seed_schedule if config["schedule_config"].kind == "random" else None,
              "max_new_tokens": max_new_tokens, "attention_implementation": attention_implementation,
              "model_revision": MODEL_REVISION,
              "stopping_enabled": config["method"] not in ("vanilla", "dense_collect_no_stop")}
    return {**record, "config_hash": hashlib.sha256(encoded(record)).hexdigest()}


def command_output(command):
    return subprocess.run(command, text=True, capture_output=True, check=True, timeout=15).stdout.strip()


def gpu_inventory():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").strip()
    if not re.fullmatch(r"(?:\d+|GPU-[A-Za-z0-9-]+)", visible):
        raise ValueError("CUDA_VISIBLE_DEVICES must select exactly one numeric index or GPU UUID")
    row = command_output(["nvidia-smi", f"--id={visible}", "--query-gpu=index,uuid,name,memory.total,driver_version",
                          "--format=csv,noheader,nounits"])
    fields = [item.strip() for item in row.split(",")]
    if len(fields) != 5 or "\n" in row or not re.fullmatch(r"GPU-[A-Za-z0-9-]+", fields[1]):
        raise ValueError("Expected exactly one physical GPU from nvidia-smi")
    # Pin by UUID before torch is imported; numeric CUDA ordering cannot select a different card.
    os.environ["CUDA_VISIBLE_DEVICES"] = fields[1]
    return {"index": fields[0], "uuid": fields[1], "name": fields[2],
            "total_memory_mib": int(fields[3]), "driver_version": fields[4],
            "requested_cuda_visible_devices": visible, "effective_cuda_visible_devices": fields[1]}


@contextmanager
def gpu_lock(gpu_uuid, *, directory=Path("/tmp")):
    """A stable physical-GPU lock, independent of checkout, run root, and PID."""
    if not re.fullmatch(r"GPU-[A-Za-z0-9-]+", gpu_uuid):
        raise ValueError("Invalid GPU UUID for lock")
    path = Path(directory) / f"codestop-common-online-{gpu_uuid}.lock"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"GPU already locked by a common-online runner: {path}") from error
        os.ftruncate(descriptor, 0)
        os.write(descriptor, encoded({"pid": os.getpid(), "gpu_uuid": gpu_uuid}))
        yield str(path)
    finally:
        os.close(descriptor)  # Never unlink: other processes may hold the same inode.


def assert_gpu_idle(gpu_uuid):
    processes = command_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
                                "--format=csv,noheader,nounits"])
    active = [line for line in processes.splitlines() if line.split(",", 1)[0].strip() == gpu_uuid]
    if active:
        raise RuntimeError("Selected GPU already has compute processes: " + "; ".join(active))
    return {"compute_processes": [], "checked_at_utc": datetime.now(timezone.utc).isoformat()}


class EventLog:
    def __init__(self, path):
        self.stream = Path(path).open("x", encoding="utf-8")

    def emit(self, event, **fields):
        record = {"time_utc": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
        text = json.dumps(json_safe(record), ensure_ascii=False, allow_nan=False)
        self.stream.write(text + "\n")
        self.stream.flush()
        print(text, flush=True)

    def close(self):
        self.stream.close()


class ProgressBackend:
    """Delegation only: the engine retains KV, boundary, RNG and stopping ownership."""
    def __init__(self, backend, log, label, every=64):
        self.backend, self.log, self.label, self.every = backend, log, label, every
        self.main_count = self.probe_count = 0

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def sample(self, state):
        token = self.backend.sample(state)
        self.main_count += 1
        if self.main_count % self.every == 0:
            self.log.emit("main_progress", configuration=self.label, main_tokens=self.main_count,
                          probes=self.probe_count)
        return token

    def probe(self, state):
        self.probe_count += 1
        self.log.emit("probe_start", configuration=self.label, probe=self.probe_count,
                      main_tokens=self.main_count)
        observation = self.backend.probe(state)
        self.log.emit("probe_end", configuration=self.label, probe=self.probe_count,
                      confidence=observation.confidence_raw, ended_with_think=observation.ended_with_think,
                      probe_tokens=len(observation.token_ids))
        return observation


def answer_record(result, source_path, gold, backend, label):
    """Preserve internal reasoning markers; remove only a known terminal EOS token."""
    boundary = result.get("answer_boundary", {})
    confirmed = result.get("answer_boundary_confirmed", boundary.get("confirmed", False)) is True
    ids = list(result.get("answer_token_ids", [])) if confirmed else []
    source_span = boundary.get("answer_token_span")
    removed = []
    if ids and ids[-1] in backend.markers.eos_ids:
        removed = [{"token_id": ids[-1], "answer_token_index": len(ids) - 1,
                    "output_token_index": source_span[1] - 1 if source_span else None}]
        ids.pop()
    return {"id": f"{result['sample_id']}::rollout-{result['rollout_id']}::{label}",
            "sample_id": result["sample_id"], "rollout_id": result["rollout_id"],
            "configuration": label, "method": result["method"],
            "execution_status": result["status"], "gold": gold,
            "answer_text": backend.decode(ids, skip_special_tokens=False) if ids else "",
            "answer_token_ids": ids, "answer_boundary_confirmed": confirmed,
            "boundary_policy": boundary.get("policy", "token_span_first_reasoning_exit_v1"),
            "answer_boundary": boundary, "removed_terminal_eos": removed,
            "recognized_eos_token_ids": list(backend.markers.eos_ids),
            "source_file": str(Path(source_path).name), "source_sha256": sha256(source_path),
            "stop_reason": result.get("stop_reason", result.get("stop_reason_before_error"))}


def run_diagnostic(args, *, backend_factory=None):
    if not 1 <= args.max_new_tokens <= 8192:
        raise ValueError("Diagnostic --max-new-tokens must be explicit and in [1,8192]")
    if not 0 <= args.master_seed < 2**63 or args.rollout_id < 0:
        raise ValueError("master-seed must be in [0,2**63); rollout-id must be nonnegative")
    configs = configurations(args.configurations, args.fixed_interval, h_max=args.h_max,
                             beta=args.beta, log_a=args.log_a, random_p=args.random_p,
                             margin_m0=args.margin_m0,
                             protocol_config=ProtocolConfig(r_max=args.r_max, tau=args.tau,
                                                            deer_threshold=args.deer_threshold))
    row, data_identity = verify_data(args.data_dir, args.data_manifest_sha256, args.sample_id)
    run_root = args.run_root.expanduser().absolute()
    if run_root.exists() or run_root.is_symlink():
        raise ValueError("Run directory already exists; choose a new --run-root")
    model_dir = args.model_dir.expanduser().resolve()
    if not model_dir.is_dir() or model_dir.name != MODEL_REVISION:
        raise ValueError("--model-dir must be an existing pinned revision snapshot")
    gpu = gpu_inventory()
    with gpu_lock(gpu["uuid"]) as lock_path:
        idle = assert_gpu_idle(gpu["uuid"])
        run_root.mkdir(parents=True, exist_ok=False)
        log = EventLog(run_root / "events.jsonl")
        answers, completed, failure = [], [], None
        code_hashes = {name: sha256(ROOT / name) for name in CODE_FILES}
        reason_seed = derive_seed(args.master_seed, row["id"], args.rollout_id, "reason")
        schedule_seed = derive_seed(args.master_seed, row["id"], args.rollout_id, "schedule")
        config_records = [configuration_record(config, seed_reason=reason_seed, seed_schedule=schedule_seed,
                                                max_new_tokens=args.max_new_tokens,
                                                attention_implementation=args.attention_implementation)
                          for config in configs]
        manifest = {"schema_version": 1, "scope": SCOPE, "runner_protocol": RUNNER_PROTOCOL,
                    "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                    "sample_id": row["id"], "problem_sha256": row["problem_sha256"],
                    "question_count": 1, "rollout_id": args.rollout_id, "planned_count": len(configs),
                    "max_new_tokens": args.max_new_tokens, "master_seed": args.master_seed,
                    "seed_reason": reason_seed, "reserved_schedule_seed": schedule_seed,
                    "schedule_rng_used": any(config["schedule_config"].kind == "random" for config in configs),
                    "schedule_rng_used_semantics": "random schedule configured; actual draws depend on observed candidates",
                    "seed_algorithm": "online-diagnostic-seed-v1 / SHA256 first 8 bytes mod 2**63; domain separated",
                    "configurations": config_records, "data_identity": data_identity,
                    "code_sha256": code_hashes, "gpu": gpu, "gpu_lock": lock_path,
                    "warmup": {"main_tokens": 16, "max_final_tokens": 30, "method": "vanilla",
                               "includes_probe_warmup": False, "included_in_measured_requests": False},
                    "logging": {"main_progress_every": 64, "probe_start_end": True,
                                "probe_decisions": "post_request_summary",
                                "synchronous_logging_overhead_in_request_timing": True},
                    "grading": "separate scripts/grade_math_answers.py invocation; never inside request timer"}
        atomic_new(run_root / "manifest.json", manifest)
        try:
            log.emit("preflight_passed", sample_id=row["id"], gpu=gpu["name"],
                     max_new_tokens=args.max_new_tokens, planned_count=len(configs))
            if backend_factory is None:
                from torch_online_backend import TorchOnlineBackend
                backend_factory = TorchOnlineBackend
            started = time.perf_counter()
            log.emit("model_load_start", snapshot=MODEL_REVISION)
            backend = backend_factory(model_dir, attention_implementation=args.attention_implementation)
            environment = {"python": sys.version, "platform": platform.platform(), "executable": sys.executable,
                           "gpu": gpu, "idle_check": idle, "backend_metadata": backend.metadata,
                           "model_load_seconds_external": time.perf_counter() - started,
                           "package_versions": {name: importlib.metadata.version(name)
                                                for name in ("torch", "transformers")}}
            atomic_new(run_root / "environment.json", environment)
            log.emit("model_load_completed", elapsed_seconds=environment["model_load_seconds_external"])
            log.emit("warmup_start", excluded_from_results=True)
            try:
                warmup = run_request(ProgressBackend(backend, log, "warmup"), question="Compute 1 + 1.",
                                     sample_id="synthetic-warmup-only", rollout_id=0, method="vanilla",
                                     seed_reason=derive_seed(args.master_seed, row["id"], args.rollout_id, "warmup"),
                                     max_new_tokens=16)
            except RequestError as error:
                atomic_new(run_root / "warmup.json", {**error.partial, "scope": "excluded_warmup"})
                raise
            atomic_new(run_root / "warmup.json", {**warmup, "scope": "excluded_warmup"})
            log.emit("warmup_completed", elapsed_ms=warmup["time_total_ms"])
            for config, config_record in zip(configs, config_records):
                label = config["label"]
                log.emit("request_start", configuration=label, method=config["method"],
                         schedule=config["schedule_config"].kind, seed_reason=reason_seed,
                         seed_schedule=config_record["seed_schedule"], config_hash=config_record["config_hash"])
                try:
                    result = run_request(ProgressBackend(backend, log, label), question=row["problem"],
                                         sample_id=row["id"], rollout_id=args.rollout_id,
                                         method=config["method"], seed_reason=reason_seed,
                                         seed_schedule=config_record["seed_schedule"],
                                         max_new_tokens=args.max_new_tokens,
                                         protocol_config=config["protocol_config"],
                                         schedule_config=config["schedule_config"])
                except RequestError as error:
                    result = error.partial
                    failure = {"type": type(error).__name__, "message": str(error), "configuration": label}
                except BaseException as error:
                    # The engine catches ordinary exceptions with full partials. Interruptions
                    # still represent an executed request, even when it cannot return a partial.
                    result = {"status": "failed", "sample_id": row["id"], "rollout_id": args.rollout_id,
                              "method": config["method"], "seed_reason": reason_seed,
                              "max_new_tokens": args.max_new_tokens, "partial_evidence_available": False,
                              "error_type": type(error).__name__, "error": str(error)}
                    failure = {"type": type(error).__name__, "message": str(error), "configuration": label}
                result.update(configuration=label, scope=SCOPE, eligible_for_primary_speed_comparison=False,
                              schedule_config=asdict(config["schedule_config"]),
                              protocol_config=asdict(config["protocol_config"]),
                              seed_schedule=config_record["seed_schedule"],
                              config_hash=config_record["config_hash"], request_configuration=config_record,
                              backend_metadata=backend.metadata)
                if "peak_memory_bytes" not in result:
                    try:
                        result["peak_memory_bytes"] = backend.peak_memory_bytes()
                    except Exception as memory_error:
                        result["peak_memory_error"] = type(memory_error).__name__
                source_path = run_root / f"{label}.json"
                atomic_new(source_path, result)
                # The engine owns decisions and returns them only after the request;
                # these are retrospective summaries, unlike probe_start/probe_end.
                for probe in result.get("probes", []):
                    if "decision" in probe:
                        log.emit("probe_decision", timing="post_request_summary", configuration=label,
                                 decision=probe["decision"], would_stop=probe.get("would_stop"),
                                 should_stop=probe.get("should_stop"), stop_applied=probe.get("stop_applied"))
                try:
                    answer = answer_record(result, source_path, row["answer"], backend, label)
                except Exception as export_error:
                    # Source evidence is already saved: retain this executed request in the
                    # grading denominator even if the final-answer decoding/export failed.
                    failure = {"type": type(export_error).__name__, "message": str(export_error),
                               "configuration": label, "stage": "final_answer_export"}
                    answer = answer_record({**result, "status": "failed", "answer_token_ids": []},
                                           source_path, row["answer"], backend, label)
                    answer.update(source_execution_status=result["status"], export_error=failure)
                atomic_new(run_root / f"{label}.final-answer.json", answer)
                answers.append(answer)
                completed.append({"configuration": label, "status": answer["execution_status"],
                                  "source_execution_status": result["status"],
                                  "source_sha256": answer["source_sha256"]})
                log.emit("request_completed" if failure is None else "request_failed", configuration=label,
                         status=answer["execution_status"], elapsed_ms=result.get("time_total_ms"),
                         probes=result.get("n_probes", len(result.get("probes", []))),
                         stop_reason=result.get("stop_reason"), answer_boundary_confirmed=answer["answer_boundary_confirmed"])
                if failure:
                    break  # Preserve a failed row, never silently retry/expand.
        except BaseException as error:
            failure = {"type": type(error).__name__, "message": str(error)}
            log.emit("run_failed", **failure)
        finally:
            text = "".join(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n" for item in answers)
            atomic_new(run_root / "final_answers.jsonl", text.encode("utf-8"), raw=True)
            current_hashes = {name: sha256(ROOT / name) for name in CODE_FILES}
            if current_hashes != code_hashes:
                failure = {"type": "SourceChanged", "message": "Runner source changed during execution"}
            summary = {"scope": SCOPE, "status": "failed" if failure else "completed",
                       "planned_count": len(configs), "executed_count": len(answers),
                       "unexecuted_count": len(configs) - len(answers), "requests": completed,
                       "failure": failure, "warmup_excluded": True, "grading_status": "not_run",
                       "code_unchanged_during_run": current_hashes == code_hashes,
                       "final_answers_sha256": sha256(run_root / "final_answers.jsonl")}
            atomic_new(run_root / "summary.json", summary)
            log.emit("run_finished", status=summary["status"], executed_count=len(answers),
                     planned_count=len(configs), run_root=str(run_root))
            log.close()
        return 1 if failure else 0


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-dir", type=Path, default=ROOT / "data/benchmarks")
    result.add_argument("--data-manifest-sha256", required=True, help="independently checked frozen manifest SHA-256")
    result.add_argument("--model-dir", type=Path, required=True, help="existing local pinned HF snapshot; no download")
    result.add_argument("--run-root", type=Path, required=True, help="new directory only; no resume or overwrite")
    result.add_argument("--sample-id", default=DEFAULT_SAMPLE, help="exact pilot20 source ID; default already-exposed q002")
    result.add_argument("--max-new-tokens", type=int, required=True, help="explicit main token cap, 1..8192; first acceptance 1024")
    result.add_argument("--configurations", nargs="+", choices=SUPPORTED_CONFIGS, default=list(DEFAULT_CONFIGS))
    schedule, protocol = ScheduleConfig(), ProtocolConfig()
    result.add_argument("--fixed-interval", type=int, default=schedule.fixed_interval,
                        help="fixed candidate interval after shared warmup, 1..9")
    result.add_argument("--h-max", type=int, default=schedule.h_max,
                        help="log/backoff/adaptive cap: 2, 4, or 8; random always has cap 8")
    result.add_argument("--beta", type=float, default=schedule.beta, help="adaptive cost target: 0.25, 0.5, or 1")
    result.add_argument("--log-a", type=float, default=schedule.log_a, help="log schedule coefficient: 0.5, 1, or 2")
    result.add_argument("--random-p", type=float, default=schedule.random_p, help="capped geometric probability from declared grid")
    result.add_argument("--margin-m0", type=float, default=schedule.margin_m0, help="backoff margin: 0.02, 0.05, or 0.10")
    result.add_argument("--r-max", type=float, default=protocol.r_max, help="CoDE maximum confidence threshold")
    result.add_argument("--tau", type=float, default=protocol.tau, help="CoDE positive degeneration threshold")
    result.add_argument("--deer-threshold", type=float, default=protocol.deer_threshold, help="DEER confidence threshold")
    result.add_argument("--rollout-id", type=int, default=0)
    result.add_argument("--master-seed", type=int, default=42)
    result.add_argument("--attention-implementation", choices=("eager", "sdpa"), default="eager")
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    try:
        return run_diagnostic(arguments)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Online diagnostic refused: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
