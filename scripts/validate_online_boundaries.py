#!/usr/bin/env python3
"""Three real-model natural-boundary checks on one fixed synthetic question.

No tokens are forced or substituted. First sample Vanilla to natural thinking
closure and EOS. Only if that succeeds, repeat with exactly the same seed and
a main budget ending at the observed closure. The second request must reproduce
the sampled main prefix and complete its answer without an injected prefix.
The third budget ends at the first naturally generated token whose decoded
answer prefix contains non-whitespace body text. It is selected only from the
first completed natural-EOS record, without looking at a gold answer.
This validates execution branches, never dataset accuracy or model speed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from online_contract import MODEL_ID, MODEL_REVISION, RUNNER_PROTOCOL
from online_engine import FINAL_TOKEN_CAP, RequestError, run_request
from online_source_manifest import METHOD_IDENTITY_FILES
from run_online_diagnostic import EventLog, ProgressBackend, assert_gpu_idle, atomic_new, gpu_inventory, gpu_lock, sha256
from validate_online_gpu import fresh_run, first_difference

SCOPE = "real_model_natural_boundary_validation_on_synthetic_question"
QUESTION = "Compute 1 + 1."
SAMPLE_ID = "synthetic-boundary-compute-1-plus-1"
SEED_REASON = 872
LABELS = ("natural-eos", "natural-answer-budget", "natural-answer-body-budget")
BRANCHES = ("actual_natural_eos", "actual_answer_phase_budget", "actual_nonempty_answer_phase_budget")
SOURCE_FILES = ("scripts/validate_online_boundaries.py", "scripts/run_online_diagnostic.py",
                "scripts/validate_online_gpu.py", "src/online_contract.py", "src/online_engine.py",
                "src/online_protocol.py", "src/torch_online_backend.py", *METHOD_IDENTITY_FILES)


class CoverageUnmet(RuntimeError):
    pass


def validate_backend(metadata):
    required = {"validation_model": False, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                "model_parameter_dtype": "torch.bfloat16", "attention_implementation": "eager", "device": "cuda:0"}
    if metadata.get("validation_model") is not False or any(metadata.get(key) != value for key, value in required.items()):
        raise ValueError("Natural-boundary GPU execution requires the real pinned Qwen model in CUDA BF16/eager")


def natural_budget(record):
    """The recorded first natural </think> position determines the second budget."""
    boundary = record.get("answer_boundary", {})
    if (record.get("status") != "completed" or record.get("stop_reason") != "natural_eos"
            or boundary.get("source") != "natural_first_end_think"
            or boundary.get("confirmed") is not True or record.get("injected_prompt_tokens") != 0
            or record.get("finalization_tokens") != 0):
        raise CoverageUnmet("First fixed-seed request did not cover natural thinking closure and natural EOS within its cap")
    close = boundary.get("closing_token_index")
    samples, output = record.get("main_samples", []), record.get("output_token_ids", [])
    markers = record.get("backend_metadata", {}).get("markers", {})
    if (type(close) is not int or not 0 <= close < len(samples) - 1 or not output or close >= len(output)
            or output[-1] not in markers.get("eos_ids", ())
            or samples[close].get("token_id") != markers.get("end_think")
            or output[close] != markers.get("end_think")
            or [item.get("token_id") for item in samples] != output
            or record.get("discarded_generated_tokens") != 0):
        raise CoverageUnmet("Natural boundary token coordinates or sampled EOS evidence are inconsistent")
    return close + 1


def answer_body_budget(first, backend):
    """Find the first visible natural answer prefix, excluding control markers."""
    answer_start = natural_budget(first)
    samples = first["main_samples"]
    markers = first["backend_metadata"]["markers"]
    controls = set(getattr(backend.tokenizer, "all_special_ids", ()))
    controls.update(markers["eos_ids"])
    controls.add(markers["end_think"])
    natural_answer_ids = [sample["token_id"] for sample in samples[answer_start:-1]]
    # Validate the whole natural answer before choosing a convenient stopping
    # position. A repeated reasoning marker anywhere is an uncovered branch.
    for token in natural_answer_ids:
        decoded = backend.decode((token,), skip_special_tokens=False).strip()
        if token in controls or decoded in ("<think>", "</think>") or (decoded.startswith("<|") and decoded.endswith("|>")):
            raise CoverageUnmet("Natural answer contains a control marker; no answer-body budget is selected")
    for offset in range(len(natural_answer_ids)):
        index = answer_start + offset
        if samples[index].get("phase") != "answer":
            raise CoverageUnmet("Natural answer token is not recorded in the answer phase")
        prefix = natural_answer_ids[:offset + 1]
        text = backend.decode(prefix, skip_special_tokens=False)
        if text.strip():
            return {"max_new_tokens": index + 1, "selected_output_token_index": index,
                    "selected_token_id": samples[index]["token_id"], "answer_start_token_index": answer_start,
                    "natural_closing_token_index": answer_start - 1,
                    "natural_answer_main_tokens_at_budget": offset + 1,
                    "decoded_answer_prefix": text, "skipped_leading_blank_prefixes": offset,
                    "terminal_EOS_index": len(samples) - 1, "terminal_EOS_after_selection": True,
                    "selection_uses_gold": False,
                    "selection_rule": "first non-control non-EOS token yielding a non-whitespace decoded natural answer prefix"}
    raise CoverageUnmet("Natural EOS followed no nonempty answer body; third branch remains uncovered")


def validate_pair(first, second, budget, *, body_selection=None):
    closing_budget = natural_budget(first)
    expected_budget = closing_budget if body_selection is None else body_selection["max_new_tokens"]
    if budget != expected_budget:
        raise CoverageUnmet("Repeated budget does not equal its predeclared observed-boundary derivation")
    boundary = second.get("answer_boundary", {})
    if (second.get("status") != "completed" or second.get("stop_reason") != "budget"
            or second.get("max_new_tokens") != budget or boundary.get("source") != "natural_first_end_think"
            or boundary.get("confirmed") is not True or second.get("injected_prompt_tokens") != 0
            or not 1 <= second.get("finalization_tokens", 0) <= FINAL_TOKEN_CAP
            or len(second.get("main_samples", [])) != budget
            or boundary.get("closing_token_index") != closing_budget - 1
            or second.get("seed_reason") != first.get("seed_reason")
            or first.get("seed_reason") != SEED_REASON):
        raise CoverageUnmet("Second request did not cover bounded answer completion after the same natural closure")
    difference = first_difference(first["main_samples"][:budget], second["main_samples"])
    if difference:
        raise CoverageUnmet("Repeated real main sampling diverged before its declared budget: " + json.dumps(difference))
    if second["output_token_ids"][:budget] != first["output_token_ids"][:budget]:
        raise CoverageUnmet("Repeated request output prefix differs despite its sample record")
    if body_selection is not None and (budget <= closing_budget or second["main_samples"][-1].get("phase") != "answer"
            or second["main_samples"][-1].get("token_id") != body_selection["selected_token_id"]
            or second["main_samples"][-1]["token_id"] in first["backend_metadata"]["markers"]["eos_ids"]
            or not body_selection["decoded_answer_prefix"].strip()):
        raise CoverageUnmet("Third request did not reach the selected nonempty natural answer prefix before its budget")
    return {"passed": True, "comparison": "exact main_samples including tokens, raw/sample probabilities, phases and indices",
            "same_seed": SEED_REASON, "main_budget": budget, "natural_closing_token_index": closing_budget - 1,
            "natural_eos_first_request": True, "injected_prompt_tokens_both_requests": 0,
            "second_finalization_tokens": second["finalization_tokens"], "forced_tokens": False,
            "nonempty_answer_body_at_budget": body_selection is not None,
            "answer_body_selection": body_selection}


def record_request(backend, log, run_root, label, budget, completed):
    log.emit("request_start", configuration=label, seed_reason=SEED_REASON, max_new_tokens=budget,
             forced_tokens=False, synthetic_question=True)
    error = None
    try:
        result = run_request(ProgressBackend(backend, log, label, every=64), question=QUESTION,
            sample_id=SAMPLE_ID, rollout_id=0, method="vanilla", seed_reason=SEED_REASON,
            max_new_tokens=budget)
    except RequestError as exc:
        result, error = exc.partial, exc
    except (Exception, KeyboardInterrupt) as exc:
        result = {"status": "failed", "sample_id": SAMPLE_ID, "rollout_id": 0, "method": "vanilla",
                  "seed_reason": SEED_REASON, "max_new_tokens": budget, "partial_evidence_available": False,
                  "error_type": type(exc).__name__, "error": str(exc)}
        error = exc
    result.update(configuration=label, scope=SCOPE, forced_tokens=False, synthetic_question=True,
                  eligible_for_primary_speed_comparison=False, backend_metadata=backend.metadata,
                  generation_source="real_model_sampling_then_engine_greedy_completion_if_budget_reached")
    destination = run_root / f"{label}.json"
    atomic_new(destination, result)
    completed.append({"configuration": label, "status": result["status"], "source_sha256": sha256(destination)})
    log.emit("request_saved", configuration=label, status=result["status"], stop_reason=result.get("stop_reason"),
             main_tokens=result.get("main_generated_tokens"), finalization_tokens=result.get("finalization_tokens"))
    if error is not None:
        raise error
    return result


def run_validation(args, *, backend_factory=None):
    if args.max_main_tokens not in (512, 1024, 4096):
        raise ValueError("Choose a predeclared main cap of 512, 1024 or 4096; no automatic escalation")
    hashes = {name: sha256(ROOT / name) for name in SOURCE_FILES}
    manifest = {"schema_version": 1, "scope": SCOPE, "runner_protocol": RUNNER_PROTOCOL,
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "sample_id": SAMPLE_ID,
        "question": QUESTION, "synthetic_question": True, "forced_tokens": False,
        "eligible_for_primary_speed_comparison": False, "question_count": 1, "rollout_id": 0,
        "seed_reason": SEED_REASON, "seed_policy": "fixed predeclared seed; no reselection or retries",
        "planned_count": 3, "configurations": list(LABELS), "max_main_tokens_first": args.max_main_tokens,
        "second_main_budget_policy": "first natural closing_token_index plus one, only after first natural EOS",
        "third_main_budget_policy": "index plus one of the first non-control non-EOS token making the decoded natural answer prefix non-whitespace",
        "third_selection_conditions": ["first request has natural EOS and natural thinking closure",
            "natural answer contains no control markers", "selection skips whitespace without using gold",
            "a terminal natural EOS remains after the selected answer-body token"],
        "max_final_tokens": FINAL_TOKEN_CAP, "attention_implementation": "eager", "dtype": "BF16",
        "code_sha256": hashes, "warmup": "not repeated; no timing performance claim",
        "grading": "not performed", "logging": {"main_progress_every": 64},
        "failure_policy": "save uncovered/failed and stop; no new seed or larger budget"}
    if args.inspect_only:
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.model_dir is None or args.run_root is None:
        raise ValueError("GPU execution requires --model-dir and a new --run-root")
    if args.run_root.resolve().is_relative_to(args.model_dir.resolve()):
        raise ValueError("Run output must be outside the read-only model snapshot")
    with fresh_run(args.run_root) as run_root:
        atomic_new(run_root / "manifest.json", manifest)
        log = EventLog(run_root / "events.jsonl")
        completed, failure, pair, body_pair = [], None, None, None
        covered, coverage_checks = [], {label: {"status": "not_run"} for label in LABELS}
        active_label = LABELS[0]
        try:
            gpu = gpu_inventory()
            with gpu_lock(gpu["uuid"]) as lock_path:
                idle = assert_gpu_idle(gpu["uuid"])
                atomic_new(run_root / "gpu-preflight.json", {"gpu": gpu, "lock": lock_path, "idle": idle})
                log.emit("model_load_start", gpu=gpu["name"], model_revision=MODEL_REVISION)
                if backend_factory is None:
                    from torch_online_backend import TorchOnlineBackend
                    backend_factory = TorchOnlineBackend
                backend = backend_factory(args.model_dir, attention_implementation="eager")
                validate_backend(backend.metadata)
                atomic_new(run_root / "environment.json", {"backend_metadata": backend.metadata, "gpu": gpu})
                first = record_request(backend, log, run_root, LABELS[0], args.max_main_tokens, completed)
                budget = natural_budget(first)
                covered.append(BRANCHES[0])
                coverage_checks[LABELS[0]] = {"status": "covered", "branch": BRANCHES[0],
                    "source_sha256": completed[0]["source_sha256"], "natural_closing_token_index": budget - 1}
                atomic_new(run_root / "second-request-plan.json", {"source_file": f"{LABELS[0]}.json",
                    "source_sha256": completed[0]["source_sha256"], "max_new_tokens": budget,
                    "seed_reason": SEED_REASON, "derivation": "observed natural closing_token_index plus one"})
                active_label = LABELS[1]
                second = record_request(backend, log, run_root, LABELS[1], budget, completed)
                pair = validate_pair(first, second, budget)
                atomic_new(run_root / "boundary-pair-validation.json", pair)
                covered.append(BRANCHES[1])
                coverage_checks[LABELS[1]] = {"status": "covered", "branch": BRANCHES[1],
                    "source_sha256": completed[1]["source_sha256"], "validation_file": "boundary-pair-validation.json"}
                active_label = LABELS[2]
                body_selection = answer_body_budget(first, backend)
                atomic_new(run_root / "third-request-plan.json", {**body_selection,
                    "source_file": f"{LABELS[0]}.json", "source_sha256": completed[0]["source_sha256"],
                    "seed_reason": SEED_REASON})
                third = record_request(backend, log, run_root, LABELS[2], body_selection["max_new_tokens"], completed)
                body_pair = validate_pair(first, third, body_selection["max_new_tokens"], body_selection=body_selection)
                atomic_new(run_root / "answer-body-pair-validation.json", body_pair)
                covered.append(BRANCHES[2])
                coverage_checks[LABELS[2]] = {"status": "covered", "branch": BRANCHES[2],
                    "source_sha256": completed[2]["source_sha256"], "validation_file": "answer-body-pair-validation.json"}
        except (Exception, KeyboardInterrupt) as exc:
            failure = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
            coverage_checks[active_label] = {"status": "uncovered" if isinstance(exc, CoverageUnmet) else "failed",
                                             "error_type": type(exc).__name__, "message": str(exc)}
            atomic_new(run_root / "failure.json", failure)
            log.emit("boundary_validation_uncovered" if isinstance(exc, CoverageUnmet) else "boundary_validation_failed",
                     error_type=type(exc).__name__, message=str(exc))
        finally:
            try:
                after = {name: sha256(ROOT / name) for name in SOURCE_FILES}
            except OSError as exc:
                after = {"source_read_error": str(exc)}
            unchanged = hashes == after
            atomic_new(run_root / "source-hashes-after.json", after)
            if not unchanged:
                failure = {"type": "SourceChanged", "message": "Execution source changed during this bounded validation"}
                atomic_new(run_root / "source-integrity-failure.json", failure)
            status = ("uncovered" if failure and failure["type"] == "CoverageUnmet" else "failed") if failure else "completed"
            summary = {"schema_version": 1, "scope": SCOPE, "status": status, "planned_count": 3,
                "executed_count": len(completed), "unexecuted_count": 3 - len(completed), "requests": completed,
                "code_unchanged_during_run": unchanged, "failure": failure, "forced_tokens": False,
                "synthetic_question": True, "eligible_for_primary_speed_comparison": False,
                "covered_branches": covered, "coverage_checks": coverage_checks,
                "coverage_complete": len(covered) == len(BRANCHES) and failure is None,
                "coverage_validated_under_unchanged_source": unchanged,
                "boundary_pair_passed": pair is not None, "answer_body_pair_passed": body_pair is not None,
                "grading_status": "not_run",
                "timing_interpretation": "branch validation only; no speed claim"}
            atomic_new(run_root / "summary.json", summary)
            log.emit("run_finished", status=status, executed_count=len(completed), boundary_pair_passed=summary["boundary_pair_passed"])
            log.close()
        return 0 if failure is None else 2 if status == "uncovered" else 1


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model-dir", type=Path)
    result.add_argument("--run-root", type=Path)
    result.add_argument("--max-main-tokens", type=int, choices=(512, 1024, 4096), default=512)
    result.add_argument("--inspect-only", action="store_true")
    return result


if __name__ == "__main__":
    raise SystemExit(run_validation(parser().parse_args()))
