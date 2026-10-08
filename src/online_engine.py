"""Backend-independent online state machine; CPU fixtures are not GPU evidence.

Boundary precedence: every main token is sampled once, before KV acceptance.
Natural EOS ends immediately without a probe or answer rescue. The first natural
end-think token is accepted and closes the reasoning span. While still thinking,
Wait is a candidate at the prefix EXCLUDING that pending token, even on the last
budget slot. A continuing decision accepts the same Wait; an early stop discards
it but counts its generation. A candidate at prefix position zero is an explicit
request error. Budget exhaustion is handled after these boundaries. Forced
completion is greedy and bounded to 30 tokens; a naturally closed answer is
continued without injecting another prefix. Answer spans never use rsplit.

Per-operation synchronizations are intentional profiling overhead. They are
included in disjoint phase wall-clock measurements and in request total. Model
loading, external file writes, and grading are outside this function's timer.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import math
import time
from typing import Any

from online_contract import RUNNER_PROTOCOL, OnlineBackend, TokenSample
from online_protocol import (
    MAX_PROBE_TOKENS, CostObservation, ProbeObservation, ProtocolConfig,
    ProtocolController, ScheduleConfig,
)

ENGINE_VERSION = "1.0.0"
FINAL_TOKEN_CAP = 30
METHODS = ("vanilla", "deer", "codestop", "dense_collect_no_stop")


def json_safe(value):
    """Keep invalid numerical evidence explicit, never replace it with a score."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": repr(value)}
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


class RequestError(RuntimeError):
    def __init__(self, message: str, partial: dict[str, Any]):
        super().__init__(message)
        self.partial = json_safe(partial)


def _ids(value, name):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} requires nonempty token IDs")
    if any(type(token) is not int or token < 0 for token in value):
        raise ValueError(f"{name} requires nonnegative integer token IDs")
    return list(value)


def _sample(value):
    if not isinstance(value, TokenSample) or type(value.token_id) is not int or value.token_id < 0:
        raise ValueError("backend sampling must return TokenSample with a nonnegative integer ID")
    for name in ("sample_probability", "raw_probability"):
        probability = getattr(value, name)
        if (not isinstance(probability, (float, int)) or isinstance(probability, bool)
                or not math.isfinite(probability) or not 0 <= probability <= 1):
            raise ValueError(f"backend returned invalid {name}")
    return value


def run_request(backend: OnlineBackend, *, question: str, sample_id: str,
                rollout_id: int, method: str, seed_reason: int,
                max_new_tokens: int = 32768, seed_schedule: int | None = None,
                protocol_config: ProtocolConfig = ProtocolConfig(),
                schedule_config: ScheduleConfig = ScheduleConfig()) -> dict:
    """Run one loaded-model request, returning evidence without grading/writes.

RequestError.partial retains committed output, pending sampled token, raw probes,
phase times and configuration. No caller should turn that exception into an
unrecorded missing row. dense_collect_no_stop deliberately ignores stop decisions
and must remain a diagnostic artifact, never an online-speed method result.
"""
    request_start = time.perf_counter()
    timings = {"prefill": 0.0, "reason": 0.0, "probe_cache": 0.0, "answer": 0.0}
    operation_counts = {name: 0 for name in timings}
    output_ids, prompt_ids, main_samples, final_samples = [], [], [], []
    probes, candidates = [], []
    pending = None
    accepted_main = discarded = injected = 0
    thinking = True
    answer_start = None
    boundary = {"confirmed": False, "policy": "token_span_first_reasoning_exit_v1",
                "source": None, "source_token_span": None}
    stop_reason = None
    stop_reasons = []
    finalization_end = None
    context_required = None
    controller = None
    state = None
    reason_since_query = 0.0
    previous_query = False
    effective_protocol = protocol_config

    def timed(phase, function, *args):
        nonlocal reason_since_query
        start = time.perf_counter()
        try:
            backend.synchronize()
            return function(*args)
        finally:
            try:
                backend.synchronize()
            finally:
                elapsed = (time.perf_counter() - start) * 1000
                timings[phase] += elapsed
                operation_counts[phase] += 1
                if phase == "reason":
                    reason_since_query += elapsed

    def partial():
        return {
            "status": "failed", "runner_protocol": RUNNER_PROTOCOL,
            "engine_version": ENGINE_VERSION, "sample_id": sample_id,
            "rollout_id": rollout_id, "method": method,
            "max_new_tokens": max_new_tokens, "seed_reason": seed_reason,
            "seed_schedule": seed_schedule, "prompt_token_ids": prompt_ids,
            "output_token_ids": output_ids, "main_samples": main_samples,
            "finalization_samples": final_samples, "pending_sample": pending,
            "n_main_samples": len(main_samples), "n_accepted_main_tokens": accepted_main,
            "discarded_generated_tokens": discarded, "injected_prompt_tokens": injected,
            "answer_boundary": boundary, "probes": probes, "candidates": candidates,
            "phase_elapsed_ms": timings.copy(), "operation_counts": operation_counts.copy(),
            "elapsed_ms_before_error_report": (time.perf_counter() - request_start) * 1000,
            "context_required": context_required, "stop_reason_before_error": stop_reason,
        }

    def accept_main(sample, phase):
        nonlocal state, pending, accepted_main
        state = timed(phase, backend.extend, state, (sample.token_id,))
        output_ids.append(sample.token_id)
        accepted_main += 1
        pending = None

    def complete_answer():
        nonlocal state, thinking, answer_start, boundary, injected, pending, finalization_end
        if thinking:
            prefix = tuple(backend.markers.final_prefix)
            start = len(output_ids)
            state = timed("answer", backend.extend, state, prefix)
            output_ids.extend(prefix)
            injected += len(prefix)
            thinking = False
            answer_start = start + 1
            boundary = {"confirmed": True, "policy": "token_span_first_reasoning_exit_v1",
                        "source": "controlled_final_prefix", "closing_token_index": start,
                        "source_token_span": [start, start + len(prefix)],
                        "includes_injected_boxed_prefix": True}
        for _ in range(FINAL_TOKEN_CAP):
            sample = _sample(timed("answer", backend.greedy, state))
            item = asdict(sample)
            item["generation_index"] = len(final_samples)
            final_samples.append(item)
            pending = {"phase": "finalization", **item}
            if sample.token_id == backend.markers.eos:
                output_ids.append(sample.token_id)
                pending = None
                finalization_end = "eos"
                break
            state = timed("answer", backend.extend, state, (sample.token_id,))
            output_ids.append(sample.token_id)
            pending = None
        else:
            finalization_end = "answer_budget"

    try:
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        if not isinstance(question, str) or not question.strip() or not isinstance(sample_id, str) or not sample_id:
            raise ValueError("question and sample_id must be nonempty strings")
        for name, value, minimum in (("rollout_id", rollout_id, 0), ("seed_reason", seed_reason, 0),
                                     ("max_new_tokens", max_new_tokens, 1)):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not isinstance(protocol_config, ProtocolConfig) or not isinstance(schedule_config, ScheduleConfig):
            raise ValueError("protocol_config and schedule_config must be their frozen dataclass types")
        if method in ("vanilla", "dense_collect_no_stop") and schedule_config.kind != "dense":
            raise ValueError(f"{method} requires an explicit dense schedule")
        if method == "vanilla" and seed_schedule is not None:
            raise ValueError("vanilla has no schedule RNG")
        effective_protocol = replace(protocol_config, rule="deer" if method == "deer" else "codestop")
        if method != "vanilla":
            controller = ProtocolController(effective_protocol, schedule_config, seed_schedule=seed_schedule,
                                            stop_enabled=method != "dense_collect_no_stop")
        prompt_ids = _ids(backend.prompt_ids(question), "prompt")
        reserve = max(len(backend.markers.final_prefix) + FINAL_TOKEN_CAP,
                      len(backend.markers.trial_prefix) + MAX_PROBE_TOKENS, FINAL_TOKEN_CAP)
        context_required = len(prompt_ids) + max_new_tokens + reserve
        if type(backend.context_limit) is not int or backend.context_limit <= 0:
            raise ValueError("backend.context_limit must be a positive integer")
        if context_required > backend.context_limit:
            raise ValueError(f"Context capacity insufficient: require {context_required}, have {backend.context_limit}")
        backend.start_request(seed_reason)
        state = timed("prefill", backend.prefill, prompt_ids)
        for main_index in range(max_new_tokens):
            phase = "reason" if thinking else "answer"
            sample = _sample(timed(phase, backend.sample, state))
            item = {**asdict(sample), "generation_index": main_index, "phase": phase}
            main_samples.append(item)
            pending = {"phase": "main", **item}
            token = sample.token_id
            if token == backend.markers.eos:
                output_ids.append(token)
                accepted_main += 1
                pending = None
                stop_reason = "natural_eos"
                break
            if thinking and token == backend.markers.end_think:
                close_position = len(output_ids)
                accept_main(sample, phase)
                thinking = False
                answer_start = len(output_ids)
                boundary = {"confirmed": True, "policy": "token_span_first_reasoning_exit_v1",
                            "source": "natural_first_end_think", "closing_token_index": close_position,
                            "source_token_span": [close_position, close_position + 1],
                            "includes_injected_boxed_prefix": False}
            elif thinking and token == backend.markers.wait:
                candidate_j = len(candidates) + 1
                token_position = len(output_ids)
                event = {"candidate_j": candidate_j, "token_position": token_position,
                         "pending_token_id": token, "queried": False}
                candidates.append(event)
                if token_position == 0:
                    raise ValueError("First-token Wait has prefix token_position=0; not silently shifted")
                if controller is not None and controller.should_probe(candidate_j):
                    before = timings["probe_cache"]
                    observation = timed("probe_cache", backend.probe, state)
                    probe_ms = timings["probe_cache"] - before
                    if not isinstance(observation, ProbeObservation):
                        raise ValueError("backend.probe must return ProbeObservation")
                    # Preserve raw observation before controller or token-identity validation.
                    probe_record = {"candidate_j": candidate_j, "token_position": token_position,
                                    "raw_observation": asdict(observation), "probe_elapsed_ms": probe_ms}
                    probes.append(probe_record)
                    event["queried"] = True
                    if observation.ended_with_think != bool(observation.token_ids and observation.token_ids[-1] == backend.markers.end_think):
                        raise ValueError("Probe ended_with_think does not match its actual last token")
                    if backend.markers.eos in observation.token_ids:
                        observation = replace(observation, invalid_reason="probe_contains_eos")
                    cost = CostObservation(probe_elapsed_ms=probe_ms,
                                           reason_elapsed_ms=reason_since_query if previous_query else None)
                    decision = controller.observe(candidate_j, token_position, observation, cost)
                    probe_record["decision"] = asdict(decision)
                    probe_record["would_stop"] = decision.would_stop
                    probe_record["should_stop"] = decision.should_stop
                    probe_record["stop_applied"] = decision.should_stop and method != "dense_collect_no_stop"
                    reason_since_query = 0.0
                    previous_query = True
                    if decision.should_stop and method != "dense_collect_no_stop":
                        discarded += 1
                        pending = None
                        stop_reasons = list(decision.stop_reasons)
                        stop_reason = "_and_".join(stop_reasons)
                        complete_answer()
                        break
                accept_main(sample, phase)
            else:
                accept_main(sample, phase)
            if main_index + 1 == max_new_tokens:
                stop_reason = "budget"
                complete_answer()
                break
        if stop_reason is None:
            raise RuntimeError("Main generation exited without a recorded termination reason")

        main_reason = sum(item["phase"] == "reason" for item in main_samples)
        natural_answer = len(main_samples) - main_reason
        probe_tokens = sum(len(record["raw_observation"]["token_ids"]) for record in probes)
        if len(main_samples) != accepted_main + discarded:
            raise RuntimeError("Main token conservation failed")
        if len(output_ids) != accepted_main + injected + len(final_samples):
            raise RuntimeError("Output token conservation failed")
        answer_ids = output_ids[answer_start:] if answer_start is not None else []
        boundary["answer_token_span"] = [answer_start, len(output_ids)] if answer_start is not None else None
        boundary["coordinate_system"] = "zero-based output_token_ids, half-open"
        record = {
            "schema_version": 1, "runner_protocol": RUNNER_PROTOCOL, "engine_version": ENGINE_VERSION,
            "status": "completed", "sample_id": sample_id, "rollout_id": rollout_id, "method": method,
            "scope": "dense_trajectory_diagnostic_only" if method == "dense_collect_no_stop" else "common_online_request",
            "eligible_for_primary_speed_comparison": method != "dense_collect_no_stop",
            "backend_metadata": backend.metadata, "protocol_config": asdict(effective_protocol),
            "schedule_config": asdict(schedule_config), "seed_reason": seed_reason, "seed_schedule": seed_schedule,
            "max_new_tokens": max_new_tokens, "answer_max_new_tokens": FINAL_TOKEN_CAP,
            "context_required": context_required, "context_limit": backend.context_limit,
            "prompt_token_ids": prompt_ids, "output_token_ids": output_ids,
            "full_sequence_token_ids": prompt_ids + output_ids,
            "output_text": backend.decode(output_ids, skip_special_tokens=True),
            "output_text_with_special_tokens": backend.decode(output_ids, skip_special_tokens=False),
            "answer_token_ids": answer_ids, "answer_text": backend.decode(answer_ids, skip_special_tokens=True),
            "answer_text_with_special_tokens": backend.decode(answer_ids, skip_special_tokens=False),
            "answer_boundary": boundary, "answer_boundary_confirmed": boundary["confirmed"],
            "stop_reason": stop_reason, "stop_reasons": stop_reasons, "finalization_end": finalization_end,
            "main_samples": main_samples, "finalization_samples": final_samples,
            "candidates": candidates, "probes": probes, "n_candidates": len(candidates), "n_probes": len(probes),
            "reason_tokens": main_reason, "natural_answer_tokens": natural_answer,
            "finalization_tokens": len(final_samples), "answer_tokens": natural_answer + len(final_samples),
            "main_generated_tokens": len(main_samples), "accepted_main_tokens": accepted_main,
            "probe_tokens": probe_tokens, "discarded_generated_tokens": discarded,
            "injected_prompt_tokens": injected, "actual_generated_tokens": len(main_samples) + len(final_samples) + probe_tokens,
            "output_token_count": len(output_ids), "token_conservation_verified": True,
            "peak_memory_bytes": backend.peak_memory_bytes(),
            "phase_elapsed_ms": timings.copy(), "operation_counts": operation_counts.copy(),
            "timing_scope": "loaded_model_request_including_decode_and_record_assembly_excluding_file_io_and_grading",
            "profiling_policy": "synchronous_before_and_after_each_operation_overhead_included",
            "notes": ["Generated discarded Wait tokens are included in reason_tokens, not added twice.",
                      "Injected final-prefix tokens are input work, not generated answer tokens.",
                      "A confirmed answer span does not imply a complete or mathematically correct answer."],
        }
        record = json_safe(record)
        # Capture after decoding and evidence assembly; this final scalar insertion is not separately timed.
        total = (time.perf_counter() - request_start) * 1000
        phase_total = sum(timings.values())
        if phase_total > total + 1e-6:
            raise RuntimeError("Disjoint phase timings exceed total wall clock")
        record.update({"time_total_ms": total, "time_prefill_ms": timings["prefill"],
                       "time_reason_ms": timings["reason"], "time_probe_cache_ms": timings["probe_cache"],
                       "time_answer_ms": timings["answer"], "time_other_ms": max(total - phase_total, 0.0)})
        return record
    except Exception as exc:
        evidence = partial()
        evidence["error_type"] = type(exc).__name__
        evidence["error"] = str(exc)
        raise RequestError(f"{type(exc).__name__}: {exc}", evidence) from exc
