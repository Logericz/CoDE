"""Pure protocol arithmetic for the planned common online backend.

This module implements EXPERIMENT_PLAN.md sections 2, 3, 6, and 7 using Python
floats and synthetic or externally supplied observations. It is NOT an online
model runner, and passing its CPU tests establishes neither BF16 numerical
equivalence nor GPU/KV-cache correctness or measured acceleration.

The caller owns token generation, candidate detection, KV/RNG isolation, timing,
and natural/forced finalization. In particular it must resolve the documented
Wait/EOS/think/budget boundaries before submitting observations here. Token
position zero is rejected, not silently shifted to make the score well-defined.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Literal, Sequence


UPSTREAM_COMMIT = "b5081e7c2abe23bb1d19649421cc13522fee7c50"
PROTOCOL_ID = "common-protocol-python-arithmetic-v1"
EPSILON = 1e-12
EMA_ALPHA = 0.2
ACTIVITY_FLOOR = 1e-3
WARMUP_VALID_COUNT = 3
MAX_PROBE_TOKENS = 21
SCHEDULE_RNG_ID = "python.random.Random/MT19937/capped-geometric-inverse-cdf-v1"
RANDOM_P_GRID = (0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.65, 0.80, 1.00)


class ProtocolError(ValueError):
    """An input does not satisfy the frozen arithmetic protocol."""


def _positive_int(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ProtocolError(f"{name} must be a positive integer; got {value!r}")


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _probability(value: object) -> bool:
    return _finite(value) and 0 <= value <= 1


@dataclass(frozen=True)
class ProtocolConfig:
    """Stop-rule constants; grids/selection are managed outside this kernel."""

    rule: Literal["codestop", "deer"] = "codestop"
    r_min: float = 0.90
    r_max: float = 0.95
    ramp_steps: int = 2
    tau: float = 2.0
    deer_threshold: float = 0.95

    def __post_init__(self) -> None:
        if self.rule not in ("codestop", "deer"):
            raise ProtocolError("rule must be codestop or deer")
        if not all(_probability(x) for x in (self.r_min, self.r_max, self.deer_threshold)):
            raise ProtocolError("confidence thresholds must be finite and in [0, 1]")
        if self.r_min > self.r_max:
            raise ProtocolError("r_min must not exceed r_max")
        if type(self.ramp_steps) is not int or self.ramp_steps < 0:
            raise ProtocolError("ramp_steps must be a nonnegative integer")
        if not _finite(self.tau) or self.tau <= 0:
            raise ProtocolError("tau must be finite and positive")


@dataclass(frozen=True)
class ScheduleConfig:
    """The predeclared schedule families and grids in the experiment plan."""

    kind: Literal["dense", "fixed", "log", "random", "backoff", "adaptive"] = "dense"
    fixed_interval: int = 4
    h_max: int = 8
    log_a: float = 1.0
    random_p: float = 0.5
    margin_m0: float = 0.05
    beta: float = 0.5

    def __post_init__(self) -> None:
        if self.kind not in ("dense", "fixed", "log", "random", "backoff", "adaptive"):
            raise ProtocolError("unknown schedule kind")
        if type(self.fixed_interval) is not int or not 1 <= self.fixed_interval <= 9:
            raise ProtocolError("fixed_interval must be in the frozen 1..9 grid")
        if type(self.h_max) is not int or self.h_max not in (2, 4, 8):
            raise ProtocolError("h_max must be 2, 4, or 8")
        if isinstance(self.log_a, bool) or self.log_a not in (0.5, 1.0, 2.0):
            raise ProtocolError("log_a must be in the frozen grid")
        if isinstance(self.random_p, bool) or self.random_p not in RANDOM_P_GRID:
            raise ProtocolError("random_p must be in the frozen grid")
        if isinstance(self.margin_m0, bool) or self.margin_m0 not in (0.02, 0.05, 0.10):
            raise ProtocolError("margin_m0 must be in the frozen grid")
        if isinstance(self.beta, bool) or self.beta not in (0.25, 0.5, 1.0):
            raise ProtocolError("beta must be in the frozen grid")
        if self.kind == "random" and self.h_max != 8:
            raise ProtocolError("the random family has the fixed cap 8")


def legacy_confidence_oracle(token_probs: Sequence[float]) -> float:
    """Python-float oracle for pinned ewt=True arithmetic, NOT a GPU estimator.

For n >= 3 positive probabilities the mathematical expression is
exp(sum(log(p_i), i=2..n-1)/(n-1)), not the usual geometric mean. We retain
the source's accumulation-then-subtraction order; zero final probability may
therefore produce NaN, just as subtracting -inf from -inf does. Consumers must
retain and invalidate nonfinite results. No 1/2-token normalization is invented.
"""
    n = len(token_probs)
    if not 3 <= n <= MAX_PROBE_TOKENS:
        raise ProtocolError("legacy confidence oracle requires 3..21 probe tokens")
    if not all(_probability(p) for p in token_probs):
        raise ProtocolError("probe probabilities must be finite and in [0, 1]")
    logs = [math.log(p) if p > 0 else -math.inf for p in token_probs]
    total = 0.0
    for value in logs[1:]:
        total += value
    return math.exp((total - logs[-1]) / (n - 1))


@dataclass(frozen=True)
class ProbeObservation:
    """Raw probe evidence. Backend confidence must not be replaced by the oracle.

The model backend supplies confidence_raw at its actual precision. To construct
synthetic CPU observations, use from_probabilities(), whose source label makes
its Python-float arithmetic explicit. The caller verifies ending-token identity.
"""

    token_ids: tuple[int, ...]
    token_probs: tuple[float, ...]
    confidence_raw: float | None
    ended_with_think: bool
    confidence_source: str = "backend_supplied"
    invalid_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_ids", tuple(self.token_ids))
        object.__setattr__(self, "token_probs", tuple(self.token_probs))
        if len(self.token_ids) != len(self.token_probs):
            raise ProtocolError("probe token IDs and probabilities must have equal lengths")
        if any(type(x) is not int or x < 0 for x in self.token_ids):
            raise ProtocolError("probe token IDs must be nonnegative integers")
        if len(self.token_ids) > MAX_PROBE_TOKENS:
            raise ProtocolError("probe exceeded the frozen 21-token cap")
        if type(self.ended_with_think) is not bool:
            raise ProtocolError("ended_with_think must be a bool")
        if not isinstance(self.confidence_source, str) or not self.confidence_source:
            raise ProtocolError("confidence_source must be explicit")
        if self.invalid_reason is not None and (not isinstance(self.invalid_reason, str)
                                                or not self.invalid_reason):
            raise ProtocolError("invalid_reason must be nonempty when supplied")

    @classmethod
    def from_probabilities(cls, token_ids: Sequence[int], token_probs: Sequence[float],
                           ended_with_think: bool) -> ProbeObservation:
        """Construct a labelled synthetic/arithmetic observation for CPU checks."""
        try:
            confidence = legacy_confidence_oracle(token_probs)
        except ProtocolError:
            confidence = None
        return cls(tuple(token_ids), tuple(token_probs), confidence, ended_with_think,
                   confidence_source="python_float_arithmetic_oracle")

    @property
    def invalid_reasons(self) -> tuple[str, ...]:
        reasons = []
        if self.invalid_reason:
            reasons.append(self.invalid_reason)
        if len(self.token_ids) <= 2:
            reasons.append("probe_token_count_le_2")
        if not all(_probability(p) for p in self.token_probs):
            reasons.append("invalid_token_probability")
        if not _probability(self.confidence_raw):
            reasons.append("invalid_confidence")
        return tuple(reasons)

    @property
    def confidence_valid(self) -> bool:
        return not self.invalid_reasons


def degeneration_score(token_positions: Sequence[int], confidences: Sequence[float]) -> float:
    """Pinned code-log score including the method's <3-observation zero rule.

All positions are positive, strictly increasing prefix lengths. This function
does not implement the paper-form control or the use_log=False rule.
"""
    if len(token_positions) != len(confidences):
        raise ProtocolError("positions and confidences must have equal lengths")
    previous = 0
    for position in token_positions:
        _positive_int(position, "token_position")
        if position <= previous:
            raise ProtocolError("token positions must be strictly increasing")
        previous = position
    if not all(_probability(c) for c in confidences):
        raise ProtocolError("D accepts only finite valid confidences in [0, 1]")
    if len(confidences) < WARMUP_VALID_COUNT:
        return 0.0
    values = [math.log(max(c, EPSILON)) for c in confidences]
    terminal = token_positions[-1]
    return sum(1.0 + math.log1p(terminal / position - 1.0)
               for previous_value, value, position in
               zip(values, values[1:], token_positions[1:])
               if previous_value > value)


def candidate_threshold(candidate_j: int, config: ProtocolConfig) -> float:
    """One-based candidate clock; never substitute the executed-probe count."""
    _positive_int(candidate_j, "candidate_j")
    if config.rule == "deer":
        return config.deer_threshold
    alpha = min(1.0, (candidate_j - 1) / config.ramp_steps) if config.ramp_steps else 1.0
    return config.r_min + (config.r_max - config.r_min) * alpha


def stopping_reasons(observation: ProbeObservation, score: float | None,
                     threshold_r: float, config: ProtocolConfig) -> tuple[str, ...]:
    """Strict source predicates; incomplete valid probes can stop through D."""
    if not _probability(threshold_r):
        raise ProtocolError("threshold_r must be a finite probability")
    if not observation.confidence_valid:
        return ()
    if not _finite(score) or score < 0:
        raise ProtocolError("a valid observation needs a finite nonnegative D")
    reasons = []
    if observation.ended_with_think and observation.confidence_raw > threshold_r:
        reasons.append("confidence")
    if config.rule == "codestop" and score > config.tau:
        reasons.append("degeneration")
    return tuple(reasons)


def capped_geometric_interval(p: float, rng: random.Random) -> int:
    """Draw min(G, 8), G~Geometric(p) on 1,2,...; not a truncated distribution.

The inverse-CDF draw consumes one random() value unless p=1. This explicit
Python random.Random identity is not a Torch/NumPy schedule RNG identity.
"""
    if not _finite(p) or not 0 < p <= 1:
        raise ProtocolError("geometric p must be in (0, 1]")
    if p == 1:
        return 1
    uniform = rng.random()
    if not _finite(uniform) or not 0 <= uniform < 1:
        raise ProtocolError("schedule RNG must return a value in [0, 1)")
    return min(1 + math.floor(math.log1p(-uniform) / math.log1p(-p)), 8)


@dataclass(frozen=True)
class CostObservation:
    """Measured durations supplied by a caller, never generated by this module.

probe_elapsed_ms includes branch preparation, probe and restoration.
reason_elapsed_ms is main reasoning since the PREVIOUS ACTUAL query, excluding
probe time. With no previous query the denominator interval is undefined; no
fictitious candidate zero is introduced. Missing/invalid timings are retained
as diagnostics and force the adaptive schedule to query densely.
"""

    probe_elapsed_ms: float | None = None
    reason_elapsed_ms: float | None = None


@dataclass(frozen=True)
class ValidObservation:
    candidate_j: int
    token_position: int
    confidence: float
    D_observed: float


@dataclass(frozen=True)
class Decision:
    candidate_j: int
    probe_k: int
    token_position: int
    observation: ProbeObservation
    cost: CostObservation
    threshold_r: float
    threshold_tau: float
    D_observed: float | None
    valid_history_count: int
    stop_reasons: tuple[str, ...]
    signal_delta_j: int | None
    cost_delta_j: int | None
    ema_probe_ms: float | None
    ema_reason_ms_per_candidate: float | None
    rho: float | None
    margin_m: float | None
    activity_u: float | None
    h_signal: int | None
    h_cost: int | None
    h_next: int | None
    next_candidate_j: int | None
    forced_dense_reasons: tuple[str, ...]
    timing_diagnostics: tuple[str, ...]
    schedule_rng_identity: str | None
    seed_schedule: int | None
    stopping_enabled: bool = True

    @property
    def should_stop(self) -> bool:
        return self.stopping_enabled and bool(self.stop_reasons)

    @property
    def would_stop(self) -> bool:
        """Predicate outcome, including a diagnostic that deliberately continues."""
        return bool(self.stop_reasons)


class ProtocolController:
    """Consume actual query observations and propose the next candidate index.

    Every family queries from candidate 1 until three valid observations exist.
    Invalid probes force the next candidate to be queried. Valid incomplete
    probes force dense recovery only for adaptive/backoff; fixed/log/random keep
    their preset cadence after warm-up (section 7.2). Signal decline/near-boundary
    and cost rules apply only to their specified families.
Natural termination may prevent a scheduled query; do not manufacture a terminal
probe. Calls after a stopping decision or at an unscheduled candidate are errors.
"""

    def __init__(self, config: ProtocolConfig = ProtocolConfig(),
                 schedule: ScheduleConfig = ScheduleConfig(), *,
                 seed_schedule: int | None = None, stop_enabled: bool = True):
        if type(stop_enabled) is not bool:
            raise ProtocolError("stop_enabled must be boolean")
        if not stop_enabled and schedule.kind != "dense":
            raise ProtocolError("disabled stopping is restricted to dense diagnostic collection")
        self.stop_enabled = stop_enabled
        if schedule.kind == "random" and (type(seed_schedule) is not int or seed_schedule < 0):
            raise ProtocolError("the random schedule requires an explicit nonnegative seed")
        if schedule.kind != "random" and seed_schedule is not None:
            raise ProtocolError("seed_schedule applies only to the random family")
        self.config, self.schedule = config, schedule
        self.seed_schedule = seed_schedule
        self._rng = random.Random(seed_schedule) if schedule.kind == "random" else None
        self._history: list[ValidObservation] = []
        self._last_j: int | None = None
        self._last_token_position: int | None = None
        self._probe_count = 0
        self._next_j: int | None = 1
        self._backoff_interval = 1
        self._ema_probe: float | None = None
        self._ema_reason: float | None = None

    @property
    def history(self) -> tuple[ValidObservation, ...]:
        return tuple(self._history)

    @property
    def next_candidate_j(self) -> int | None:
        return self._next_j

    @property
    def schedule_rng_identity(self) -> str | None:
        return SCHEDULE_RNG_ID if self._rng is not None else None

    def should_probe(self, candidate_j: int) -> bool:
        _positive_int(candidate_j, "candidate_j")
        if self._next_j is None:
            raise ProtocolError("controller already stopped")
        if self._last_j is not None and candidate_j <= self._last_j:
            raise ProtocolError("candidate inspection must advance beyond the last query")
        if candidate_j > self._next_j:
            raise ProtocolError("a scheduled query was missed; cannot silently move it")
        return candidate_j == self._next_j

    @staticmethod
    def _ema(previous: float | None, value: float) -> float:
        return value if previous is None else (1 - EMA_ALPHA) * previous + EMA_ALPHA * value

    def _update_cost(self, cost: CostObservation, delta_j: int | None) -> tuple[str, ...]:
        diagnostics = []
        if not _finite(cost.probe_elapsed_ms) or cost.probe_elapsed_ms < 0:
            diagnostics.append("missing_or_invalid_probe_time")
        else:
            self._ema_probe = self._ema(self._ema_probe, cost.probe_elapsed_ms)
        if delta_j is None:
            diagnostics.append("no_previous_actual_query_for_reason_interval")
        elif not _finite(cost.reason_elapsed_ms) or cost.reason_elapsed_ms <= 0:
            diagnostics.append("missing_or_nonpositive_reason_time")
        else:
            self._ema_reason = self._ema(self._ema_reason, cost.reason_elapsed_ms / delta_j)
        return tuple(diagnostics)

    def observe(self, candidate_j: int, token_position: int,
                observation: ProbeObservation, cost: CostObservation = CostObservation()) -> Decision:
        """Update from one executed query. Invalid confidence still incurs cost.

For an invalid observation D_observed is None, not a newly computed D nor a
stale D advertised as a current measurement. The previous valid score remains
in history for the next valid signal difference.
"""
        _positive_int(token_position, "token_position")
        if self._last_token_position is not None and token_position <= self._last_token_position:
            raise ProtocolError("queried token positions must be strictly increasing")
        if not self.should_probe(candidate_j):
            raise ProtocolError("probe submitted at an unscheduled candidate")
        previous = self._history[-1] if self._history else None
        signal_delta = candidate_j - previous.candidate_j if previous is not None else None
        cost_delta = candidate_j - self._last_j if self._last_j is not None else None
        timing_diagnostics = self._update_cost(cost, cost_delta)
        self._last_j, self._last_token_position = candidate_j, token_position
        self._probe_count += 1
        threshold = candidate_threshold(candidate_j, self.config)
        score = None
        if observation.confidence_valid:
            score = degeneration_score(
                [entry.token_position for entry in self._history] + [token_position],
                [entry.confidence for entry in self._history] + [observation.confidence_raw])
            self._history.append(ValidObservation(candidate_j, token_position,
                                                  observation.confidence_raw, score))
        reasons = stopping_reasons(observation, score, threshold, self.config)
        rho = (self._ema_probe / self._ema_reason
               if self._ema_probe is not None and self._ema_reason is not None
               and self._ema_reason > 0 else None)
        forced: list[str] = []
        margin = activity = h_signal = h_cost = h_next = None
        if reasons and self.stop_enabled:
            self._next_j = None
        else:
            if not observation.confidence_valid:
                forced.append("invalid_probe")
            if len(self._history) < WARMUP_VALID_COUNT:
                forced.append("fewer_than_three_valid_observations")
            if (not observation.ended_with_think
                    and self.schedule.kind in ("adaptive", "backoff")):
                forced.append("incomplete_probe")
            if self.schedule.kind == "adaptive":
                if timing_diagnostics or rho is None or not math.isfinite(rho):
                    forced.append("timing_state_unavailable_or_invalid")
                if previous is None or not observation.confidence_valid:
                    forced.append("signal_state_unavailable")
                elif observation.confidence_raw < previous.confidence:
                    forced.append("confidence_decline")
            if forced:
                h_next = 1
            elif self.schedule.kind == "dense":
                h_next = 1
            elif self.schedule.kind == "fixed":
                h_next = self.schedule.fixed_interval
            elif self.schedule.kind == "log":
                h_next = min(max(math.ceil(self.schedule.log_a * math.log1p(candidate_j)), 1),
                             self.schedule.h_max)
            elif self.schedule.kind == "random":
                h_next = capped_geometric_interval(self.schedule.random_p, self._rng)
            elif self.schedule.kind == "backoff":
                log_decline = math.log(max(observation.confidence_raw, EPSILON)) < math.log(
                    max(previous.confidence, EPSILON))
                if observation.confidence_raw >= threshold - self.schedule.margin_m0 or log_decline:
                    h_next = 1
                    forced.append("backoff_near_boundary_or_log_decline")
                else:
                    h_next = min(2 * self._backoff_interval, self.schedule.h_max)
            else:
                margin = min(max(threshold - observation.confidence_raw, 0),
                             max(1 - score / self.config.tau, 0))
                activity = max(abs(observation.confidence_raw - previous.confidence) / signal_delta,
                               max(score - previous.D_observed, 0) / (self.config.tau * signal_delta),
                               ACTIVITY_FLOOR)
                h_signal = min(max(math.floor(margin / activity), 1), self.schedule.h_max)
                h_cost = min(max(math.ceil(rho / self.schedule.beta), 1), self.schedule.h_max)
                h_next = min(h_signal, h_cost, self.schedule.h_max)
            self._backoff_interval = h_next
            self._next_j = candidate_j + h_next
        return Decision(candidate_j, self._probe_count, token_position, observation, cost,
                        threshold, self.config.tau, score, len(self._history), reasons,
                        signal_delta, cost_delta, self._ema_probe, self._ema_reason, rho,
                        margin, activity, h_signal, h_cost, h_next, self._next_j,
                        tuple(forced), timing_diagnostics, self.schedule_rng_identity,
                        self.seed_schedule, self.stop_enabled)
