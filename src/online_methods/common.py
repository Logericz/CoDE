"""调度器的共享输入、输出和预热规则；不生成 token，也不判断早停。"""

from __future__ import annotations

from dataclasses import dataclass
import random

from online_protocol import (
    WARMUP_VALID_COUNT, ProbeObservation, ProtocolConfig, ScheduleConfig,
    ValidObservation,
)


@dataclass(frozen=True)
class ScheduleContext:
    """本次真实探测后可见的状态。跳过的候选点没有虚构的置信度。

    previous 是上一次有效观测；timing_diagnostics/rho 则来自实际探测成本。
    两条历史在遇到无效探测时可能不同，由共享控制器分别维护。
    """

    candidate_j: int
    observation: ProbeObservation
    previous: ValidObservation | None
    valid_history_count: int
    threshold_r: float
    score: float | None
    signal_delta_j: int | None
    rho: float | None
    timing_diagnostics: tuple[str, ...]
    protocol: ProtocolConfig
    schedule: ScheduleConfig
    previous_interval: int
    rng: random.Random | None


@dataclass(frozen=True)
class ScheduleResult:
    """h_next 是候选点间隔：1 表示下一点探测，4 表示跳过中间 3 点。"""

    h_next: int
    forced_dense_reasons: tuple[str, ...] = ()
    margin_m: float | None = None
    activity_u: float | None = None
    h_signal: int | None = None
    h_cost: int | None = None


def common_recovery_reasons(context: ScheduleContext) -> list[str]:
    """所有调度器共用：无效探测、未满 3 次有效观测时保持逐点探测。

    停止判断先于本函数，所以高置信度仍可在第 1 次探测时提前停止。
    """
    reasons = []
    if not context.observation.confidence_valid:
        reasons.append("invalid_probe")
    if context.valid_history_count < WARMUP_VALID_COUNT:
        reasons.append("fewer_than_three_valid_observations")
    return reasons
