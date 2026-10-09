"""Guarded adaptive v1：风险较低时只跳过一个候选点。

输入仅为本次真实探测和过去观测；共同控制器已经检查停止条件。
输出 h_next=1（下一候选就探测）或 2（跳过一个候选再探测）。
valid incomplete probe 可以参与调度，不再因未生成 </think> 一律退回 dense。
这是新的开发候选，不替换旧 adaptive，也不承诺不会错过停止机会。
"""

import math

from online_protocol import ACTIVITY_FLOOR
from .common import ScheduleContext, ScheduleResult, common_recovery_reasons


def choose_next(context: ScheduleContext) -> ScheduleResult:
    """用余量和近期变化选择 1/2 间隔；不读取未来、不改变停止判据。

    margin_m0 是边界保护带。h_max、beta 不参与本候选：最多只跳一个点，
    rho 仅作有效计时检查和日志诊断，不用旧 h_cost 把间隔再次压回 1。
    """
    observation, previous = context.observation, context.previous
    forced = common_recovery_reasons(context)
    if context.timing_diagnostics or context.rho is None or not math.isfinite(context.rho):
        forced.append("timing_state_unavailable_or_invalid")
    if (previous is None or not observation.confidence_valid
            or context.signal_delta_j is None or context.signal_delta_j <= 0
            or context.score is None or not math.isfinite(context.score)):
        forced.append("signal_state_unavailable")
    elif observation.confidence_raw < previous.confidence:
        forced.append("confidence_decline")
    if forced:
        return ScheduleResult(h_next=1, forced_dense_reasons=tuple(forced))

    # m 是离两个停止边界较近的余量；D 先除以 tau，和置信余量保持同量纲。
    # u 是每候选的近期变化幅度。D 在 warm-up 结束时跳升也保留，不抹平信号。
    margin = min(max(context.threshold_r - observation.confidence_raw, 0),
                 max(1 - context.score / context.protocol.tau, 0))
    activity = max(abs(observation.confidence_raw - previous.confidence) / context.signal_delta_j,
                   max(context.score - previous.D_observed, 0)
                   / (context.protocol.tau * context.signal_delta_j),
                   ACTIVITY_FLOOR)

    # 保护带以内立即恢复 dense；若按近期速度，两步内可能进入保护带，也不跳。
    # 这只是局部变化启发式：未来信号可能突变，不能解释成安全或延迟的保证。
    if margin <= context.schedule.margin_m0:
        forced.append("guarded_near_stop_boundary")
    elif 2 * activity >= margin - context.schedule.margin_m0:
        forced.append("guarded_fast_signal_change")
    interval = 1 if forced else 2
    return ScheduleResult(h_next=interval, forced_dense_reasons=tuple(forced),
                          margin_m=margin, activity_u=activity, h_signal=interval)
