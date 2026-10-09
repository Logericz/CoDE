"""Adaptive CoDE 候选方法：保护条件 + 信号间隔 + 成本间隔。

修改本文件可改变自适应调度；共同停止阈值、D 分数、模型探测另有归属。
本次仅迁移原实现，不修改公式、比较符号或保护条件。
"""

import math

from online_protocol import ACTIVITY_FLOOR
from .common import ScheduleContext, ScheduleResult, common_recovery_reasons


def choose_next(context: ScheduleContext) -> ScheduleResult:
    """返回继续推理后的下一探测间隔及解释字段。先检查保护，再算公式。"""
    observation, previous = context.observation, context.previous
    schedule = context.schedule

    # 1. 保护条件：数据不足、试答未完成、成本异常或置信度下降时逐点探测。
    # 原因顺序也是日志契约的一部分；即使已有原因，仍收集其余适用原因。
    forced = common_recovery_reasons(context)
    if not observation.ended_with_think:
        forced.append("incomplete_probe")
    if context.timing_diagnostics or context.rho is None or not math.isfinite(context.rho):
        forced.append("timing_state_unavailable_or_invalid")
    if previous is None or not observation.confidence_valid:
        forced.append("signal_state_unavailable")
    elif observation.confidence_raw < previous.confidence:
        forced.append("confidence_decline")
    if forced:
        return ScheduleResult(h_next=1, forced_dense_reasons=tuple(forced))

    # 2. 信号间隔：离停止边界越远、信号变化越慢，允许的候选点间隔越大。
    # m 取置信度余量和归一化退化余量的较小值；u 带原有的变化率下限。
    margin = min(max(context.threshold_r - observation.confidence_raw, 0),
                 max(1 - context.score / context.protocol.tau, 0))
    activity = max(abs(observation.confidence_raw - previous.confidence) / context.signal_delta_j,
                   max(context.score - previous.D_observed, 0) / (context.protocol.tau * context.signal_delta_j),
                   ACTIVITY_FLOOR)
    h_signal = min(max(math.floor(margin / activity), 1), schedule.h_max)

    # 3. 成本间隔：rho = 探测耗时 EMA / 每候选间隔推理耗时 EMA。
    # beta 是成本启发式参数，不是已保证的开销上限。保留原有 min 合并规则：
    # 即使 h_signal 很大，只要 h_cost=1，最终仍为逐点探测。
    h_cost = min(max(math.ceil(context.rho / schedule.beta), 1), schedule.h_max)
    h_next = min(h_signal, h_cost, schedule.h_max)
    return ScheduleResult(h_next=h_next, margin_m=margin, activity_u=activity,
                          h_signal=h_signal, h_cost=h_cost)
