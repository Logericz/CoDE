"""预声明的 backoff 对照：稳定时倍增间隔，接近边界或退化时恢复探测。"""

import math

from online_protocol import EPSILON
from .common import ScheduleContext, ScheduleResult, common_recovery_reasons


def choose_next(context: ScheduleContext) -> ScheduleResult:
    forced = common_recovery_reasons(context)
    if not context.observation.ended_with_think:
        forced.append("incomplete_probe")
    if forced:
        return ScheduleResult(h_next=1, forced_dense_reasons=tuple(forced))
    confidence = context.observation.confidence_raw
    log_decline = math.log(max(confidence, EPSILON)) < math.log(
        max(context.previous.confidence, EPSILON))
    if confidence >= context.threshold_r - context.schedule.margin_m0 or log_decline:
        return ScheduleResult(h_next=1, forced_dense_reasons=("backoff_near_boundary_or_log_decline",))
    return ScheduleResult(h_next=min(2 * context.previous_interval, context.schedule.h_max))
