"""预声明的 log 对照：按候选点序号的对数增加间隔。"""

import math

from .common import ScheduleContext, ScheduleResult, common_recovery_reasons


def choose_next(context: ScheduleContext) -> ScheduleResult:
    forced = common_recovery_reasons(context)
    if forced:
        return ScheduleResult(h_next=1, forced_dense_reasons=tuple(forced))
    interval = min(max(math.ceil(context.schedule.log_a * math.log1p(context.candidate_j)), 1),
                   context.schedule.h_max)
    return ScheduleResult(h_next=interval)
