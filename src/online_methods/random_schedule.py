"""预声明的 random 对照：使用独立调度 RNG，不能消耗主生成随机流。"""

from online_protocol import capped_geometric_interval
from .common import ScheduleContext, ScheduleResult, common_recovery_reasons


def choose_next(context: ScheduleContext) -> ScheduleResult:
    forced = common_recovery_reasons(context)
    if forced:
        return ScheduleResult(h_next=1, forced_dense_reasons=tuple(forced))
    return ScheduleResult(h_next=capped_geometric_interval(context.schedule.random_p, context.rng))
