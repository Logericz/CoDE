"""Dense CoDE：每个候选点都探测；停止条件仍由共同协议判断。"""

from .common import ScheduleContext, ScheduleResult, common_recovery_reasons


def choose_next(context: ScheduleContext) -> ScheduleResult:
    """继续时固定 h=1。保留预热/无效观测原因，供旧日志逐字段对照。"""
    return ScheduleResult(h_next=1, forced_dense_reasons=tuple(common_recovery_reasons(context)))
