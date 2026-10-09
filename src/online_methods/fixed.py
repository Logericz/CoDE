"""Fixed CoDE：预热后采用固定候选点间隔。"""

from .common import ScheduleContext, ScheduleResult, common_recovery_reasons


def choose_next(context: ScheduleContext) -> ScheduleResult:
    """默认 h=4：有效预热后候选点依次为 1, 2, 3, 7, 11, ...。

    无效探测仍回退到 h=1。有效但未完成的试答在预热后保持固定节奏；
    这与 Adaptive 的 incomplete_probe 保护不同，是已有协议的一部分。
    """
    forced = common_recovery_reasons(context)
    if forced:
        return ScheduleResult(h_next=1, forced_dense_reasons=tuple(forced))
    return ScheduleResult(h_next=context.schedule.fixed_interval)
