"""各方法的调度实现。阅读入口见同目录 README.md。

生成和早停判断由共享引擎处理；这里仅选择“下次在哪个候选点探测”。
"""

from collections.abc import Callable

from . import adaptive, backoff, dense, fixed, guarded, logarithmic, random_schedule
from .common import ScheduleContext, ScheduleResult


def scheduler_for(kind: str) -> Callable[[ScheduleContext], ScheduleResult]:
    """在创建控制器时选择方法，避免每次观测都穿过多方法分支。"""
    return {
        "dense": dense.choose_next,
        "fixed": fixed.choose_next,
        "adaptive": adaptive.choose_next,
        "guarded": guarded.choose_next,
        "log": logarithmic.choose_next,
        "random": random_schedule.choose_next,
        "backoff": backoff.choose_next,
    }[kind]
