"""Vanilla：正常生成，不创建探测或早停控制器。

自然 EOS、思考结束、预算耗尽及最终补答由 online_engine.run_request
统一处理。Vanilla 也使用同一采样和 token 循环，便于公平比较。
"""

from __future__ import annotations

from online_protocol import ScheduleConfig


def create_controller(schedule: ScheduleConfig, seed_schedule: int | None) -> None:
    """检查 Vanilla 配置并返回 None；引擎遇到候选点时因此不会调用 probe。

    dense 在这里是公共配置的占位值，并不表示 Vanilla 会逐点探测。
    """
    if schedule.kind != "dense":
        raise ValueError("vanilla requires an explicit dense schedule")
    if seed_schedule is not None:
        raise ValueError("vanilla has no schedule RNG")
    return None
