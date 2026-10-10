"""固定小特征集合：只压缩已保存观测，不拟合、不重算D、不读取答案。

输入是原 prepare_records 产出的 prepared_row，包含 B/H/long_H 三个视图。
f 在付费追加之前预测观测价值；g_long 只能在追加完成后读取长试答信息。
"""
from __future__ import annotations

import math


FEATURE_SPEC = {
    "f_B": ["prefix_tokens", "D_short", "short_confidence", "confidence_change",
            "short_prob_slope"],
    "f_H": ["prefix_tokens", "D_short", "short_confidence", "confidence_change",
            "short_prob_slope", "short_entropy_mean", "short_entropy_slope"],
    "g_short": ["prefix_tokens", "D_short", "short_confidence", "confidence_change",
                "short_prob_slope", "short_entropy_mean", "short_entropy_slope",
                "short_probe_tokens", "short_ended_with_think"],
    "g_long": ["prefix_tokens", "D_short", "short_confidence", "confidence_change",
               "short_prob_slope", "short_entropy_mean", "short_entropy_slope",
               "short_probe_tokens", "short_ended_with_think", "long_confidence",
               "long_prob_slope", "long_entropy_mean", "long_entropy_slope",
               "long_probe_tokens", "long_ended_with_think"],
}
FEATURE_DIMENSIONS = {name: len(features) for name, features in FEATURE_SPEC.items()}
FEATURE_METADATA = {
    "protocol": "entropy-compact-features-v1",
    "dimensions": FEATURE_DIMENSIONS,
    "views": FEATURE_SPEC,
    "confidence_change": "B.short_confidence - B.history_previous_confidence; missing/nonfinite previous -> NaN",
    "slope": "Reuse the existing curve_summary slope from the original prepared feature views",
    "D": "Copy B.D_short unchanged; long probe never updates D",
    "missing_values": "Preserve NaN for the original fold-local imputation; do not fill from held-out data",
}


def compact_views(prepared_row: dict) -> dict[str, dict[str, float]]:
    """返回预先固定的5/7/9/15维视图；只访问明确列出的观测字段。

    短试答长度与结束标志在可追加的f样本中是协议常数，故不进入f。
    g仍用包含终止点的完整配对集训练，必须保留这两个字段。长g完整复制短g，
    再追加长观测，避免把“得到新信息”误做成“丢掉旧信息再另拟合”。
    """
    base, entropy, extended = (prepared_row[name] for name in ("B", "H", "long_H"))
    previous = base.get("history_previous_confidence")
    # 缺少过去观测不是零变化；保留NaN，由原训练折内预处理负责填补。
    change = (base["short_confidence"] - previous
              if type(previous) in (int, float) and math.isfinite(previous)
              else float("nan"))
    f_base = {name: change if name == "confidence_change" else base[name]
              for name in FEATURE_SPEC["f_B"]}
    f_entropy = {**f_base, "short_entropy_mean": entropy["short_entropy_mean"],
                 "short_entropy_slope": entropy["short_entropy_slope"]}
    short = {**f_entropy, "short_probe_tokens": base["short_probe_tokens"],
             "short_ended_with_think": base["short_ended_with_think"]}
    long = {**short, **{name: extended[name] for name in FEATURE_SPEC["g_long"]
                       if name.startswith("long_")}}
    return {"f_B": f_base, "f_H": f_entropy, "g_short": short, "g_long": long}
