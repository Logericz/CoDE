"""短观测融合的固定特征：共同上下文Z、置信信号C、熵信号H及二者CH。

只接受原 prepare_records 已算好的 B/H 字段，不读取长观测、答案、标签或成本。
H不含D与置信度变化，因为这两项本身就是置信信号，不能冒充共同上下文。
"""
from __future__ import annotations

import math


FEATURE_SPEC = {
    "Z": ["prefix_tokens", "short_probe_tokens", "short_ended_with_think"],
    "C": ["prefix_tokens", "short_probe_tokens", "short_ended_with_think",
          "short_confidence", "confidence_change", "D_short", "short_prob_slope"],
    "H": ["prefix_tokens", "short_probe_tokens", "short_ended_with_think",
          "short_entropy_mean", "short_entropy_slope"],
    "CH": ["prefix_tokens", "short_probe_tokens", "short_ended_with_think",
           "short_confidence", "confidence_change", "D_short", "short_prob_slope",
           "short_entropy_mean", "short_entropy_slope"],
}
FEATURE_DIMENSIONS = {name: len(fields) for name, fields in FEATURE_SPEC.items()}
DIMENSIONS = FEATURE_DIMENSIONS
FEATURE_METADATA = {
    "protocol": "entropy-short-features-v1",
    "dimensions": FEATURE_DIMENSIONS,
    "views": FEATURE_SPEC,
    "availability": "All fields are available after the short probe, before any extension",
    "confidence_change": "B.short_confidence - B.history_previous_confidence; missing/nonfinite previous -> NaN",
    "D_and_slopes": "Copy saved B.D_short, B.short_prob_slope and H.short_entropy_slope unchanged",
    "missing_values": "Preserve NaN for the original training-fold-only imputation",
    "compatibility": "CH field values equal compact_views(prepared_row)['g_short']; field order is not the matrix order",
}


def feature_views(prepared_row: dict) -> dict[str, dict[str, float]]:
    """返回3/7/5/9维独立字典；矩阵列仍由调用方按字段名排序。

    不调用旧 compact_views：它会同时构造long视图，要求调用者已有未来观测。
    这里显式挑选B/H字段，连不存在long_H的短观测记录也可以使用。
    """
    base, entropy = prepared_row["B"], prepared_row["H"]
    shared = {name: base[name] for name in FEATURE_SPEC["Z"]}
    previous = base.get("history_previous_confidence")
    # 第一条历史缺失时，变化未知而不是零；训练折中的缺失处理保持不变。
    change = (base["short_confidence"] - previous
              if type(previous) in (int, float) and math.isfinite(previous)
              else float("nan"))
    confidence = {**shared, "short_confidence": base["short_confidence"],
                  "confidence_change": change, "D_short": base["D_short"],
                  "short_prob_slope": base["short_prob_slope"]}
    entropy_only = {**shared, "short_entropy_mean": entropy["short_entropy_mean"],
                    "short_entropy_slope": entropy["short_entropy_slope"]}
    combined = {**confidence, "short_entropy_mean": entropy_only["short_entropy_mean"],
                "short_entropy_slope": entropy_only["short_entropy_slope"]}
    return {"Z": shared, "C": confidence, "H": entropy_only, "CH": combined}
