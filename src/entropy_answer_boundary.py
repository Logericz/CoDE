"""熵小试 E/T 输出的补充边界提取；不判断正确性，不修改原严格判分。

仅用于已知注入 FINAL_PREFIX 后的 answer span，不能扫描普通 Vanilla 全文。
规则识别字面量 <think> / </think>（大小写与 token 解码一致）；其他推理标记
仍交由原提取器拒绝。返回的 accepted_prefix 只说明结构满足补充规则，不是正确答案。
"""
from __future__ import annotations

import hashlib
import re

if __package__:
    from .math_grading import BOX_START, extract_final_answer
else:
    from math_grading import BOX_START, extract_final_answer

PROTOCOL_VERSION = "entropy-answer-boundary-supplement-v1"
KNOWN_BOUNDARY_POLICY = "known_injected_final_prefix_v1"
THOUGHT_DELIMITER = re.compile(r"</?think>")
BOX_COMMAND = re.compile(r"(?<!\\)\\boxed\b")


def extract_entropy_answer_boundary(answer_text: str, *, answer_boundary_confirmed: bool,
                                    boundary_policy: str) -> dict:
    """返回可审计的字符区间与结构状态；无 gold、阈值或方法效果输入。

    无思考标记时完全保留输入，原判分器继续负责自己的解析规则。存在标记时只
    考察第一次标记之前的前缀；不往标记后搜索更方便的答案，也不拼接或修复括号。
    被拒绝的候选不提供可用 answer_text/span，delimiter 位置仍保留用于复核。
    """
    if not isinstance(answer_text, str):
        raise TypeError("answer_text must be the original string answer span")
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "rejected",
        "reason": None,
        "raw_text_sha256": hashlib.sha256(answer_text.encode("utf-8")).hexdigest(),
        "raw_text_length": len(answer_text),
        "source_boundary_policy": boundary_policy,
        "source_answer_boundary_confirmed": answer_boundary_confirmed is True,
        "answer_text": None,
        "span_start": None,
        "span_end": None,
        "delimiter": None,
        "delimiter_start": None,
        "delimiter_end": None,
        "structure": None,
    }
    if answer_boundary_confirmed is not True:
        return {**result, "reason": "known_answer_boundary_required"}
    if boundary_policy != KNOWN_BOUNDARY_POLICY:
        return {**result, "reason": "unsupported_boundary_policy"}
    marker = THOUGHT_DELIMITER.search(answer_text)
    if marker is None:
        return {**result, "status": "unchanged", "reason": "no_thought_delimiter",
                "answer_text": answer_text, "span_start": 0, "span_end": len(answer_text)}
    result.update(delimiter=marker.group(), delimiter_start=marker.start(), delimiter_end=marker.end())
    prefix = answer_text[:marker.start()]
    if not prefix.strip():
        return {**result, "reason": "empty_prefix_before_thought_delimiter"}
    structure = extract_final_answer(prefix)
    result["structure"] = {name: structure.get(name) for name in (
        "status", "reason", "complete_box_count", "unclosed_box_positions",
        "nested_box_positions", "has_unclosed_tail")}
    # 原提取器有完整box时未必拒绝后续裸 \boxed 命令；本补充规则显式拒绝它。
    opened = {match.start() for match in BOX_START.finditer(prefix)}
    dangling_commands = [match.start() for match in BOX_COMMAND.finditer(prefix)
                         if match.start() not in opened]
    result["structure"]["dangling_boxed_commands"] = dangling_commands
    if dangling_commands or structure.get("unclosed_box_positions"):
        return {**result, "reason": "dangling_boxed_before_thought_delimiter"}
    if structure.get("complete_box_count") != 1:
        return {**result, "reason": "exactly_one_complete_top_level_box_required"}
    if structure.get("status") != "extracted":
        return {**result, "reason": "original_extractor_rejected_prefix"}
    return {**result, "status": "accepted_prefix", "reason": "single_box_before_first_thought_delimiter",
            "answer_text": prefix, "span_start": 0, "span_end": marker.start()}
