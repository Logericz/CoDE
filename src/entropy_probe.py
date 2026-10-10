"""独立的短/长试答观测实验，不修改冻结的 CoDE probe 或停止协议。

21→42 沿同一试答 KV 继续；熵使用独立 float32 全词表分布，置信度仍按
原输出 dtype 与累加顺序计算。无 EOS 的前 21 tokens 可与旧 probe 精确核对。
本实验遇 EOS 立即结束并标无效；旧 probe 遇 EOS 继续到 think/cap，二者有意不同。
调用者负责包住 start/extend/close 的同步计时，包含分支克隆及释放成本。
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any

MAX_ENTROPY_PROBE_TOKENS = 42
ENTROPY_PROBE_PROTOCOL = "entropy-observation-paired-v1"


@dataclass
class EntropyProbeSession:
    """只属于一个请求的试答分支；不可把 branch 赋回主推理状态。

    pending_token 已采样但尚未进入 KV。恢复时先接受它，再生成下一个 token。
    total/last_maximum 保留 torch 标量，不能用导出的 Python 概率重建 BF16 累加。
    """

    backend: Any = field(repr=False)
    branch: Any = field(repr=False)
    max_tokens: int
    token_ids: list[int] = field(default_factory=list)
    token_probs: list[float] = field(default_factory=list)
    entropies: list[float] = field(default_factory=list)
    pending_token: int | None = None
    total: Any = field(default=0.0, repr=False)
    last_maximum: Any = field(default=None, repr=False)
    terminal_reason: str | None = None
    invalid_reason: str | None = None
    logits_dtype: str | None = None
    closed: bool = False


def _token_limit(value: int, name: str, maximum: int) -> None:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in 1..{maximum}")


def clone_state(backend, state):
    """复制主 KV/logits，供配对终点的统一 finalizer 使用；不推进 RNG。"""
    backend._validate_state(state)
    with backend.torch.no_grad():
        return replace(state, cache=backend._clone_cache(state.cache), logits=state.logits.clone())


def start_probe(backend, state, max_tokens: int = MAX_ENTROPY_PROBE_TOKENS) -> EntropyProbeSession:
    """克隆主分支并加入同一 trial prefix；此时尚未生成第一个试答 token。"""
    _token_limit(max_tokens, "max_tokens", MAX_ENTROPY_PROBE_TOKENS)
    backend._validate_state(state)
    if state.length + len(backend.markers.trial_prefix) + max_tokens > backend.context_limit:
        raise ValueError("Insufficient context reserve for the entire entropy probe")
    with backend.torch.no_grad():
        branch = backend._forward(backend._clone_cache(state.cache),
                                  backend.markers.trial_prefix, keep_last_only=False)
    return EntropyProbeSession(backend, branch, max_tokens, logits_dtype=str(branch.logits.dtype))


def _nonfinite_kind(value: float) -> str | None:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    return None


def _snapshot(session: EntropyProbeSession) -> dict[str, Any]:
    n = len(session.token_ids)
    confidence = None
    if n:
        torch = session.backend.torch
        # 原公式跳过首项、累加其余项后减去末项，再除 n-1；不换成标准几何平均。
        with torch.no_grad():
            confidence = float(torch.exp((session.total - torch.log(session.last_maximum)) / (n - 1)).item())
    reasons = [session.invalid_reason] if session.invalid_reason else []
    if n <= 2:
        reasons.append("probe_token_count_le_2")
    if confidence is None or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        reasons.append("invalid_confidence")
    entropy_valid = bool(n and len(session.entropies) == n
                         and all(math.isfinite(x) and x >= 0 for x in session.entropies))
    if not entropy_valid:
        reasons.append("invalid_entropy")
    reasons = list(dict.fromkeys(reasons))
    # JSON 禁止 NaN；以 null 加明确诊断保留异常，而非悄悄替成有利的数值。
    return {
        "protocol": ENTROPY_PROBE_PROTOCOL,
        "token_ids": list(session.token_ids),
        "token_probs": [p if math.isfinite(p) else None for p in session.token_probs],
        "confidence_raw": confidence if confidence is not None and math.isfinite(confidence) else None,
        "confidence_nonfinite": _nonfinite_kind(confidence) if confidence is not None else None,
        "confidence_source": f"torch_model_logits_softmax/{session.logits_dtype}",
        "confidence_valid": n > 2 and not session.invalid_reason
                            and confidence is not None and math.isfinite(confidence) and 0 <= confidence <= 1,
        "entropy_raw_fp32_nats": [h if math.isfinite(h) else None for h in session.entropies],
        "entropy_valid": entropy_valid,
        "entropy_source": "unwarped_full_vocabulary_logits_float32_natural_log",
        "ended_with_think": bool(n and session.token_ids[-1] == session.backend.markers.end_think),
        "ended_with_eos": bool(n and session.token_ids[-1] in session.backend.markers.eos_ids),
        "invalid_reason": reasons[0] if reasons else None,
        "invalid_reasons": reasons,
        "actual_length": n,
        "max_tokens": session.max_tokens,
        "termination_reason": session.terminal_reason or "target_reached",
        "can_extend": session.terminal_reason is None and n < session.max_tokens,
    }


def extend_probe(session: EntropyProbeSession, target_tokens: int) -> dict[str, Any]:
    """延长到累计 target_tokens；同目标重复调用不产生新工作，终止后不再追加。"""
    if not isinstance(session, EntropyProbeSession) or session.closed:
        raise ValueError("An open EntropyProbeSession is required")
    _token_limit(target_tokens, "target_tokens", session.max_tokens)
    if target_tokens < len(session.token_ids):
        raise ValueError("target_tokens cannot move backwards")
    backend, torch = session.backend, session.backend.torch
    backend._validate_state(session.branch)
    with torch.no_grad():
        while len(session.token_ids) < target_tokens and session.terminal_reason is None:
            if session.pending_token is not None:
                session.branch = backend._forward(session.branch.cache, (session.pending_token,),
                                                  keep_last_only=False)
                session.pending_token = None
            logits = session.branch.logits
            if not bool(torch.isfinite(logits).all().item()):
                session.invalid_reason = session.terminal_reason = "nonfinite_probe_logits"
                break
            # CoDE 原 BF16 路径独立保留；熵不覆盖 logits/probabilities/累加器。
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            maximum, token_tensor = torch.max(probabilities, dim=0)
            if session.token_ids:
                session.total += torch.log(maximum)
            token = int(token_tensor.item())
            session.token_ids.append(token)
            session.token_probs.append(float(maximum.item()))
            session.last_maximum = maximum
            session.pending_token = token
            logp = torch.nn.functional.log_softmax(logits.float(), dim=-1)
            p = logp.exp()
            # 0*log(0) 按零处理，避免极小概率下溢造成伪 NaN。
            entropy = float((-torch.where(p > 0, p * logp, torch.zeros_like(p)).sum()).item())
            session.entropies.append(entropy)
            if not math.isfinite(entropy) or entropy < 0:
                session.invalid_reason = session.terminal_reason = "nonfinite_or_negative_entropy"
            elif token in backend.markers.eos_ids:
                session.invalid_reason = "probe_generated_eos"
                session.terminal_reason = "eos"
            elif token == backend.markers.end_think:
                session.terminal_reason = "end_think"
            elif len(session.token_ids) == session.max_tokens:
                session.terminal_reason = "max_tokens"
    return _snapshot(session)


def close_probe(session: EntropyProbeSession) -> None:
    """释放分支GPU张量；不清理全局CUDA缓存，也不影响主KV。可重复调用。"""
    if not isinstance(session, EntropyProbeSession):
        raise ValueError("EntropyProbeSession is required")
    session.branch = session.total = session.last_maximum = None
    session.pending_token = None
    session.closed = True
