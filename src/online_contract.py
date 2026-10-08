"""Shared interfaces for an explicitly versioned common online runner.

No model is loaded here. The production backend requires CUDA/BF16; independent
CPU fixtures must identify themselves as synthetic, never as GPU validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from online_protocol import ProbeObservation

MODEL_ID = "Qwen/Qwen3-4B"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
RUNNER_PROTOCOL = "common-online-validation-v1"
TRIAL_PREFIX = "\n**Final Answer**\n\nThe final answer is \\boxed"
FINAL_PREFIX = "</think>" + TRIAL_PREFIX
TASK_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


@dataclass(frozen=True)
class TokenMarkers:
    wait: int
    end_think: int
    eos: int
    trial_prefix: tuple[int, ...]
    final_prefix: tuple[int, ...]

    def __post_init__(self):
        for value in (self.wait, self.end_think, self.eos):
            if type(value) is not int or value < 0:
                raise ValueError("markers must be nonnegative integer token IDs")
        if len({self.wait, self.end_think, self.eos}) != 3:
            raise ValueError("Wait, end-think, and EOS must have distinct token IDs")
        for name in ("trial_prefix", "final_prefix"):
            ids = tuple(getattr(self, name))
            if not ids or any(type(x) is not int or x < 0 for x in ids):
                raise ValueError(f"{name} requires nonempty token IDs")
            object.__setattr__(self, name, ids)
        if self.final_prefix[0] != self.end_think:
            raise ValueError("controlled final prefix must begin with the end-think token")


@dataclass(frozen=True)
class TokenSample:
    token_id: int
    sample_probability: float
    raw_probability: float


class OnlineBackend(Protocol):
    """All state changes are explicit; sampling alone never extends main KV.

    start_request creates a private main-sampling generator and resets peak
    memory; it must not change the global RNG. probe clones its branch, preserves
    state/logits/main RNG, and returns probabilities at actual model precision.
    extend accepts existing tokens; it must never resample a pending Wait.
    synchronize supplies a boundary for inclusive, disjoint wall-clock timings.
    """

    markers: TokenMarkers
    context_limit: int
    metadata: dict[str, Any]

    def prompt_ids(self, question: str) -> list[int]: ...
    def start_request(self, seed_reason: int) -> None: ...
    def prefill(self, prompt_ids: Sequence[int]) -> Any: ...
    def sample(self, state: Any) -> TokenSample: ...
    def greedy(self, state: Any) -> TokenSample: ...
    def extend(self, state: Any, token_ids: Sequence[int]) -> Any: ...
    def probe(self, state: Any) -> ProbeObservation: ...
    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = True) -> str: ...
    def synchronize(self) -> None: ...
    def peak_memory_bytes(self) -> int | None: ...
