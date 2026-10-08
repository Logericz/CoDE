"""Torch execution primitives for the common online validation runner.

Production loads an existing, pinned Qwen3-4B snapshot on one CUDA GPU in BF16.
The explicit from_test_model factory is for synthetic CPU fixtures only. CPU
checks of actual Transformers caches are not GPU validation or research results.
No weights are downloaded here; no SSH, global seeding, or empty_cache is used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Sequence

from online_contract import (
    FINAL_PREFIX, MODEL_ID, MODEL_REVISION, RUNNER_PROTOCOL, TASK_SUFFIX,
    TRIAL_PREFIX, TokenMarkers, TokenSample,
)
from online_protocol import MAX_PROBE_TOKENS, ProbeObservation


@dataclass(frozen=True)
class TorchState:
    """KV covers exactly length accepted tokens; logits predict the next token.

extend consumes this state: the returned state must replace it because the
DynamicCache is updated in place. probe instead owns an independent cloned KV.
"""

    cache: Any
    logits: Any
    length: int
    request_id: int
    _owner: object = field(repr=False)


def _runtime():
    import torch
    import transformers
    if torch.__version__.split("+")[0] != "2.9.1":
        raise RuntimeError("This backend requires torch==2.9.1")
    if transformers.__version__ != "4.51.3":
        raise RuntimeError("This backend requires transformers==4.51.3")
    return torch, transformers


class TorchOnlineBackend:
    """Single-request backend implementing online_contract.OnlineBackend.

The enclosing runner detects candidate/end boundaries and measures inclusive
wall-clock time. All cache clones and probe forwards occur inside probe().
Sampling uses float32 logits and HF Temperature -> TopK -> TopP -> MinP(0),
followed by multinomial with a private device generator. Probe softmax and
confidence accumulation retain the model's actual output dtype.
"""

    def __init__(self, model_dir: Path, *, attention_implementation: str = "eager"):
        torch, transformers = _runtime()
        if attention_implementation not in ("eager", "sdpa"):
            raise ValueError("attention_implementation must be eager or sdpa")
        if not torch.cuda.is_available():
            raise RuntimeError("Production backend requires CUDA; use no CPU fallback")
        torch.cuda.set_device(0)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("cuda:0 does not support BF16")
        model_dir = Path(model_dir).expanduser().resolve()
        if not model_dir.is_dir() or model_dir.name != MODEL_REVISION:
            raise ValueError("model_dir must be the existing pinned HF snapshot directory")
        required = ("config.json", "generation_config.json", "tokenizer.json",
                    "tokenizer_config.json", "model.safetensors.index.json")
        for name in required:
            if not (model_dir / name).is_file():
                raise ValueError(f"Incomplete local snapshot: {name}")
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        if config.get("model_type") != "qwen3" or config.get("quantization_config"):
            raise ValueError("Expected the unquantized pinned Qwen3 causal model")
        index = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
        shards = sorted(set(index.get("weight_map", {}).values()))
        if not shards:
            raise ValueError("The local weight index is empty")
        for shard in shards:
            if Path(shard).is_absolute() or ".." in Path(shard).parts or not (model_dir / shard).is_file():
                raise ValueError("Invalid or missing local weight shard")
        torch.cuda.synchronize(0)
        start = time.perf_counter()
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16, attn_implementation=attention_implementation).to("cuda:0").eval()
        self._initialize(model, tokenizer, attention_implementation, synthetic=False)
        if any(parameter.dtype != torch.bfloat16 for parameter in model.parameters()):
            raise RuntimeError("All production model parameters must be BF16")
        reserve = max(len(self.markers.final_prefix) + 30, len(self.markers.trial_prefix) + MAX_PROBE_TOKENS)
        if self.context_limit < 32768 + reserve + 1:
            raise ValueError("Context capacity cannot accommodate the main budget and final/probe reserve")
        torch.cuda.synchronize(0)
        self.metadata.update({
            "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "tokenizer_revision": MODEL_REVISION,
            "snapshot_name": model_dir.name,
            "snapshot_config_sha256": {name: hashlib.sha256((model_dir / name).read_bytes()).hexdigest()
                                       for name in required if name != "tokenizer.json"},
            "tokenizer_sha256": hashlib.sha256((model_dir / "tokenizer.json").read_bytes()).hexdigest(),
            "weight_shards": {name: (model_dir / name).stat().st_size for name in shards},
            "model_load_seconds": time.perf_counter() - start,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        })
        try:
            driver = subprocess.run(
                ["nvidia-smi", "--id=0", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True, capture_output=True, timeout=5, check=True).stdout.strip()
            self.metadata["driver_version"] = driver or None
        except (OSError, subprocess.SubprocessError) as exc:
            self.metadata["driver_version"] = None
            self.metadata["driver_inventory_error"] = type(exc).__name__

    @classmethod
    def from_test_model(cls, model: Any, tokenizer: Any, *,
                        attention_implementation: str = "eager") -> TorchOnlineBackend:
        """Inject an actual tiny CPU Qwen3; never a production CLI code path."""
        instance = cls.__new__(cls)
        instance._initialize(model, tokenizer, attention_implementation, synthetic=True)
        instance.metadata.update({"model_id": "synthetic/tiny-qwen3",
                                  "model_revision": None, "tokenizer_revision": None,
                                  "gpu_name": None, "driver_version": None,
                                  "model_load_seconds": None})
        return instance

    def _initialize(self, model: Any, tokenizer: Any, attention_implementation: str,
                    *, synthetic: bool) -> None:
        torch, transformers = _runtime()
        from transformers.cache_utils import DynamicCache
        from transformers.generation.logits_process import (
            LogitsProcessorList, MinPLogitsWarper, TemperatureLogitsWarper,
            TopKLogitsWarper, TopPLogitsWarper,
        )
        if attention_implementation not in ("eager", "sdpa"):
            raise ValueError("attention_implementation must be eager or sdpa")
        if model.config.model_type != "qwen3":
            raise ValueError("Only Qwen3 causal models are supported")
        if getattr(model.config, "_attn_implementation", None) != attention_implementation:
            raise ValueError("Actual model attention implementation does not match the declared one")
        devices = {parameter.device for parameter in model.parameters()}
        if len(devices) != 1:
            raise ValueError("The common backend requires all parameters on a single device")
        device = next(iter(devices))
        if synthetic and device.type != "cpu":
            raise ValueError("from_test_model is restricted to explicitly synthetic CPU fixtures")
        if not synthetic and device != torch.device("cuda:0"):
            raise ValueError("Production requires the single device cuda:0")
        self.torch, self.model, self.tokenizer = torch, model.eval(), tokenizer
        self.device, self._DynamicCache, self._synthetic = device, DynamicCache, synthetic
        self._owner, self._request_id, self._generator = object(), 0, None
        self._samplers = LogitsProcessorList([
            TemperatureLogitsWarper(0.6), TopKLogitsWarper(20, min_tokens_to_keep=1),
            TopPLogitsWarper(0.95, min_tokens_to_keep=1), MinPLogitsWarper(0.0, min_tokens_to_keep=1),
        ])
        wait = tokenizer.encode("Wait", add_special_tokens=False)
        end_think = tokenizer.encode("</think>", add_special_tokens=False)
        if len(wait) != 1 or len(end_think) != 1:
            raise ValueError("Pinned protocol requires single-token Wait and </think> markers")
        self.markers = TokenMarkers(
            wait[0], end_think[0], tokenizer.eos_token_id,
            tuple(tokenizer.encode(TRIAL_PREFIX, add_special_tokens=False)),
            tuple(tokenizer.encode(FINAL_PREFIX, add_special_tokens=False)))
        self.vocab_size = int(model.config.vocab_size)
        self._tokens((self.markers.wait, self.markers.end_think, self.markers.eos))
        self._tokens(self.markers.trial_prefix)
        self._tokens(self.markers.final_prefix)
        capacity = model.config.max_position_embeddings
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("Model config must declare a positive context capacity")
        tokenizer_capacity = getattr(tokenizer, "model_max_length", None)
        # HF's enormous sentinel is not an actual tokenizer context limit.
        if type(tokenizer_capacity) is int and 0 < tokenizer_capacity < 10**9:
            capacity = min(capacity, tokenizer_capacity)
        self.context_limit = capacity
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError("A recorded chat template is required")
        self.metadata = {
            "runner_protocol": RUNNER_PROTOCOL,
            "scope": "synthetic_cpu_backend_validation" if synthetic else "common_online_validation",
            "validation_model": synthetic,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__, "transformers_version": transformers.__version__,
            "cuda_version": torch.version.cuda, "device": str(device),
            "model_class": type(model).__name__, "model_type": model.config.model_type,
            "model_config": model.config.to_dict(),
            "model_generation_config": model.generation_config.to_dict(),
            "model_parameter_dtype": str(next(model.parameters()).dtype),
            "attention_implementation": attention_implementation, "batch_size": 1,
            "context_limit": self.context_limit, "tokenizer_chat_template": tokenizer.chat_template,
            "tokenizer_class": type(tokenizer).__name__,
            "markers": {"wait": self.markers.wait, "end_think": self.markers.end_think,
                        "eos": self.markers.eos, "trial_prefix": list(self.markers.trial_prefix),
                        "final_prefix": list(self.markers.final_prefix)},
            "sampling": {"temperature": 0.6, "top_k": 20, "top_p": 0.95, "min_p": 0.0,
                         "logits_dtype": "torch.float32", "raw_probability_dtype": "torch.float32",
                         "processor_order": [type(x).__name__ for x in self._samplers],
                         "rng": "private torch.Generator on execution device"},
            "probe": {"max_new_tokens": MAX_PROBE_TOKENS, "greedy": True,
                      "confidence_arithmetic": "pinned torch dtype; skip first, subtract last, denominator n-1",
                      "eos_policy": "continue to end-think/cap, then mark observation invalid",
                      "cache": "independent DynamicCache with cloned layer tensors",
                      "logits_to_keep": 0},
            "main_logits_to_keep": 1,
            "timing_scope": "caller times loaded-model request including clone/probe/restoration",
        }

    def _tokens(self, token_ids: Sequence[int]) -> tuple[int, ...]:
        tokens = tuple(token_ids)
        if not tokens or any(type(value) is not int or not 0 <= value < self.vocab_size for value in tokens):
            raise ValueError("Expected nonempty token IDs within the model vocabulary")
        return tokens

    def prompt_ids(self, question: str) -> list[int]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a nonempty string")
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": question + TASK_SUFFIX}],
            tokenize=True, add_generation_prompt=True, enable_thinking=True)
        tokens = self._tokens(ids)
        if len(tokens) > self.context_limit:
            raise ValueError("Prompt exceeds the model context capacity")
        return list(tokens)

    def start_request(self, seed_reason: int) -> None:
        if type(seed_reason) is not int or not 0 <= seed_reason < 2**63:
            raise ValueError("seed_reason must be an integer in [0, 2**63)")
        self._request_id += 1
        self._generator = self.torch.Generator(device=self.device)
        self._generator.manual_seed(seed_reason)
        if not self._synthetic:
            self.torch.cuda.reset_peak_memory_stats(self.device)

    def _validate_state(self, state: TorchState) -> None:
        if not isinstance(state, TorchState) or state._owner is not self._owner:
            raise ValueError("State belongs to a different backend")
        if state.request_id != self._request_id or self._generator is None:
            raise ValueError("State belongs to a different or uninitialized request")
        if not isinstance(state.cache, self._DynamicCache) or state.cache.get_seq_length() != state.length:
            raise ValueError("Stale state: use the state returned by the latest extend call")

    def _forward(self, cache: Any, token_ids: Sequence[int], *, keep_last_only: bool) -> TorchState:
        tokens = self._tokens(token_ids)
        length = int(cache.get_seq_length()) + len(tokens)
        if length > self.context_limit:
            raise ValueError("Accepted prefix would exceed model context capacity")
        with self.torch.no_grad():
            outputs = self.model(
                input_ids=self.torch.tensor([tokens], dtype=self.torch.long, device=self.device),
                past_key_values=cache, use_cache=True, return_dict=True,
                logits_to_keep=1 if keep_last_only else 0)
            logits = outputs.logits[0, -1].detach().clone()
        return TorchState(outputs.past_key_values, logits, length, self._request_id, self._owner)

    def prefill(self, prompt_ids: Sequence[int]) -> TorchState:
        if self._generator is None:
            raise RuntimeError("start_request must precede prefill")
        return self._forward(self._DynamicCache(), prompt_ids, keep_last_only=True)

    def extend(self, state: TorchState, token_ids: Sequence[int]) -> TorchState:
        self._validate_state(state)
        return self._forward(state.cache, token_ids, keep_last_only=True)

    def _sampling_probabilities(self, state: TorchState):
        self._validate_state(state)
        logits = state.logits.float().unsqueeze(0)
        if not self.torch.isfinite(logits).all().item():
            raise ValueError("Main-generation logits are nonfinite")
        # These four warpers do not inspect historical input IDs. No hidden
        # generation-config penalties or forced-token processors are applied.
        scores = self._samplers(self.torch.empty((1, 0), dtype=self.torch.long,
                                                 device=self.device), logits.clone())
        return self.torch.softmax(logits, dim=-1), self.torch.softmax(scores, dim=-1)

    def sample(self, state: TorchState) -> TokenSample:
        raw, sampling = self._sampling_probabilities(state)
        token = int(self.torch.multinomial(sampling, num_samples=1, generator=self._generator).item())
        return TokenSample(token, float(sampling[0, token].item()), float(raw[0, token].item()))

    def greedy(self, state: TorchState) -> TokenSample:
        self._validate_state(state)
        logits = state.logits.float()
        if not self.torch.isfinite(logits).all().item():
            raise ValueError("Greedy-generation logits are nonfinite")
        probabilities = self.torch.softmax(logits, dim=-1)
        value, token = self.torch.max(probabilities, dim=0)
        return TokenSample(int(token.item()), float(value.item()), float(value.item()))

    def _clone_cache(self, cache: Any):
        clone = self._DynamicCache()
        for layer_index in range(len(cache)):
            key, value = cache[layer_index]
            clone.update(key.clone(), value.clone(), layer_index)
        return clone

    def probe(self, state: TorchState) -> ProbeObservation:
        self._validate_state(state)
        if state.length + len(self.markers.trial_prefix) + MAX_PROBE_TOKENS > self.context_limit:
            raise ValueError("Insufficient context reserve for the complete pinned probe")
        token_ids, token_probs = [], []
        # Only this branch is extended. Main cache, logits, and sampling generator
        # are never touched; Python references to the branch expire on return.
        with self.torch.no_grad():
            branch = self._forward(self._clone_cache(state.cache), self.markers.trial_prefix,
                                   keep_last_only=False)
            total = 0.0
            while True:
                probabilities = self.torch.nn.functional.softmax(branch.logits, dim=-1)
                maximum, token_tensor = self.torch.max(probabilities, dim=0)
                if token_ids:
                    total += self.torch.log(maximum)
                token = int(token_tensor.item())
                token_ids.append(token)
                token_probs.append(float(maximum.item()))
                if token == self.markers.end_think or len(token_ids) >= MAX_PROBE_TOKENS:
                    break
                branch = self._forward(branch.cache, (token,), keep_last_only=False)
            # Retain original torch dtype/order even for short or nonfinite
            # results. ProbeObservation guards, rather than repairs, those cases.
            confidence = self.torch.exp((total - self.torch.log(maximum)) / (len(token_ids) - 1))
            raw = float(confidence.item())
            dtype = str(branch.logits.dtype)
        return ProbeObservation(
            tuple(token_ids), tuple(token_probs), raw, token_ids[-1] == self.markers.end_think,
            confidence_source=f"torch_model_logits_softmax/{dtype}",
            invalid_reason="probe_generated_eos" if self.markers.eos in token_ids else None)

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        tokens = tuple(token_ids)
        if not tokens:
            return ""
        self._tokens(tokens)
        return self.tokenizer.decode(list(tokens), skip_special_tokens=skip_special_tokens)

    def synchronize(self) -> None:
        if not self._synthetic:
            self.torch.cuda.synchronize(self.device)

    def peak_memory_bytes(self) -> int | None:
        if self._synthetic:
            return None
        return int(self.torch.cuda.max_memory_allocated(self.device))
