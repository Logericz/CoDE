"""Actual tiny-HF-model CPU checks; no GPU or pretrained benchmark claims."""
import ast
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from online_contract import FINAL_PREFIX, TASK_SUFFIX, TRIAL_PREFIX
from online_protocol import UPSTREAM_COMMIT
from torch_online_backend import TorchOnlineBackend

try:
    import torch
    import transformers
    from transformers import GenerationConfig, Qwen3Config, Qwen3ForCausalLM
    from transformers.generation.logits_process import LogitsProcessorList
    RUNTIME_AVAILABLE = (torch.__version__.split("+")[0] == "2.9.1"
                         and transformers.__version__ == "4.51.3")
except ImportError:
    RUNTIME_AVAILABLE = False


class SyntheticTokenizer:
    eos_token_id = 2
    model_max_length = 256
    chat_template = "synthetic tiny-model template; not the pretrained tokenizer"

    def encode(self, text, **kwargs):
        return {"Wait": [3], "</think>": [4], TRIAL_PREFIX: [10, 11, 12],
                FINAL_PREFIX: [4, 10, 11, 12]}[text]

    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text, **kwargs)}

    def apply_chat_template(self, messages, **kwargs):
        self.last_messages, self.last_template_kwargs = messages, kwargs
        return [1, 5, 6, 7, 8]

    def decode(self, values, skip_special_tokens=True):
        return " ".join(str(value) for value in values
                        if not skip_special_tokens or value not in (0, 1, 2, 4))


@unittest.skipUnless(RUNTIME_AVAILABLE, "requires the pinned torch/transformers CPU validation environment")
class TinyQwenBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        result = subprocess.run(
            ["git", "-C", str(ROOT / "upstream" / "CoDE-Stop"), "show",
             f"{UPSTREAM_COMMIT}:method_codestop.py"], check=True, capture_output=True, text=True)
        function = next(node for node in ast.parse(result.stdout).body
                        if isinstance(node, ast.FunctionDef) and node.name == "calcu_max_probs_w_kv")
        namespace = {"torch": torch, "F": torch.nn.functional}
        exec(compile(ast.Module(body=[function], type_ignores=[]),
                     f"{UPSTREAM_COMMIT}:method_codestop.py", "exec"), namespace)
        cls.reference_probe = staticmethod(namespace["calcu_max_probs_w_kv"])

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(123)
            config = Qwen3Config(
                vocab_size=64, hidden_size=32, intermediate_size=64,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                head_dim=8, max_position_embeddings=256, bos_token_id=1,
                eos_token_id=2, pad_token_id=0, attention_dropout=0.0,
                tie_word_embeddings=False)
            config._attn_implementation = "eager"
            self.model = Qwen3ForCausalLM(config).eval()
        self.tokenizer = SyntheticTokenizer()
        self.backend = TorchOnlineBackend.from_test_model(self.model, self.tokenizer)

    def state(self, seed=171):
        self.backend.start_request(seed)
        return self.backend.prefill(self.backend.prompt_ids("2+2?"))

    def cache_snapshot(self, state):
        return [(key.clone(), value.clone()) for key, value in state.cache]

    def assert_cache_equal(self, snapshot, state):
        self.assertEqual(len(snapshot), len(state.cache))
        for (old_key, old_value), (key, value) in zip(snapshot, state.cache):
            self.assertTrue(torch.equal(old_key, key))
            self.assertTrue(torch.equal(old_value, value))

    def test_probe_preserves_main_KV_logits_and_private_and_global_rng(self):
        state = self.state()
        snapshot, logits = self.cache_snapshot(state), state.logits.clone()
        private_rng = self.backend._generator.get_state().clone()
        global_rng = torch.random.get_rng_state().clone()
        result = self.backend.probe(state)
        self.assertGreaterEqual(len(result.token_ids), 1)
        self.assertLessEqual(len(result.token_ids), 21)
        self.assert_cache_equal(snapshot, state)
        self.assertTrue(torch.equal(logits, state.logits))
        self.assertTrue(torch.equal(private_rng, self.backend._generator.get_state()))
        self.assertTrue(torch.equal(global_rng, torch.random.get_rng_state()))

    def test_real_main_trajectory_identical_with_inserted_probes(self):
        traces = []
        for with_probe in (False, True):
            state = self.state(seed=872)
            trace = []
            for index in range(15):
                if with_probe and index in (0, 3, 5, 9):
                    self.backend.probe(state)
                token = self.backend.sample(state)
                trace.append(token)
                state = self.backend.extend(state, (token.token_id,))
            traces.append(trace)
        self.assertEqual(traces[0], traces[1])

    def test_cached_probe_matches_pinned_full_prefix_reference(self):
        state = self.state()
        extra = (15, 16, 17, 18)
        state = self.backend.extend(state, extra)
        actual = self.backend.probe(state)
        full = self.backend.prompt_ids("2+2?") + list(extra) + list(self.backend.markers.trial_prefix)
        with patch.object(torch.cuda, "empty_cache"):
            reference = self.reference_probe(self.model, torch.tensor([full]), None,
                                             self.tokenizer, method=0)
        self.assertEqual(actual.token_ids, tuple(reference["token_ids"]))
        torch.testing.assert_close(torch.tensor(actual.token_probs), torch.tensor(reference["token_probs"]),
                                   atol=5e-7, rtol=5e-6)
        torch.testing.assert_close(torch.tensor(actual.confidence_raw),
                                   torch.tensor(reference["total_prob_max"]),
                                   atol=5e-7, rtol=5e-6, equal_nan=True)
        self.assertEqual(actual.ended_with_think, reference["ended_with_think"])

    def test_sampler_matches_HF_processor_order_and_multinomial(self):
        state = self.state()
        config = GenerationConfig(do_sample=True, temperature=0.6, top_k=20, top_p=0.95,
                                  min_p=0.0, repetition_penalty=1.0, num_beams=1)
        config._eos_token_tensor = torch.tensor([2])
        processors = self.model._get_logits_processor(
            generation_config=config, input_ids_seq_length=state.length,
            encoder_input_ids=torch.tensor([[1, 5, 6, 7, 8]]),
            prefix_allowed_tokens_fn=None, logits_processor=LogitsProcessorList(), device="cpu")
        self.assertEqual([type(x).__name__ for x in self.backend._samplers],
                         [type(x).__name__ for x in processors])
        reference_scores = processors(torch.tensor([[1, 5, 6, 7, 8]]), state.logits.float().unsqueeze(0).clone())
        expected = torch.softmax(reference_scores, dim=-1)
        raw, sampling = self.backend._sampling_probabilities(state)
        torch.testing.assert_close(sampling, expected, rtol=0, atol=0)
        rng = torch.Generator(device="cpu")
        rng.set_state(self.backend._generator.get_state())
        token = int(torch.multinomial(expected, 1, generator=rng).item())
        actual = self.backend.sample(state)
        self.assertEqual(actual.token_id, token)
        self.assertEqual(actual.sample_probability, expected[0, token].item())
        self.assertEqual(actual.raw_probability, raw[0, token].item())

    def test_sampling_does_not_extend_and_extend_accepts_pending_token(self):
        state = self.state()
        snapshot = self.cache_snapshot(state)
        token = self.backend.sample(state)
        self.assertEqual(state.length, state.cache.get_seq_length())
        self.assert_cache_equal(snapshot, state)
        following = self.backend.extend(state, (token.token_id,))
        self.assertEqual(following.length, state.length + 1)
        self.assertEqual(following.cache.get_seq_length(), following.length)
        with self.assertRaisesRegex(ValueError, "Stale state"):
            self.backend.sample(state)

    def test_greedy_and_start_do_not_advance_global_rng(self):
        before = torch.random.get_rng_state().clone()
        state = self.state()
        private = self.backend._generator.get_state().clone()
        token = self.backend.greedy(state)
        self.assertEqual(token.token_id, int(torch.argmax(state.logits).item()))
        self.assertTrue(torch.equal(private, self.backend._generator.get_state()))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def forced_tokens(self, tokens):
        counter = {"index": 0}
        def hook(_module, _args, output):
            target = tokens[min(counter["index"], len(tokens) - 1)]
            counter["index"] += 1
            changed = torch.zeros_like(output)
            changed[..., target] = 2.0
            return changed
        return self.model.lm_head.register_forward_hook(hook)

    def test_probe_EOS_continues_to_cap_then_marks_invalid(self):
        state = self.state()
        handle = self.forced_tokens([self.backend.markers.eos])
        try:
            result = self.backend.probe(state)
        finally:
            handle.remove()
        self.assertEqual(result.token_ids, (self.backend.markers.eos,) * 21)
        self.assertFalse(result.ended_with_think)
        self.assertFalse(result.confidence_valid)
        self.assertIn("probe_generated_eos", result.invalid_reasons)

    def test_generation_config_secondary_eos_invalidates_probe(self):
        secondary_eos = 63
        self.model.generation_config.eos_token_id = [2, secondary_eos]
        self.backend = TorchOnlineBackend.from_test_model(self.model, self.tokenizer)
        self.assertEqual(self.backend.markers.eos, 2)
        self.assertEqual(self.backend.markers.eos_ids, (2, secondary_eos))
        self.assertEqual(self.backend.metadata["markers"]["eos_ids"], [2, secondary_eos])
        state = self.state()
        handle = self.forced_tokens([secondary_eos])
        try:
            result = self.backend.probe(state)
        finally:
            handle.remove()
        self.assertEqual(result.token_ids, (secondary_eos,) * 21)
        self.assertFalse(result.confidence_valid)
        self.assertIn("probe_generated_eos", result.invalid_reasons)

    def test_generation_config_eos_fallback_and_invalid_values(self):
        self.model.generation_config.eos_token_id = None
        backend = TorchOnlineBackend.from_test_model(self.model, self.tokenizer)
        self.assertEqual(backend.markers.eos_ids, (2,))
        self.assertEqual(backend.metadata["markers"]["eos_source"], "tokenizer.eos_token_id_fallback")
        for ids in ([], True, [True], [-1], [3], [4], [64]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.model.generation_config.eos_token_id = ids
                TorchOnlineBackend.from_test_model(self.model, self.tokenizer)

    def test_two_token_probe_keeps_legacy_raw_one_but_invalid(self):
        state = self.state()
        handle = self.forced_tokens([10, self.backend.markers.end_think])
        try:
            result = self.backend.probe(state)
        finally:
            handle.remove()
        self.assertEqual(result.token_ids, (10, self.backend.markers.end_think))
        self.assertEqual(result.confidence_raw, 1.0)
        self.assertFalse(result.confidence_valid)
        self.assertTrue(result.ended_with_think)

    def test_one_token_probe_nonfinite_is_preserved_without_python_division_error(self):
        state = self.state()
        handle = self.forced_tokens([self.backend.markers.end_think])
        try:
            result = self.backend.probe(state)
        finally:
            handle.remove()
        self.assertEqual(len(result.token_ids), 1)
        self.assertFalse(result.confidence_valid)

    def test_nonfinite_probe_keeps_raw_evidence_and_main_cache(self):
        state = self.state()
        snapshot = self.cache_snapshot(state)
        handle = self.model.lm_head.register_forward_hook(
            lambda _module, _args, output: torch.full_like(output, float("nan")))
        try:
            result = self.backend.probe(state)
        finally:
            handle.remove()
        self.assertFalse(result.confidence_valid)
        self.assertEqual(len(result.token_ids), 21)
        self.assertIn("invalid_token_probability", result.invalid_reasons)
        self.assert_cache_equal(snapshot, state)

    def test_context_refusals_preserve_cache_and_request_identity(self):
        state = self.state()
        with self.assertRaisesRegex(ValueError, "context"):
            self.backend.extend(state, (10,) * self.backend.context_limit)
        self.assertEqual(state.cache.get_seq_length(), state.length)
        almost_full = self.backend.extend(state, (10,) * (self.backend.context_limit - state.length - 2))
        with self.assertRaisesRegex(ValueError, "context reserve"):
            self.backend.probe(almost_full)
        self.backend.start_request(99)
        with self.assertRaisesRegex(ValueError, "different or uninitialized request"):
            self.backend.greedy(almost_full)

    def test_marker_attention_context_and_synthetic_metadata(self):
        self.assertTrue(self.backend.metadata["validation_model"])
        self.assertEqual(self.backend.metadata["scope"], "synthetic_cpu_backend_validation")
        self.assertIsNone(self.backend.peak_memory_bytes())
        self.backend.prompt_ids("a question")
        self.assertEqual(self.tokenizer.last_messages[0]["content"], "a question" + TASK_SUFFIX)
        self.assertTrue(self.tokenizer.last_template_kwargs["enable_thinking"])
        with self.assertRaisesRegex(ValueError, "attention"):
            TorchOnlineBackend.from_test_model(self.model, self.tokenizer, attention_implementation="sdpa")
        bad = SyntheticTokenizer()
        original = bad.encode
        bad.encode = lambda text, **kwargs: [3, 5] if text == "Wait" else original(text, **kwargs)
        with self.assertRaisesRegex(ValueError, "single-token"):
            TorchOnlineBackend.from_test_model(self.model, bad)
        bounded = SyntheticTokenizer()
        bounded.model_max_length = 128
        self.assertEqual(TorchOnlineBackend.from_test_model(self.model, bounded).context_limit, 128)

    def test_production_refuses_CPU_without_loading_any_model(self):
        with patch.object(torch.cuda, "is_available", return_value=False), \
                patch.object(transformers.AutoModelForCausalLM, "from_pretrained") as loader:
            with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
                TorchOnlineBackend(ROOT / "does-not-exist")
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
