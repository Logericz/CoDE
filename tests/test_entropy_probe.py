"""熵观测的接口与tiny-Qwen CPU检查；不代表预训练模型/GPU实验。"""
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from entropy_probe import EntropyProbeSession, clone_state, close_probe, extend_probe, start_probe
from test_torch_online_backend import RUNTIME_AVAILABLE, SyntheticTokenizer

if RUNTIME_AVAILABLE:
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from torch_online_backend import TorchOnlineBackend


class EntropyProbeContractTests(unittest.TestCase):
    def test_invalid_maximum_rejected_before_backend_access(self):
        for maximum in (0, 43, True, 1.5):
            with self.subTest(maximum=maximum), self.assertRaises(ValueError):
                start_probe(None, None, maximum)

    def test_closed_session_rejected_and_close_is_idempotent(self):
        session = EntropyProbeSession(None, object(), 42)
        close_probe(session)
        close_probe(session)
        self.assertIsNone(session.branch)
        with self.assertRaisesRegex(ValueError, "open"):
            extend_probe(session, 21)

    def test_target_validation_without_runtime(self):
        session = EntropyProbeSession(None, None, 42)
        for target in (0, 43, False, 1.5):
            with self.subTest(target=target), self.assertRaises(ValueError):
                extend_probe(session, target)
        session.token_ids = [7, 8]
        with self.assertRaisesRegex(ValueError, "backwards"):
            extend_probe(session, 1)


@unittest.skipUnless(RUNTIME_AVAILABLE, "requires pinned torch/transformers CPU environment")
class TinyEntropyProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(123)
            config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                head_dim=8, max_position_embeddings=256, bos_token_id=1,
                eos_token_id=2, pad_token_id=0, attention_dropout=0.0, tie_word_embeddings=False)
            config._attn_implementation = "eager"
            model = Qwen3ForCausalLM(config).to(dtype=torch.bfloat16).eval()
        self.backend = TorchOnlineBackend.from_test_model(model, SyntheticTokenizer())
        self.backend.start_request(171)
        self.state = self.backend.prefill(self.backend.prompt_ids("2+2?"))

    def nonterminal_forward(self):
        original = self.backend._forward
        def forward(*args, **kwargs):
            state = original(*args, **kwargs)
            logits = state.logits.clone()
            logits[list(self.backend.markers.eos_ids) + [self.backend.markers.end_think]] = -100
            return replace(state, logits=logits)
        return patch.object(self.backend, "_forward", side_effect=forward)

    def test_twenty_one_exact_parity_with_original_non_eos_probe(self):
        with self.nonterminal_forward():
            old = self.backend.probe(self.state)
            session = start_probe(self.backend, self.state)
            new = extend_probe(session, 21)
        self.assertEqual(len(old.token_ids), 21)
        self.assertEqual(new["token_ids"], list(old.token_ids))
        self.assertEqual(new["token_probs"], list(old.token_probs))
        self.assertEqual(new["confidence_raw"], old.confidence_raw)
        self.assertEqual(new["ended_with_think"], old.ended_with_think)
        self.assertTrue(new["entropy_valid"])
        json.dumps(new, allow_nan=False)
        close_probe(session)

    def test_resumed_twenty_one_to_forty_two_matches_one_shot(self):
        with self.nonterminal_forward():
            session = start_probe(self.backend, self.state)
            short = extend_probe(session, 21)
            self.assertEqual(extend_probe(session, 21), short)
            resumed = extend_probe(session, 42)
            direct_session = start_probe(self.backend, self.state)
            direct = extend_probe(direct_session, 42)
        self.assertEqual(resumed, direct)
        self.assertEqual(short["actual_length"], 21)  # 导出快照不能随session后续增长。
        self.assertEqual(len(short["token_ids"]), 21)
        self.assertEqual(resumed["actual_length"], 42)
        self.assertEqual(resumed["termination_reason"], "max_tokens")
        self.assertEqual(session.branch.length, self.state.length + len(self.backend.markers.trial_prefix) + 41)
        self.assertFalse(resumed["can_extend"])
        close_probe(session)
        close_probe(direct_session)

    def test_main_cache_logits_and_private_global_rng_unchanged(self):
        cache = [(key.clone(), value.clone()) for key, value in self.state.cache]
        logits = self.state.logits.clone()
        private = self.backend._generator.get_state().clone()
        global_rng = torch.random.get_rng_state().clone()
        with self.nonterminal_forward():
            session = start_probe(self.backend, self.state)
            extend_probe(session, 21)
            extend_probe(session, 42)
        close_probe(session)
        for (old_k, old_v), (key, value) in zip(cache, self.state.cache):
            self.assertTrue(torch.equal(old_k, key))
            self.assertTrue(torch.equal(old_v, value))
        self.assertTrue(torch.equal(logits, self.state.logits))
        self.assertTrue(torch.equal(private, self.backend._generator.get_state()))
        self.assertTrue(torch.equal(global_rng, torch.random.get_rng_state()))

    def test_clone_can_extend_without_mutating_original(self):
        cloned = clone_state(self.backend, self.state)
        length = self.state.length
        cloned = self.backend.extend(cloned, (7,))
        self.assertEqual(cloned.length, length + 1)
        self.assertEqual(self.state.cache.get_seq_length(), length)
        self.backend._validate_state(self.state)

    def test_think_and_eos_terminate_and_do_not_resume(self):
        for token, reason in ((self.backend.markers.end_think, "end_think"),
                              (self.backend.markers.eos, "eos")):
            session = start_probe(self.backend, self.state)
            logits = torch.zeros_like(session.branch.logits)
            logits[token] = 100
            session.branch = replace(session.branch, logits=logits)
            first = extend_probe(session, 21)
            with patch.object(self.backend, "_forward", side_effect=AssertionError("resumed terminal probe")):
                self.assertEqual(extend_probe(session, 42), first)
            self.assertEqual(first["actual_length"], 1)
            self.assertEqual(first["termination_reason"], reason)
            self.assertFalse(first["confidence_valid"])
            self.assertEqual(first["confidence_nonfinite"], "nan")
            self.assertIn("probe_token_count_le_2", first["invalid_reasons"])
            if reason == "eos":
                self.assertEqual(first["invalid_reason"], "probe_generated_eos")
            json.dumps(first, allow_nan=False)
            close_probe(session)

    def test_nonfinite_logits_are_explicit_without_fabricated_tokens(self):
        session = start_probe(self.backend, self.state)
        session.branch = replace(session.branch, logits=torch.full_like(session.branch.logits, float("nan")))
        result = extend_probe(session, 21)
        self.assertEqual(result["actual_length"], 0)
        self.assertEqual(result["invalid_reason"], "nonfinite_probe_logits")
        self.assertIsNone(result["confidence_raw"])
        self.assertFalse(result["entropy_valid"])
        self.assertFalse(result["can_extend"])
        json.dumps(result, allow_nan=False)
        close_probe(session)

    def test_entropy_is_full_distribution_fp32_not_max_probability(self):
        session = start_probe(self.backend, self.state)
        session.branch = replace(session.branch, logits=torch.zeros_like(session.branch.logits))
        result = extend_probe(session, 1)
        self.assertAlmostEqual(result["entropy_raw_fp32_nats"][0], float(torch.log(torch.tensor(64.)).item()), places=6)
        self.assertEqual(result["token_probs"], [1 / 64])
        self.assertTrue(result["entropy_valid"])
        close_probe(session)

    def test_context_reserve_and_stale_session_are_rejected(self):
        with patch.object(self.backend, "context_limit", self.state.length + 20):
            with self.assertRaisesRegex(ValueError, "context reserve"):
                start_probe(self.backend, self.state)
        session = start_probe(self.backend, self.state)
        self.backend.start_request(172)
        with self.assertRaisesRegex(ValueError, "request"):
            extend_probe(session, 21)
        close_probe(session)


if __name__ == "__main__":
    unittest.main()
