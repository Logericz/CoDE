"""Bounded-plan, per-question rule, and actual tiny-BF16 CPU checks."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_validate_online_gpu as base_tests

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("validate_online_continuation", ROOT / "scripts" / "validate_online_continuation.py")
continuation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(continuation)
validation = continuation.validation


def case(question, index, position):
    return {"name": f"{question}-probe-{index}", "question": question,
            "probe_index": index, "stop_token_idx": position}


def observation(confidence, ended=False, tokens=(10, 11, 12)):
    return continuation.ProbeObservation(tokens, (0.9,) * len(tokens), confidence, ended,
                                         confidence_source="synthetic_CPU_rule_fixture")


class DenseRuleTests(unittest.TestCase):
    def test_histories_restart_for_each_question_and_do_not_pool_three_points(self):
        history = continuation.DenseRuleHistory()
        first = history.observe(case("q001", 0, 202), "online", observation(0.95))
        second = history.observe(case("q001", 1, 989), "online", observation(0.7))
        other = history.observe(case("q002", 0, 3046), "online", observation(0.5))
        self.assertEqual([first["D"], second["D"], other["D"]], [0.0, 0.0, 0.0])
        self.assertEqual(other["valid_history_positions"], [3046])
        self.assertEqual(other["confidence_threshold"], 0.9)

    def test_third_decreasing_confidence_computes_real_D_and_degen_only_stop(self):
        history = continuation.DenseRuleHistory()
        for index, (position, confidence) in enumerate(zip((3046, 3217, 5331), (0.95, 0.8, 0.7))):
            result = history.observe(case("q002", index, position), "online", observation(confidence))
        expected = (1.0 + math.log(5331 / 3217)) + 1.0
        self.assertAlmostEqual(result["D"], expected)
        self.assertTrue(result["degeneration_stop"])
        self.assertFalse(result["confidence_stop"])
        self.assertTrue(result["combined_stop"])
        self.assertFalse(result["stop_applied"])

    def test_confidence_uses_strict_ramp_threshold_and_ended_flag(self):
        for confidence, ended, expected in ((0.9, True, False), (0.91, True, True), (0.99, False, False)):
            with self.subTest(confidence=confidence, ended=ended):
                result = continuation.DenseRuleHistory().observe(case("q001", 0, 202), "online",
                                                                 observation(confidence, ended))
                self.assertEqual(result["confidence_stop"], expected)

    def test_online_and_fullprefix_paths_keep_independent_histories(self):
        history = continuation.DenseRuleHistory()
        for index, position in enumerate((3046, 3217, 5331)):
            online = history.observe(case("q002", index, position), "online", observation((0.95, 0.8, 0.7)[index]))
            full = history.observe(case("q002", index, position), "full_prefix", observation(0.95))
        self.assertGreater(online["D"], 2)
        self.assertEqual(full["D"], 0)
        self.assertNotEqual(online["combined_stop"], full["combined_stop"])

    def test_invalid_confidence_is_kept_out_of_D_history_and_cannot_stop(self):
        history = continuation.DenseRuleHistory()
        first = history.observe(case("q001", 0, 202), "online", observation(float("nan"), True))
        second = history.observe(case("q001", 1, 989), "online", observation(0.99, True))
        self.assertIsNone(first["D"])
        self.assertFalse(first["combined_stop"])
        self.assertEqual(second["valid_history_count"], 1)
        self.assertEqual(second["confidence_threshold"], 0.925)
        self.assertTrue(second["confidence_stop"])

    def test_refuses_missing_dense_candidate_or_nonincreasing_position(self):
        history = continuation.DenseRuleHistory()
        with self.assertRaisesRegex(ValueError, "consecutive"):
            history.observe(case("q001", 1, 989), "online", observation(0.8))
        history.observe(case("q001", 0, 202), "online", observation(0.8))
        with self.assertRaisesRegex(ValueError, "positions must increase"):
            history.observe(case("q001", 1, 202), "online", observation(0.8))

    def test_reference_eos_gets_the_same_validity_guard_without_changing_raw_confidence(self):
        record = {"token_ids": [10, 2, 12], "token_probs": [0.9, 0.9, 0.9], "confidence": 0.99, "ended": True}
        result = continuation.observation_from_record(record, (2, 3), "fixed_original_BF16")
        self.assertEqual(result.confidence_raw, 0.99)
        self.assertFalse(result.confidence_valid)
        self.assertIn("probe_generated_eos", result.invalid_reasons)

    def test_pending_eos_and_post_think_wait_do_not_enter_online_history(self):
        backend = SimpleNamespace(markers=SimpleNamespace(wait=3, eos_ids=(2,), end_think=4))
        history = continuation.DenseRuleHistory()
        for response, reason in (([10, 2], "pending_EOS"), ([4, 3], "prefix_already")):
            backend.tokenizer = lambda _text, values=response: {"input_ids": values}
            item = {**case("q001", 0, 1), "sample_response": "synthetic"}
            item["candidate_identity"] = continuation.candidate_identity(item, backend)
            self.assertFalse(item["candidate_identity"]["online_candidate_eligible"])
            self.assertTrue(any(reason in value for value in item["candidate_identity"]["exclusion_reasons"]))
            decision = history.observe(item, "online", observation(0.99, True))
            self.assertIsNone(decision["D"])
            self.assertIsNone(decision["candidate_j"])
            self.assertFalse(decision["combined_stop"])
        self.assertEqual(history.histories, {})

    def test_threshold_grid_reuses_raw_probabilities_and_detects_strict_confidence_crossing(self):
        history, completed = continuation.DenseRuleHistory(), []
        raw = {"token_ids": [10, 11, 12], "token_probs": [0.9, 0.9, 0.9], "ended": True}
        item = case("q001", 0, 202)
        records = {"online": {**raw, "confidence": 0.95}, "same_KV": {**raw, "confidence": 0.95},
                   "full_prefix": {**raw, "confidence": 0.96}}
        decisions = {path: history.observe(item, path, continuation.observation_from_record(record, (2,), "CPU_fixture"))
                     for path, record in records.items()}
        completed.append({"case": item["name"], "question": "q001", "raw_observations": records,
                          "stopping_decisions": decisions})
        report = continuation.threshold_sensitivity(completed, (2,))
        self.assertEqual(len(report["entries"]), 18)
        self.assertEqual(report["new_GPU_calls"], 0)
        deer = next(entry for entry in report["entries"] if entry["config"]["rule"] == "deer"
                    and entry["config"]["deer_threshold"] == 0.95)
        self.assertTrue(deer["rows"][0]["cross_path_stop_flags_differ"])
        self.assertFalse(deer["rows"][0]["by_path"]["same_KV"]["combined_stop"])
        self.assertTrue(deer["rows"][0]["by_path"]["full_prefix"]["combined_stop"])


class PlanIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cases = [case("q001", 0, 202), case("q001", 1, 989), case("q002", 0, 3046),
                      case("q002", 1, 3217), case("q002", 2, 5331)]
        for index, item in enumerate(self.cases):
            item.update(prefix_ids_sha256=f"saved-prefix-{index}", input_sha256={"stage": "recorded"})

    def write_identity(self):
        records = {"plan.json": {"scope": continuation.parity.SCOPE,
                    "prefix_ids_sha256": self.cases[0]["prefix_ids_sha256"],
                    "attention_implementation": "eager", "seed": 872},
                   "summary.json": {"status": "diagnostic_completed", "same_KV_parity": {"status": "passed"}},
                   "environment.json": {},
                   "source-hashes.json": {filename: validation.sha((ROOT / filename).read_bytes())
                       for filename in continuation.parity.CRITICAL_SOURCE_FILES}}
        for filename, record in records.items():
            validation.write_json(self.root / filename, record)

    def test_budget_is_explicit_and_uses_five_points_from_two_histories(self):
        result = continuation.budget(self.cases)
        self.assertEqual(result["fixed_selectors"], ["q001:0", "q001:1", "q002:0", "q002:1", "q002:2"])
        self.assertEqual(result["comparison_saved_token_extends"], 6320)
        self.assertEqual(result["isolation_saved_token_extends"], 9138)
        self.assertEqual(result["maximum_new_probe_tokens"], 378)
        self.assertEqual(result["maximum_new_continuation_tokens"], 192)

    def test_each_input_uses_old_helper_with_one_selector_without_lifting_old_cap(self):
        with patch.object(validation, "load_saved_prefixes", side_effect=[[item] for item in self.cases]) as load:
            self.assertEqual(continuation.load_cases(self.root), self.cases)
        self.assertEqual([call.args[1] for call in load.call_args_list], [[value] for value in continuation.SELECTORS])

    def test_identity_preserves_source_and_prefix_before_gpu_access(self):
        self.write_identity()
        result = continuation.load_identity(self.root, self.cases)
        self.assertEqual(result["seed"], 872)
        self.assertEqual(len(result["sha256"]), 4)
        altered = [dict(self.cases[0], prefix_ids_sha256="changed"), *self.cases[1:]]
        with self.assertRaisesRegex(ValueError, "different first saved prefix"):
            continuation.load_identity(self.root, altered)

    def test_inspect_only_has_no_gpu_or_writes_and_declares_new_contract(self):
        self.write_identity()
        before = set(self.root.iterdir())
        output = io.StringIO()
        with patch.object(continuation, "load_cases", return_value=self.cases), \
                patch.object(continuation, "gpu_inventory") as inventory, redirect_stdout(output):
            code = continuation.main(["--evidence-root", str(self.root), "--identity-root", str(self.root), "--inspect-only"])
        inventory.assert_not_called()
        self.assertEqual(code, 0)
        self.assertEqual(set(self.root.iterdir()), before)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["reference_contract"], continuation.REFERENCE_CONTRACT)
        self.assertIn("not an equivalence acceptance gate", plan["cross_path_policy"])
        self.assertEqual(plan["stop_rule_config"]["tau"], 2.0)

    def test_mutated_source_or_input_is_detected_and_recorded(self):
        source = self.root / "saved-input.json"
        source.write_text("original")
        expected = {str(source): validation.sha(source.read_bytes())}
        source.write_text("changed")
        with self.assertRaises(validation.AcceptanceFailure):
            continuation.verify_integrity(self.root, expected)
        record = json.loads((self.root / "first-difference.json").read_text())
        self.assertEqual(record["phase"], "source_input_identity_integrity")
        self.assertTrue((self.root / "integrity-after.json").exists())

    def test_keyboard_interrupt_creates_atomic_failure_and_no_success_summary(self):
        self.write_identity()
        run = self.root / "new-run"
        with patch.object(continuation, "load_cases", return_value=self.cases), \
                patch.object(continuation, "gpu_inventory", side_effect=KeyboardInterrupt), redirect_stdout(io.StringIO()):
            code = continuation.main(["--evidence-root", str(self.root / "evidence"), "--identity-root", str(self.root),
                                      "--model-dir", str(self.root / "model"), "--run-root", str(self.root.parent / (self.root.name + "-run"))])
        actual_run = self.root.parent / (self.root.name + "-run")
        import shutil
        self.addCleanup(shutil.rmtree, actual_run)
        self.assertEqual(code, 1)
        failure = json.loads((actual_run / "failure.json").read_text())
        self.assertEqual(failure["error_type"], "KeyboardInterrupt")
        self.assertFalse(failure["gate_passed"])
        self.assertFalse((actual_run / "summary.json").exists())


@unittest.skipUnless(base_tests.RUNTIME_AVAILABLE, "requires pinned tiny BF16 CPU environment")
class TinyContinuationTests(unittest.TestCase):
    setUp = base_tests.TinyBF16AcceptanceTests.setUp

    def reference(self):
        _, probe_function, _ = validation.reference_source()
        _, cache_function, _ = continuation.parity.cache_source()
        return continuation.parity.load_cached_reference(probe_function, cache_function, self.backend.torch)

    def test_consecutive_prefixes_reuse_main_kv_and_never_sample_saved_tokens(self):
        initial = {**self.case, **case("q001", 0, 4)}
        following = {**self.case, **case("q001", 1, 6), "prefix_ids": self.case["prefix_ids"] + [19, 20]}
        with patch.object(self.backend, "prefill", wraps=self.backend.prefill) as prefill, \
                patch.object(self.backend, "sample", side_effect=AssertionError("must not sample saved tokens")):
            state = continuation.advance_saved_prefix(self.backend, initial, None, None, self.root, 872)
            original_cache = state.cache
            state = continuation.advance_saved_prefix(self.backend, following, initial, state, self.root, 872)
        self.assertIs(state.cache, original_cache)
        self.assertEqual(state.length, len(following["prefix_ids"]))
        prefill.assert_called_once_with(initial["prompt_ids"])

    def test_crosspath_probability_difference_is_saved_but_not_a_same_KV_failure(self):
        import torch
        current = {**self.case, **case("q001", 0, 4)}
        self.backend.start_request(872)
        state = validation.rebuild_online_prefix(self.backend, current, self.root)
        reference = self.reference()
        def changed_fullprefix(model, input_ids, kv_cache, tokenizer, method=0):
            result = reference(model, input_ids, kv_cache, tokenizer, method=method)
            if kv_cache is None:
                result = {**result, "token_probs": [0.0, *result["token_probs"][1:]]}
            return result
        with patch.object(torch.cuda, "empty_cache"), redirect_stdout(io.StringIO()):
            result = continuation.evaluate_prefix(self.backend, changed_fullprefix, current, state, self.root,
                                                   continuation.DenseRuleHistory())
        self.assertEqual(result["same_KV_gate"]["status"], "passed")
        self.assertEqual(result["cross_path_sensitivity"]["status"], "different")
        self.assertFalse(result["cross_path_sensitivity"]["is_acceptance_gate"])
        self.assertTrue((self.root / "q001-probe-0-cross-path-sensitivity.json").exists())
        self.assertFalse((self.root / "first-difference.json").exists())

    def test_same_KV_difference_fails_before_fullprefix_and_preserves_partial_raw_output(self):
        import torch
        current = {**self.case, **case("q001", 0, 4)}
        self.backend.start_request(872)
        state = validation.rebuild_online_prefix(self.backend, current, self.root)
        reference = self.reference()
        calls = []
        def changed_same_kv(model, input_ids, kv_cache, tokenizer, method=0):
            calls.append(kv_cache is not None)
            result = reference(model, input_ids, kv_cache, tokenizer, method=method)
            return {**result, "token_probs": [0.0, *result["token_probs"][1:]]}
        with patch.object(torch.cuda, "empty_cache"), redirect_stdout(io.StringIO()), \
                self.assertRaises(validation.AcceptanceFailure):
            continuation.evaluate_prefix(self.backend, changed_same_kv, current, state, self.root,
                                          continuation.DenseRuleHistory())
        self.assertEqual(calls, [True])
        failure = json.loads((self.root / "first-difference.json").read_text())
        self.assertEqual(failure["phase"], "same_KV_probe_numerics")
        self.assertTrue((self.root / "path-q001-probe-0-online.json").exists())
        self.assertTrue((self.root / "path-q001-probe-0-same-KV.json").exists())


if __name__ == "__main__":
    unittest.main()
