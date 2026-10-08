"""CPU orchestration tests using explicit fixtures; not natural GPU evidence."""
from contextlib import nullcontext, redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("validate_online_boundaries", ROOT / "scripts" / "validate_online_boundaries.py")
boundaries = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boundaries)


class BoundaryValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.metadata = {"validation_model": False, "model_id": boundaries.MODEL_ID,
            "model_revision": boundaries.MODEL_REVISION, "model_parameter_dtype": "torch.bfloat16",
            "attention_implementation": "eager", "device": "cuda:0", "markers": {"end_think": 4, "eos_ids": [2, 3]}}
        samples = [{"token_id": token, "raw_probability": 0.5, "sample_probability": 0.7,
                    "generation_index": index, "phase": "reason" if index <= 1 else "answer"}
                   for index, token in enumerate((10, 4, 11, 2))]
        self.first = {"status": "completed", "stop_reason": "natural_eos", "method": "vanilla",
            "seed_reason": boundaries.SEED_REASON, "max_new_tokens": 512,
            "answer_boundary": {"source": "natural_first_end_think", "confirmed": True, "closing_token_index": 1},
            "injected_prompt_tokens": 0, "finalization_tokens": 0, "discarded_generated_tokens": 0,
            "main_samples": samples, "output_token_ids": [10, 4, 11, 2], "backend_metadata": self.metadata}
        self.second = {**deepcopy(self.first), "stop_reason": "budget", "max_new_tokens": 2,
                       "main_samples": samples[:2], "output_token_ids": [10, 4, 12, 2], "finalization_tokens": 2}
        self.third = {**deepcopy(self.first), "stop_reason": "budget", "max_new_tokens": 3,
                      "main_samples": samples[:3], "output_token_ids": [10, 4, 11, 12, 2], "finalization_tokens": 2}
        pieces = {2: "<|im_end|>", 3: "<|endoftext|>", 4: "</think>", 5: "<think>",
                  10: "reason", 11: "2", 12: ".", 20: "\n", 21: " "}
        self.backend = SimpleNamespace(metadata=self.metadata, tokenizer=SimpleNamespace(all_special_ids=[2, 3]),
            decode=lambda ids, **_kwargs: "".join(pieces[token] for token in ids))

    def test_budget_is_observed_natural_close_plus_one_and_exact_pair_passes(self):
        budget = boundaries.natural_budget(self.first)
        self.assertEqual(budget, 2)
        report = boundaries.validate_pair(self.first, self.second, budget)
        self.assertTrue(report["passed"])
        self.assertEqual(report["injected_prompt_tokens_both_requests"], 0)
        self.assertFalse(report["forced_tokens"])

    def test_main_sampling_probability_difference_is_not_hidden_by_equal_tokens(self):
        second = deepcopy(self.second)
        second["main_samples"][0]["raw_probability"] = 0.5001
        with self.assertRaisesRegex(boundaries.CoverageUnmet, "diverged"):
            boundaries.validate_pair(self.first, second, 2)

    def test_body_budget_skips_blank_prefixes_and_uses_first_visible_natural_token(self):
        first = deepcopy(self.first)
        tokens = [10, 4, 20, 21, 11, 12, 2]
        first["output_token_ids"] = tokens
        first["main_samples"] = [{**self.first["main_samples"][min(i, 3)], "token_id": token,
                                  "generation_index": i, "phase": "reason" if i <= 1 else "answer"}
                                 for i, token in enumerate(tokens)]
        selection = boundaries.answer_body_budget(first, self.backend)
        self.assertEqual(selection["max_new_tokens"], 5)
        self.assertEqual(selection["selected_output_token_index"], 4)
        self.assertEqual(selection["decoded_answer_prefix"], "\n 2")
        self.assertEqual(selection["skipped_leading_blank_prefixes"], 2)
        self.assertTrue(selection["terminal_EOS_after_selection"])
        self.assertFalse(selection["selection_uses_gold"])

    def test_body_budget_requires_nonempty_body_and_no_control_anywhere_in_answer(self):
        for middle in ([20, 21], [11, 5], [11, 4], [11, 3]):
            first = deepcopy(self.first)
            first["output_token_ids"] = [10, 4, *middle, 2]
            first["main_samples"] = [{"token_id": token, "phase": "reason" if i <= 1 else "answer"}
                                     for i, token in enumerate(first["output_token_ids"])]
            with self.subTest(middle=middle), self.assertRaises(boundaries.CoverageUnmet):
                boundaries.answer_body_budget(first, self.backend)

    def test_body_pair_requires_exact_samples_and_reaches_answer_phase(self):
        selection = boundaries.answer_body_budget(self.first, self.backend)
        report = boundaries.validate_pair(self.first, self.third, 3, body_selection=selection)
        self.assertTrue(report["nonempty_answer_body_at_budget"])
        self.assertEqual(report["natural_closing_token_index"], 1)
        third = deepcopy(self.third)
        third["main_samples"][-1]["sample_probability"] = 0.70001
        with self.assertRaisesRegex(boundaries.CoverageUnmet, "diverged"):
            boundaries.validate_pair(self.first, third, 3, body_selection=selection)

    def test_rejects_artificial_closure_or_no_natural_eos_without_rebudgeting(self):
        for change in ({"stop_reason": "budget"}, {"injected_prompt_tokens": 12}, {"finalization_tokens": 1},
                       {"answer_boundary": {"source": "controlled_final_prefix", "confirmed": True}}):
            with self.subTest(change=change), self.assertRaises(boundaries.CoverageUnmet):
                boundaries.natural_budget({**self.first, **change})

    def test_rejects_budget_change_seed_change_and_injected_second_prefix(self):
        for change in ({"max_new_tokens": 3}, {"seed_reason": boundaries.SEED_REASON + 1},
                       {"injected_prompt_tokens": 12}, {"finalization_tokens": 31}, {"finalization_tokens": 0}):
            with self.subTest(change=change), self.assertRaises(boundaries.CoverageUnmet):
                boundaries.validate_pair(self.first, {**self.second, **change}, 2)

    def test_backend_refuses_tiny_cpu_or_different_dtype_before_request(self):
        boundaries.validate_backend(self.metadata)
        for key, value in (("validation_model", True), ("device", "cpu"), ("model_parameter_dtype", "torch.float32"),
                           ("attention_implementation", "sdpa")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                boundaries.validate_backend({**self.metadata, key: value})

    def args(self, run="run"):
        return boundaries.parser().parse_args(["--model-dir", str(self.root / "model"), "--run-root", str(self.root / run)])

    def execute_fixture(self, request_results, run="run", cap=512):
        with patch.object(boundaries, "gpu_inventory", return_value={"uuid": "GPU-fixture", "name": "CPU-test fixture"}), \
                patch.object(boundaries, "gpu_lock", return_value=nullcontext("fixture.lock")), \
                patch.object(boundaries, "assert_gpu_idle", return_value={}), \
                patch.object(boundaries, "run_request", side_effect=request_results) as requests, \
                redirect_stdout(io.StringIO()):
            args = self.args(run)
            args.max_main_tokens = cap
            code = boundaries.run_validation(args, backend_factory=lambda *_args, **_kwargs: self.backend)
        return code, requests

    def test_successful_orchestration_has_three_actual_calls_same_seed_and_derived_budgets(self):
        code, requests = self.execute_fixture([deepcopy(self.first), deepcopy(self.second), deepcopy(self.third)])
        self.assertEqual(code, 0)
        self.assertEqual([call.kwargs["max_new_tokens"] for call in requests.call_args_list], [512, 2, 3])
        self.assertEqual([call.kwargs["seed_reason"] for call in requests.call_args_list], [872, 872, 872])
        summary = json.loads((self.root / "run" / "summary.json").read_text())
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["executed_count"], 3)
        self.assertEqual(summary["unexecuted_count"], 0)
        self.assertTrue(summary["code_unchanged_during_run"])
        self.assertTrue(summary["coverage_complete"])
        self.assertEqual(summary["covered_branches"], list(boundaries.BRANCHES))
        plan = json.loads((self.root / "run" / "third-request-plan.json").read_text())
        self.assertEqual(plan["source_sha256"], summary["requests"][0]["source_sha256"])
        self.assertEqual(plan["max_new_tokens"], 3)
        for row in summary["requests"]:
            path = self.root / "run" / (row["configuration"] + ".json")
            self.assertEqual(row["source_sha256"], boundaries.sha256(path))
            record = json.loads(path.read_text())
            self.assertFalse(record["forced_tokens"])
            self.assertTrue(record["synthetic_question"])
            self.assertFalse(record["eligible_for_primary_speed_comparison"])

    def test_uncovered_first_request_stops_without_seed_or_cap_retry(self):
        code, requests = self.execute_fixture([{**deepcopy(self.first), "stop_reason": "budget"}])
        self.assertEqual(code, 2)
        self.assertEqual(requests.call_count, 1)
        summary = json.loads((self.root / "run" / "summary.json").read_text())
        self.assertEqual(summary["status"], "uncovered")
        self.assertEqual(summary["unexecuted_count"], 2)
        self.assertEqual(summary["covered_branches"], [])
        self.assertFalse((self.root / "run" / "natural-answer-budget.json").exists())

    def test_third_uncovered_keeps_two_observed_branches_and_prior_pair(self):
        first = deepcopy(self.first)
        first["main_samples"][2]["token_id"] = 20
        first["output_token_ids"][2] = 20
        code, requests = self.execute_fixture([first, deepcopy(self.second)])
        self.assertEqual(code, 2)
        self.assertEqual(requests.call_count, 2)
        summary = json.loads((self.root / "run" / "summary.json").read_text())
        self.assertEqual(summary["status"], "uncovered")
        self.assertEqual(summary["covered_branches"], list(boundaries.BRANCHES[:2]))
        self.assertTrue(summary["boundary_pair_passed"])
        self.assertFalse(summary["answer_body_pair_passed"])
        self.assertFalse(summary["coverage_complete"])
        self.assertEqual(summary["coverage_checks"][boundaries.LABELS[2]]["status"], "uncovered")
        self.assertFalse((self.root / "run" / "natural-answer-body-budget.json").exists())

    def test_third_sampling_failure_keeps_saved_partial_and_earlier_coverage(self):
        code, requests = self.execute_fixture([deepcopy(self.first), deepcopy(self.second), KeyboardInterrupt()])
        self.assertEqual(code, 1)
        self.assertEqual(requests.call_count, 3)
        summary = json.loads((self.root / "run" / "summary.json").read_text())
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["covered_branches"], list(boundaries.BRANCHES[:2]))
        self.assertEqual(summary["requests"][-1]["status"], "failed")
        self.assertTrue(summary["boundary_pair_passed"])

    def test_explicit_4096_cap_does_not_retry_when_uncovered(self):
        code, requests = self.execute_fixture([{**deepcopy(self.first), "stop_reason": "budget"}], cap=4096)
        self.assertEqual(code, 2)
        self.assertEqual(requests.call_count, 1)
        self.assertEqual(requests.call_args.kwargs["max_new_tokens"], 4096)
        self.assertEqual(requests.call_args.kwargs["seed_reason"], 872)

    def test_keyboard_interrupt_saves_executed_failed_request_and_summary(self):
        code, requests = self.execute_fixture([KeyboardInterrupt()])
        self.assertEqual(code, 1)
        self.assertEqual(requests.call_count, 1)
        summary = json.loads((self.root / "run" / "summary.json").read_text())
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["requests"][0]["status"], "failed")
        self.assertTrue((self.root / "run" / "failure.json").exists())

    def test_existing_run_refused_without_gpu_activity(self):
        (self.root / "run").mkdir()
        with patch.object(boundaries, "gpu_inventory") as inventory, self.assertRaises(FileExistsError):
            boundaries.run_validation(self.args())
        inventory.assert_not_called()

    def test_inspect_only_predeclares_seed_question_and_cap_without_gpu(self):
        args = boundaries.parser().parse_args(["--inspect-only"])
        output = io.StringIO()
        with patch.object(boundaries, "gpu_inventory") as inventory, redirect_stdout(output):
            self.assertEqual(boundaries.run_validation(args), 0)
        inventory.assert_not_called()
        manifest = json.loads(output.getvalue())
        self.assertEqual(manifest["question"], "Compute 1 + 1.")
        self.assertEqual(manifest["seed_reason"], 872)
        self.assertEqual(manifest["max_main_tokens_first"], 512)
        self.assertEqual(manifest["planned_count"], 3)
        self.assertEqual(manifest["configurations"], list(boundaries.LABELS))
        self.assertIn("non-whitespace", manifest["third_main_budget_policy"])
        for cap in (512, 1024, 4096):
            self.assertEqual(boundaries.parser().parse_args(["--max-main-tokens", str(cap)]).max_main_tokens, cap)
        args.max_main_tokens = 8192
        with self.assertRaises(ValueError):
            boundaries.run_validation(args)


if __name__ == "__main__":
    unittest.main()
