"""Synthetic CPU fixtures: token evidence inspection, never model/benchmark results."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("inspect_saved_finalization", ROOT / "scripts/inspect_saved_finalization.py")
inspector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inspector)


class FakeTokenizer:
    eos_token_id = 99
    pieces = {1: "<special>", 2: "reason", 3: "</think>", 4: "\\boxed{2008}",
              5: "explanation unfinished", 99: "<eos>"}

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert clean_up_tokenization_spaces is False
        return "".join(self.pieces[token] for token in ids if not skip_special_tokens or token not in (1, 99))


class SavedFinalizationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.tokenizer_path = self.root / "tokenizer"
        self.tokenizer_path.mkdir()
        (self.tokenizer_path / "tokenizer_config.json").write_text('{}', encoding="utf-8")
        self.stage_path = self.root / "codestop.json"
        self.environment_path = self.root / "environment.json"
        self.record = {"method_output": {"response": "reason</think>\\boxed{2008}</think>explanation unfinished"},
                       "generation_calls": [{"status": "ok", "stage": "codestop", "call_index": 0,
                                             "input_token_ids": [[1, 2, 3]],
                                             "output_token_ids": [[1, 2, 3, 4, 3, 5]],
                                             "generated_token_ids": [[4, 3, 5]],
                                             "max_new_tokens": 3, "do_sample": True}]}
        self.environment = {"model": {"snapshot": str(self.tokenizer_path)}}
        self.save()

    def save(self):
        self.stage_path.write_text(json.dumps(self.record), encoding="utf-8")
        self.environment_path.write_text(json.dumps(self.environment), encoding="utf-8")

    def inspect(self, **kwargs):
        return inspector.inspect_saved(self.stage_path, tokenizer=FakeTokenizer(), **kwargs)

    def test_double_think_exposes_omitted_answer_without_grading_or_writes(self):
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        report = self.inspect()
        think = report["saved_response"]["thinking_end"]
        self.assertEqual(think["count"], 2)
        self.assertEqual(think["text_after_last_marker"], "explanation unfinished")
        self.assertIn("\\boxed{2008}", think["text_before_last_marker"])
        for boundary in think["boundaries"]:
            self.assertEqual(self.record["method_output"]["response"][boundary["start_character"]:boundary["end_character"]], "</think>")
        self.assertTrue(report["length_and_eos_facts"]["recorded_budget_reached"])
        self.assertFalse(report["length_and_eos_facts"]["last_generated_token_is_tokenizer_eos"])
        self.assertNotIn("correct", report)
        self.assertIn("<special>", report["decoded"]["input"]["special_tokens_preserved"]["text"])
        self.assertNotIn("<special>", report["decoded"]["input"]["special_tokens_removed"]["text"])
        self.assertEqual(report["sources"]["stage"]["sha256"], inspector.file_sha256(self.stage_path))
        self.assertEqual(before, {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()})

    def test_selects_last_success_before_failed_call_and_reports_eos_fact(self):
        self.record["generation_calls"].insert(0, dict(self.record["generation_calls"][0]))
        call = self.record["generation_calls"][1]
        call["output_token_ids"][0][-1] = 99
        call["generated_token_ids"][0][-1] = 99
        self.record["generation_calls"].append({"status": "failed", "error": "synthetic failure"})
        self.save()
        report = self.inspect()
        self.assertEqual(report["selected_call"]["list_index"], 1)
        self.assertEqual(report["selected_call"]["later_failed_calls"], 1)
        self.assertEqual(report["length_and_eos_facts"]["generated_tokenizer_eos_positions"], [2])
        self.assertTrue(report["length_and_eos_facts"]["last_generated_token_is_tokenizer_eos"])

    def test_prefix_and_suffix_mismatches_fail_before_tokenizer_loading(self):
        for name, message in (("input_token_ids", "complete input"), ("generated_token_ids", "output suffix")):
            with self.subTest(name=name):
                original = self.record["generation_calls"][0][name][0][0]
                self.record["generation_calls"][0][name][0][0] = 88
                self.save()
                with patch.object(inspector, "load_local_tokenizer") as loader:
                    with self.assertRaisesRegex(inspector.InspectionError, message):
                        inspector.inspect_saved(self.stage_path)
                    loader.assert_not_called()
                self.record["generation_calls"][0][name][0][0] = original

    def test_malformed_token_ids_budget_and_no_success_fail(self):
        cases = [({"input_token_ids": [[True]]}, "integer token"),
                 ({"input_token_ids": [[-1]]}, "integer token"),
                 ({"input_token_ids": [[1], [2]]}, "batch=1"),
                 ({"input_token_ids": [[]]}, "cannot be empty"),
                 ({"max_new_tokens": 2}, "exceeds"),
                 ({"max_new_tokens": False}, "positive integer"),
                 ({"status": "failed"}, "No successful")]
        original = json.loads(json.dumps(self.record))
        for updates, message in cases:
            with self.subTest(updates=updates):
                self.record = json.loads(json.dumps(original))
                self.record["generation_calls"][0].update(updates)
                self.save()
                with self.assertRaisesRegex(inspector.InspectionError, message):
                    self.inspect()

    def test_corrupt_json_and_missing_response_fail(self):
        for content in ("{", '{"a": 1, "a": 2}', '{"a": NaN}', "[]"):
            with self.subTest(content=content):
                self.stage_path.write_text(content, encoding="utf-8")
                with self.assertRaises(inspector.InspectionError):
                    self.inspect()
        self.record["method_output"].pop("response")
        self.save()
        with self.assertRaisesRegex(inspector.InspectionError, "response"):
            self.inspect()

    def test_missing_snapshot_requires_existing_local_override(self):
        self.environment["model"]["snapshot"] = "/missing/old-server/snapshot"
        self.save()
        with self.assertRaisesRegex(inspector.InspectionError, "not available locally"):
            self.inspect()
        report = self.inspect(tokenizer_path=self.tokenizer_path)
        self.assertTrue(report["tokenizer"]["path_overridden"])
        with self.assertRaisesRegex(inspector.InspectionError, "not available locally"):
            self.inspect(tokenizer_path="Qwen/Qwen3-4B")

    def test_loader_forces_local_only_without_remote_code(self):
        loader = SimpleNamespace(from_pretrained=lambda *args, **kwargs: (args, kwargs))
        with patch.dict(sys.modules, {"transformers": SimpleNamespace(AutoTokenizer=loader)}), patch.dict(inspector.os.environ, {}, clear=False):
            args, kwargs = inspector.load_local_tokenizer(self.tokenizer_path)
            self.assertEqual(args, (str(self.tokenizer_path),))
            self.assertEqual(kwargs, {"local_files_only": True, "trust_remote_code": False})
            self.assertEqual(inspector.os.environ["HF_HUB_OFFLINE"], "1")

    def test_cli_outputs_json_and_returns_nonzero_on_corruption(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(inspector, "load_local_tokenizer", return_value=FakeTokenizer()), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            rc = inspector.main([str(self.stage_path)])
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "inspected")
        self.assertEqual(stderr.getvalue(), "")
        self.stage_path.write_text("{", encoding="utf-8")
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            rc = inspector.main([str(self.stage_path)])
        self.assertEqual(rc, 1)
        self.assertEqual(stdout.getvalue(), "")
        error = json.loads(stderr.getvalue())
        self.assertEqual(error["status"], "error")
        self.assertEqual(error["sources"]["stage"]["sha256"], inspector.file_sha256(self.stage_path))


if __name__ == "__main__":
    unittest.main()
