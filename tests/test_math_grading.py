"""CPU evidence for grading only; no model quality or GPU performance claim."""
from importlib import metadata
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import math_grading as grading


def record(answer=r"\boxed{2}", gold="2", identifier="q1", **kwargs):
    return {"id": identifier, "gold": gold, "answer_text": answer, **kwargs}


def dependencies_available():
    try:
        return all(metadata.version(name) == version for name, version in grading.REQUIRED_VERSIONS.items())
    except metadata.PackageNotFoundError:
        return False


class ExtractionAndAccountingTests(unittest.TestCase):
    def test_last_box_not_first_correct_box(self):
        result = grading.extract_final_answer(r"First \boxed{2}; correction \boxed{3}")
        self.assertEqual(result["selected_answer"], "3")
        self.assertEqual(result["complete_box_count"], 2)

    def test_balanced_nested_fraction_and_escaped_set_braces(self):
        result = grading.extract_final_answer(r"Answer: \boxed{\frac{1}{2}}")
        self.assertEqual(result["selected_answer"], r"\frac{1}{2}")
        self.assertEqual(grading.extract_final_answer(r"\boxed{\{1,2\}}")['selected_answer'], r"\{1,2\}")

    def test_last_complete_box_with_unclosed_tail_preserved(self):
        result = grading.extract_final_answer(r"\boxed{2} correction \boxed{\frac{1}{")
        self.assertEqual(result["selected_answer"], "2")
        self.assertTrue(result["has_unclosed_tail"])
        self.assertEqual(len(result["unclosed_box_positions"]), 1)

    def test_nested_box_is_not_promoted_to_final_answer(self):
        with patch.object(grading, "_run_worker") as worker:
            result = grading.grade_record(record(r"\boxed{\frac{1}{\boxed{2}}}", r"\frac{1}{2}"))
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["reason"], "nested_boxed_answer_not_supported")
        self.assertNotEqual(result["selected_answer"], "2")
        worker.assert_not_called()
        unclosed_outer = grading.extract_final_answer(r"\boxed{\frac{1}{\boxed{2}}")
        self.assertEqual(unclosed_outer["status"], "malformed_box")

    def test_empty_malformed_and_empty_last_box_are_distinct(self):
        self.assertEqual(grading.extract_final_answer("")["status"], "empty_answer")
        self.assertEqual(grading.extract_final_answer(r"\boxed{2")["status"], "malformed_box")
        self.assertEqual(grading.extract_final_answer(r"\boxed")["status"], "malformed_box")
        self.assertEqual(grading.extract_final_answer(r"\boxed{2}\boxed{}")["status"], "empty_answer")

    def test_think_spans_and_duplicate_think_never_reach_verifier(self):
        for answer in (r"<think>\boxed{2}</think>\boxed{3}",
                       r"<think>\boxed{2}</think><think>again</think>\boxed{2}",
                       r"</think>\boxed{2}", r"<think\boxed{2}",
                       r"<|analysis|>\boxed{2}", r"[THINK]\boxed{2}"):
            with self.subTest(answer=answer), patch.object(grading, "_run_worker") as worker:
                result = grading.grade_record(record(answer))
                self.assertEqual(result["status"], "needs_review")
                self.assertIsNone(result["grade"])
                worker.assert_not_called()

    def test_unboxed_prose_not_scanned_for_a_correct_number(self):
        result = grading.extract_final_answer("I considered 2 but could not solve it.")
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(grading.extract_final_answer("$2$")["selected_answer"], "2")

    def test_duplicate_ids_and_invalid_schema_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            grading.grade_records([record(), record()], 2)
        for rows, planned in (([record()], 0), ([record(source_sha256="bad")], 1),
                              ([record(execution_status="not_run")], 1), ([record(gold="")], 1)):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                grading.grade_records(rows, planned)

    def test_budget_exhaustion_is_not_a_wrong_answer(self):
        with patch.object(grading, "_run_worker", return_value={"status": "correct", "grade": True, "reason": None}):
            result = grading.grade_record(record(stop_reason="budget"))
        self.assertEqual(result["status"], "correct")
        self.assertEqual(result["boundary_policy"], "unknown")
        self.assertEqual(result["provenance_status"], "unknown")

    def test_explicit_unconfirmed_boundary_never_reaches_verifier(self):
        with patch.object(grading, "_run_worker") as worker:
            result = grading.grade_record(record(answer_boundary_confirmed=False))
        worker.assert_not_called()
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["reason"], "answer_boundary_unconfirmed")
        self.assertIsNone(result["grade"])
        self.assertFalse(result["answer_boundary_confirmed"])
        self.assertEqual(result["boundary_check_version"], "explicit-answer-boundary-v1")

    def test_confirmed_boundary_still_rejects_repeated_think(self):
        with patch.object(grading, "_run_worker") as worker:
            result = grading.grade_record(record(r"\boxed{2}</think>", answer_boundary_confirmed=True))
        worker.assert_not_called()
        self.assertEqual(result["reason"], "reasoning_boundary_not_isolated")
        self.assertTrue(result["answer_boundary_confirmed"])

    def test_boundary_flag_requires_boolean_and_failure_takes_precedence(self):
        for value in (None, 0, 1, "false"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "boolean"):
                grading.validate_records([record(answer_boundary_confirmed=value)], 1)
        result = grading.grade_record(record(execution_status="failed", answer_boundary_confirmed=False))
        self.assertEqual(result["status"], "run_failure")

    def test_provenance_is_preserved_but_not_claimed_verified(self):
        result = grading.grade_record(record("", source_sha256="a" * 64,
                                              boundary_policy="caller preserved injected boxed prefix and new tokens"))
        self.assertEqual(result["source_sha256"], "a" * 64)
        self.assertEqual(result["provenance_status"], "caller_asserted_not_independently_verified")

    def test_executed_denominator_includes_failures_and_unresolved(self):
        with patch.object(grading, "_run_worker", side_effect=[
            {"status": "correct", "grade": True}, {"status": "incorrect", "grade": False}]):
            report = grading.grade_records([
                record(identifier="right"), record(r"\boxed{3}", identifier="wrong"),
                record("", identifier="empty"), record("", identifier="failed", execution_status="failed")], 6)
        summary = report["summary"]
        self.assertEqual(summary["executed_count"], 4)
        self.assertEqual(summary["unexecuted_count"], 2)
        self.assertEqual(summary["run_failure_count"], 1)
        self.assertEqual(summary["unresolved_count"], 1)
        self.assertEqual(summary["accuracy_confirmed_over_executed"], .25)
        self.assertEqual(summary["accuracy_upper_if_all_unresolved_correct"], .5)
        self.assertFalse(summary["matrix_execution_complete"])

    def test_zero_executed_rows_is_not_zero_accuracy(self):
        summary = grading.grade_records([], 20)["summary"]
        self.assertIsNone(summary["accuracy_confirmed_over_executed"])
        self.assertEqual(summary["unexecuted_count"], 20)

    def test_worker_timeout_error_and_malformed_response_are_not_incorrect(self):
        with patch.object(grading.subprocess, "run", side_effect=subprocess.TimeoutExpired("worker", 1)):
            self.assertEqual(grading.grade_record(record())["status"], "timeout")
        with patch.object(grading.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, "", "")):
            self.assertEqual(grading.grade_record(record())["status"], "evaluator_error")
        with patch.object(grading.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, '{"status":"incorrect","grade":null}', "")):
            self.assertEqual(grading.grade_record(record())["status"], "evaluator_error")

    def test_real_child_process_is_killed_by_outer_deadline(self):
        started = time.monotonic()
        result = grading._run_worker({"gold": "2", "prediction": "2"}, .00001)
        self.assertEqual(result["status"], "timeout")
        self.assertIsNone(result["grade"])
        self.assertLess(time.monotonic() - started, 5)

    def test_worker_reports_baseexception_timeout_instead_of_false(self):
        class TimeoutException(BaseException):
            pass
        output = io.StringIO()
        with patch.object(grading, "_compare_payload", side_effect=TimeoutException()), \
             patch.object(sys, "stdin", io.StringIO('{}')), patch.object(sys, "stdout", output):
            grading._worker_main()
        self.assertEqual(json.loads(output.getvalue())["status"], "timeout")

    def test_blank_jsonl_row_is_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.jsonl"
            path.write_text(json.dumps(record()) + "\n\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "blank"):
                grading.read_jsonl(path)

    def test_duplicate_json_keys_and_nonfinite_json_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.jsonl"
            for bad in ('{"id":"a","id":"b"}', '{"value":NaN}', '{"value":Infinity}'):
                path.write_text(bad + "\n", encoding="utf-8")
                with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "invalid JSON"):
                    grading.read_jsonl(path)

    def test_dependency_drift_is_rejected(self):
        with patch.object(grading.metadata, "version", return_value="0.0.0"):
            with self.assertRaisesRegex(RuntimeError, "Expected math-verify"):
                grading._check_dependencies()


@unittest.skipUnless(dependencies_available(), "requires math-verify[antlr4_13_2]==0.9.0")
class RealMathVerifyTests(unittest.TestCase):
    def test_integer_fraction_radical_set_interval_and_wrong(self):
        cases = [("2", r"\boxed{2}", "correct"),
                 (r"\frac{1}{2}", r"\boxed{0.5}", "correct"),
                 (r"\sqrt{8}", r"\boxed{2\sqrt{2}}", "correct"),
                 (r"\{1,2\}", r"\boxed{\{2,1\}}", "correct"),
                 (r"[0,1)", r"\boxed{[0,1)}", "correct"),
                 ("2", r"\boxed{3}", "incorrect"),
                 ("2", r"\boxed{2} \boxed{3}", "incorrect")]
        for gold, answer, expected in cases:
            with self.subTest(gold=gold, answer=answer):
                self.assertEqual(grading.grade_record(record(answer, gold))["status"], expected)

    def test_parsing_failures_are_not_negative_grades(self):
        result = grading.grade_record(record(r"\boxed{\unknowncommand{???}}"))
        self.assertIn(result["status"], {"prediction_parse_failure", "unsupported_type"})
        self.assertIsNone(result["grade"])
        result = grading.grade_record(record(gold=r"\unknowncommand{???}"))
        self.assertEqual(result["status"], "gold_parse_failure")

    def test_string_fallback_is_rejected_and_argument_order_is_gold_first(self):
        from sympy import Integer
        with patch("math_verify.parse", return_value=["2"]):
            result = grading._compare_payload({"gold": "2", "prediction": "2"})
            self.assertEqual(result["status"], "gold_parse_failure")
        with patch("math_verify.parse", side_effect=[[Integer(2)], [Integer(3)]]) as parse, \
             patch("math_verify.verify", return_value=False) as verify:
            grading._compare_payload({"gold": "2", "prediction": "3"})
            self.assertEqual(verify.call_args.args, ([Integer(2)], [Integer(3)]))
            self.assertIsNone(verify.call_args.kwargs["timeout_seconds"])
            self.assertTrue(verify.call_args.kwargs["raise_on_error"])
            self.assertEqual(parse.call_args.kwargs["fallback_mode"], "no_fallback")
            self.assertIsNone(parse.call_args.kwargs["parsing_timeout"])

    def test_cli_creates_new_report_records_versions_and_preserves_input(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "answers.jsonl", Path(folder) / "graded.json"
            source.write_text(json.dumps(record()) + "\n", encoding="utf-8")
            original = source.read_bytes()
            command = [sys.executable, str(ROOT / "scripts/grade_math_answers.py"), "--input", str(source),
                       "--planned-count", "2", "--output", str(output)]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(report["summary"]["executed_count"], 1)
            self.assertEqual(report["summary"]["confirmed_correct_count"], 1)
            self.assertEqual(report["summary"]["unexecuted_count"], 1)
            self.assertEqual(report["manifest"]["required_versions"]["math-verify"], "0.9.0")
            self.assertEqual(report["manifest"]["input"]["sha256"], grading.file_sha256(source))
            self.assertGreaterEqual(len(report["manifest"]["source_sha256"]), 5)
            saved = output.read_bytes()
            again = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(again.returncode, 2)
            self.assertEqual(output.read_bytes(), saved)
            command[-1] = str(source)
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 2)
            self.assertEqual(source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
