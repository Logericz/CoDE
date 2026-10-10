"""只测字符边界与结构；不用模型、gold或数学比较器。"""
import hashlib
from pathlib import Path
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from entropy_answer_boundary import KNOWN_BOUNDARY_POLICY, extract_entropy_answer_boundary
from math_grading import extract_final_answer


def extract(text, **kwargs):
    return extract_entropy_answer_boundary(text, answer_boundary_confirmed=kwargs.get("confirmed", True),
                                          boundary_policy=kwargs.get("policy", KNOWN_BOUNDARY_POLICY))


class SupplementalBoundaryTests(unittest.TestCase):
    def test_package_import_in_clean_interpreter(self):
        root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [sys.executable, "-I", "-c",
             "import sys; sys.path.insert(0, sys.argv[1]); "
             "from src.entropy_answer_boundary import extract_entropy_answer_boundary; "
             "assert callable(extract_entropy_answer_boundary)", str(root)],
            capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_nested_braces_preserve_exact_prefix_and_full_raw_hash(self):
        prefix = "  The final answer is \\boxed{\\frac{1}{2}}\n"
        raw = prefix + "</think>later reasoning with \\boxed{3}"
        result = extract(raw)
        self.assertEqual(result["status"], "accepted_prefix")
        self.assertEqual(result["answer_text"], prefix)
        self.assertEqual((result["span_start"], result["span_end"]), (0, len(prefix)))
        self.assertEqual(result["delimiter"], "</think>")
        self.assertEqual(result["raw_text_sha256"], hashlib.sha256(raw.encode()).hexdigest())
        self.assertEqual(result["structure"]["complete_box_count"], 1)

    def test_unclosed_box_is_not_repaired(self):
        result = extract("The final answer is \\boxed{\\frac{1}{2}</think>}")
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["reason"], "dangling_boxed_before_thought_delimiter")
        self.assertIsNone(result["answer_text"])
        self.assertIsNone(result["span_end"])

    def test_multiple_top_level_boxes_are_not_selected_by_outcome(self):
        result = extract(r"\boxed{1} or \boxed{2}</think>tail")
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["structure"]["complete_box_count"], 2)
        self.assertEqual(result["reason"], "exactly_one_complete_top_level_box_required")

    def test_marker_before_box_does_not_search_later_text(self):
        for raw in (r"</think>\boxed{7}", r"The final answer is </think>\boxed{7}"):
            with self.subTest(raw=raw):
                result = extract(raw)
                self.assertEqual(result["status"], "rejected")
                self.assertIsNone(result["answer_text"])

    def test_no_marker_preserves_full_text_and_original_parser_semantics(self):
        raw = r"  \boxed{1} then \boxed{2}  "
        result = extract(raw)
        self.assertEqual(result["status"], "unchanged")
        self.assertEqual(result["answer_text"], raw)
        self.assertEqual(result["span_end"], len(raw))
        self.assertIsNone(result["structure"])
        self.assertEqual(extract_final_answer(result["answer_text"]), extract_final_answer(raw))

    def test_no_marker_empty_answer_is_left_to_original_parser(self):
        result = extract("")
        self.assertEqual(result["status"], "unchanged")
        self.assertEqual(result["answer_text"], "")
        self.assertEqual(extract_final_answer(result["answer_text"])["status"], "empty_answer")

    def test_unknown_or_truthy_nonboolean_boundary_is_rejected(self):
        for confirmed in (False, None, 1, "true"):
            with self.subTest(confirmed=confirmed):
                result = extract(r"\boxed{7}</think>", confirmed=confirmed)
                self.assertEqual(result["reason"], "known_answer_boundary_required")
                self.assertIsNone(result["answer_text"])

    def test_vanilla_or_other_boundary_policy_is_not_allowed(self):
        for policy in ("natural_first_end_think", "vanilla", "unknown", ""):
            with self.subTest(policy=policy):
                result = extract(r"\boxed{7}</think>", policy=policy)
                self.assertEqual(result["status"], "rejected")
                self.assertEqual(result["reason"], "unsupported_boundary_policy")

    def test_first_opening_think_wins_over_later_closing_marker(self):
        prefix = r"\boxed{7}"
        result = extract(prefix + "<think>text</think>" + r"\boxed{8}")
        self.assertEqual(result["answer_text"], prefix)
        self.assertEqual(result["delimiter"], "<think>")
        self.assertEqual(result["delimiter_start"], len(prefix))

    def test_dangling_boxed_command_is_rejected_even_after_complete_box(self):
        for tail in (r" and \boxed", r" and \boxed  ", r" and \boxed{"):
            with self.subTest(tail=tail):
                result = extract(r"\boxed{7}" + tail + "</think>")
                self.assertEqual(result["reason"], "dangling_boxed_before_thought_delimiter")
                self.assertIsNone(result["answer_text"])

    def test_nested_boxed_rejection_from_original_parser_is_preserved(self):
        result = extract(r"\boxed{1+\boxed{2}}</think>")
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["reason"], "original_extractor_rejected_prefix")
        self.assertEqual(result["structure"]["reason"], "nested_boxed_answer_not_supported")

    def test_empty_box_is_not_accepted(self):
        result = extract(r"\boxed{   }</think>")
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["structure"]["status"], "empty_answer")

    def test_other_forbidden_marker_before_delimiter_is_not_removed(self):
        result = extract(r"<analysis>\boxed{7}</think>")
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["structure"]["reason"], "reasoning_boundary_not_isolated")

    def test_no_thought_delimiter_does_not_remove_other_reasoning_markers(self):
        raw = r"<analysis>\boxed{7}"
        result = extract(raw)
        self.assertEqual(result["status"], "unchanged")
        self.assertEqual(result["answer_text"], raw)
        self.assertEqual(extract_final_answer(raw)["status"], "needs_review")

    def test_structure_result_contains_no_grade_or_correctness(self):
        result = extract(r"\boxed{7}</think>")
        self.assertNotIn("grade", result)
        self.assertNotIn("correct", result)
        self.assertNotIn("gold", result)

    def test_nonstring_input_is_not_coerced_into_answer_text(self):
        with self.assertRaises(TypeError):
            extract(None)


if __name__ == "__main__":
    unittest.main()
