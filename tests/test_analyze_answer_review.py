"""CPU evidence-chain tests; assistant reviews are not human adjudication."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_answer_review as review
import prepare_answer_review as packets


class PolicyTests(unittest.TestCase):
    def test_repeated_closing_marker_only_tail_keeps_original_box_substring(self):
        for tail in (" </think> ", "\n</think>\n</think>\n**Final Answer**\nThe final answer is \\boxed{",
                     "</think>\n**Final Answer**", "</think>The final answer is"):
            with self.subTest(tail=tail):
                text = "中文答案：" + r"\boxed{\frac{1}{2}}" + tail
                result = review.policy_selection(text)
                self.assertEqual(result["rule_id"], "T1")
                start, end = result["selected_expression_char_span"]
                self.assertEqual(text[start:end], r"\boxed{\frac{1}{2}}")
                self.assertEqual(result["selected_expression_verbatim"], text[start:end])

    def test_new_reasoning_numbers_or_open_think_after_answer_remain_unresolved(self):
        for text in (r"\boxed{7}</think>Wait, compute again.",
                     r"\boxed{7}</think>The final answer is \boxed{8",
                     r"\boxed{7}<think>more reasoning",
                     r"</think>\boxed{7}", r"\boxed{7}</analysis>",
                     r"\boxed{\boxed{7}}", r"\boxed{}</think>"):
            with self.subTest(text=text):
                self.assertIsNone(review.policy_selection(text))

    def test_normal_last_complete_box_and_unclosed_tail_match_strict_protocol(self):
        text = r"First \boxed{1}, final \boxed{2}, truncated \boxed{3"
        result = review.policy_selection(text)
        self.assertEqual(result["rule_id"], "N1")
        self.assertEqual(result["selected_expression_verbatim"], r"\boxed{2}")
        self.assertEqual(review.policy_selection("  $7$  ")["selected_expression_char_span"], [2, 5])
        self.assertIsNone(review.policy_selection("I might answer 7"))

    def test_bare_truncation_is_unresolved_before_symbolic_comparison(self):
        for text in (r"\frac{1}{", r"\[7", "7 +", "(1,2", r"\frac{1}{2}}"):
            with self.subTest(text=text):
                self.assertIsNone(review.policy_selection(text))
        for text in (r"\frac{1}{2}", r"\[7\]", "(0,1]", r"\{1,2\}"):
            with self.subTest(text=text):
                self.assertEqual(review.policy_selection(text)["rule_id"], "N1")

    def test_channel_control_tokens_cannot_be_hidden_by_a_later_box(self):
        for text in (r"<|meta_sep|>analysis\boxed{7}",
                     r"\boxed{7}<|fim_suffix|>", r"\boxed{7}</think><|meta_sep|>"):
            with self.subTest(text=text):
                self.assertIsNone(review.policy_selection(text))


class ReviewChainTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.original = self.root / "original.jsonl"
        self.original.write_bytes(review.jsonl_bytes([
            {"id": "fixture-a", "gold": "7", "answer_text": r"\boxed{7}</think>",
             "source_sha256": "1" * 64, "answer_boundary_confirmed": True},
            {"id": "fixture-b", "gold": "8", "answer_text": r"\boxed{8}",
             "source_sha256": "2" * 64, "answer_boundary_confirmed": True}]))
        self.packet = self.root / "packet"
        packets.prepare_review([self.original], self.packet, scope=packets.SCOPE)
        self.cases = review.read_jsonl(self.packet / "reviewer/cases.jsonl")
        self.rows = []
        for case in self.cases:
            self.rows.append({"case_id": case["case_id"],
                "answer_text_sha256": review.text_hash(case["answer_text"]),
                "boundary_decision": "accepted", **review.policy_selection(case["answer_text"]),
                "review_note": "The original complete expression has an exact character span."})
        self.a, self.b = self.root / "a.jsonl", self.root / "b.jsonl"
        self.a.write_bytes(review.jsonl_bytes(self.rows))
        self.b.write_bytes(review.jsonl_bytes(self.rows))
        self.sealed = self.root / "sealed"
        self.policy = ROOT / "docs/ANSWER_BOUNDARY_REVIEW_V1.md"

    def seal(self):
        return review.seal_reviews(self.packet, self.policy, self.a, self.b, self.sealed)

    def test_seal_does_not_need_private_mapping_and_then_prepares_all_rows(self):
        private = self.packet / "private/mapping.jsonl"
        original_bytes = private.read_bytes()
        private.unlink()  # Only a disposable fixture: seal must not read gold.
        sealed = self.seal()
        self.assertFalse(sealed["gold_or_private_mapping_read_in_seal"])
        self.assertFalse(sealed["human_review_performed"])
        private.write_bytes(original_bytes)
        original_hash = review.file_sha256(self.original)
        output = self.root / "prepared"
        manifest = review.prepare_grading(self.packet, self.sealed, output)
        rows = review.read_jsonl(output / "grading-input.jsonl")
        self.assertEqual(manifest["planned_count"], 2)
        self.assertEqual({r["id"] for r in rows}, {"fixture-a", "fixture-b"})
        self.assertTrue(all(r["answer_boundary_confirmed"] for r in rows))
        self.assertTrue(all("</think>" not in r["answer_text"] for r in rows))
        self.assertEqual(review.file_sha256(self.original), original_hash)
        with self.assertRaisesRegex(ValueError, "Output exists"):
            review.prepare_grading(self.packet, self.sealed, output)

    def test_reviewer_disagreement_keeps_case_in_grading_denominator_as_unresolved(self):
        rows = copy.deepcopy(self.rows)
        rows[0].update(boundary_decision="unresolved", rule_id="U1",
                       selected_expression_verbatim=None, selected_expression_char_span=None)
        self.b.write_bytes(review.jsonl_bytes(rows))
        sealed = self.seal()
        self.assertEqual(sealed["disagreements"], 1)
        output = self.root / "prepared"
        review.prepare_grading(self.packet, self.sealed, output)
        grading = review.read_jsonl(output / "grading-input.jsonl")
        self.assertEqual(len(grading), 2)
        self.assertFalse(grading[0]["answer_boundary_confirmed"])
        self.assertEqual(grading[0]["answer_text"], self.cases[0]["answer_text"])

    def test_bad_span_hash_rule_or_case_coverage_is_rejected_before_seal(self):
        variants = []
        for key, value in (("selected_expression_char_span", [0, 1]),
                           ("answer_text_sha256", "0" * 64), ("rule_id", "made_up")):
            bad = copy.deepcopy(self.rows)
            bad[0][key] = value
            variants.append(bad)
        variants += [self.rows[:1], [self.rows[0], self.rows[0]]]
        for rows in variants:
            with self.subTest(rows=rows):
                self.b.write_bytes(review.jsonl_bytes(rows))
                with self.assertRaises(ValueError):
                    self.seal()
                self.assertFalse(self.sealed.exists())

    def test_changed_review_cases_policy_or_consensus_cannot_be_used_after_seal(self):
        self.seal()
        for path in (self.a, self.packet / "reviewer/cases.jsonl", self.sealed / "consensus.jsonl"):
            with self.subTest(path=path):
                before = path.read_bytes()
                path.write_bytes(before + b"\n")
                with self.assertRaises(ValueError):
                    review.prepare_grading(self.packet, self.sealed, self.root / "prepared")
                path.write_bytes(before)
        wrong_policy = self.root / "policy.md"
        wrong_policy.write_bytes(self.policy.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "policy differs"):
            review.seal_reviews(self.packet, wrong_policy, self.a, self.b, self.root / "other")

    def test_changed_private_mapping_and_missing_seal_cannot_prepare_grading(self):
        with self.assertRaises(FileNotFoundError):
            review.prepare_grading(self.packet, self.sealed, self.root / "prepared")
        self.seal()
        private = self.packet / "private/mapping.jsonl"
        private.write_bytes(private.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "Private mapping differs"):
            review.prepare_grading(self.packet, self.sealed, self.root / "prepared")


if __name__ == "__main__":
    unittest.main()
