"""CPU-only checks for the supplement's frozen evidence and label isolation.

All answers and grades below are synthetic.  Accepted spans use a fake grader;
the real grader is used only to test its no-worker, unconfirmed-boundary exit.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import supplement_entropy_boundaries as supplement
from src import math_grading


def synthetic_fixture():
    """Include decided, unresolved, accepted, rejected and gold-review cases."""
    cases = [
        (r"\boxed{7}", True, None),
        (r"\boxed{8}", False, None),
        ("  \\boxed{7}\n</think>later \\boxed{8}", None, "reasoning_boundary_not_isolated"),
        (r"</think>later \boxed{7}", None, "reasoning_boundary_not_isolated"),
        (r"\boxed{7}<think>later", None, "source_gold_requires_review"),
        ("", None, "empty_answer"),
    ]
    records, grades = [], []
    for index in range(3):
        record = {
            "question_id": f"synthetic/q{index}", "candidate_index": 1, "status": "ok",
            "prefix_tokens": 99 + index, "prefix_sha256": str(index) * 64,
            "short_probe_ms": 700.25, "extra_probe_ms": 690.125,
            "long_additional_tokens": 21,
            "history": {"valid_count": index, "previous_D": None if not index else .01},
            "short": {"confidence": .82, "D": .02, "max_probabilities": [.8, .9],
                      "entropies_nats": [1.2, .7]},
            "long": {"confidence": .81, "D": .03, "max_probabilities": [.8, .9, .7],
                     "entropies_nats": [1.2, .7, 1.4]},
            "actions": {},
        }
        for action_index, action in enumerate(("E", "T")):
            text, correct, reason = cases[2 * index + action_index]
            status = "correct" if correct is True else "incorrect" if correct is False else "needs_review"
            record["actions"][action] = {
                "answer_text": text, "answer_boundary_confirmed": True,
                "boundary_policy": "known_injected_final_prefix_v1",
                "remaining_ms": 4000.5 + action_index * 9000,
                "generated_tokens": 3, "answer_token_ids": [4, 5, 6], "token_ids": [4, 5, 6],
                "termination": "cap", "error": 0 if correct is True else 1 if correct is False else None,
                "grading_status": status,
            }
            if action == "T":
                record["actions"][action].update(main_generated_tokens=256,
                                                 main_token_ids=[11, 12], main_termination="cap")
            grades.append({
                "id": supplement.identity(record, action), "answer_text": text,
                "answer_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "gold": "synthetic gold; never mathematically evaluated",
                "status": status, "grade": correct, "reason": reason,
                "answer_boundary_confirmed": True, "review_required": correct is None,
                "grading_elapsed_seconds": .123, "extraction": {"saved": True},
            })
        records.append(record)
    return {"schema_version": "entropy-value-v1",
            "question_ids": [record["question_id"] for record in records] + ["synthetic/no-candidate"],
            "records": records}, grades


def fake_grader(payload):
    """Model only the evaluator's boundary gate, without reading mathematical gold."""
    confirmed = payload["answer_boundary_confirmed"] is True
    return {**deepcopy(payload), "status": "correct" if confirmed else "needs_review",
            "grade": True if confirmed else None,
            "reason": "synthetic_test_result" if confirmed else "answer_boundary_unconfirmed"}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class SupplementalLabelTests(unittest.TestCase):
    def setUp(self):
        self.document, self.grades = synthetic_fixture()
        self.decisions = supplement.structural_decisions(self.document)

    def test_unchanged_spans_reuse_saved_grades_without_mathematical_evaluation(self):
        document = deepcopy(self.document)
        document["records"] = document["records"][:1]
        original = deepcopy(self.grades[:2])
        grader = Mock(side_effect=AssertionError("An unchanged span must never be regraded"))
        updated, grades, changes = supplement.apply_grades(
            document, supplement.structural_decisions(document), original, grader)
        grader.assert_not_called()
        self.assertEqual(updated, document)
        for actual, saved in zip(grades, original):
            self.assertEqual(actual, {**saved, "reused_original_grade": True})
        self.assertEqual([row["old_grade"] for row in changes], [row["new_grade"] for row in changes])
        self.assertEqual(original, self.grades[:2])

    def test_only_labels_change_and_all_inputs_are_preserved(self):
        before = deepcopy((self.document, self.decisions, self.grades))
        grader = Mock(side_effect=fake_grader)
        updated, grades, _ = supplement.apply_grades(self.document, self.decisions, self.grades, grader)
        self.assertEqual((self.document, self.decisions, self.grades), before)
        self.assertEqual(supplement.strip_labels(updated), supplement.strip_labels(self.document))
        self.assertEqual(updated["question_ids"], self.document["question_ids"])
        self.assertIsNot(updated, self.document)
        self.assertEqual(updated["records"][1]["actions"]["E"]["error"], 0)
        self.assertEqual(updated["records"][1]["actions"]["E"]["grading_status"], "correct")
        # Exact original costs, raw generated text and all token/features survive.
        for new_record, old_record in zip(updated["records"], self.document["records"]):
            for action in ("E", "T"):
                for field in ("answer_text", "remaining_ms", "answer_token_ids", "token_ids"):
                    self.assertEqual(new_record["actions"][action][field], old_record["actions"][action][field])
        self.assertEqual(sum(row["reused_original_grade"] for row in grades), 3)

    def test_accepted_prefix_is_exact_and_carries_full_original_hash(self):
        grader = Mock(side_effect=fake_grader)
        supplement.apply_grades(self.document, self.decisions, self.grades, grader)
        calls = {call.args[0]["id"]: call.args[0] for call in grader.call_args_list}
        key = "synthetic/q1:1:E"
        original_text = self.document["records"][1]["actions"]["E"]["answer_text"]
        payload = calls[key]
        self.assertEqual(payload["answer_text"], "  \\boxed{7}\n")
        self.assertIs(payload["answer_boundary_confirmed"], True)
        self.assertEqual(payload["boundary_policy"], supplement.PROTOCOL)
        self.assertEqual(payload["source_sha256"], hashlib.sha256(original_text.encode()).hexdigest())
        self.assertEqual(payload["gold"], self.grades[2]["gold"])
        self.assertEqual(set(calls), {"synthetic/q1:1:E", "synthetic/q1:1:T", "synthetic/q2:1:E"})

    def test_rejected_span_remains_unknown_without_a_math_worker(self):
        def check_boundary_then_grade(payload):
            if payload["id"] == "synthetic/q1:1:T":
                self.assertIs(payload["answer_boundary_confirmed"], False)
                self.assertEqual(payload["answer_text"], "")
                return math_grading.grade_record(payload)
            return fake_grader(payload)

        with patch.object(math_grading, "_run_worker", side_effect=AssertionError("No mathematical comparison allowed")) as worker:
            updated, grades, _ = supplement.apply_grades(
                self.document, self.decisions, self.grades, check_boundary_then_grade)
        worker.assert_not_called()
        rejected = next(row for row in grades if row["id"] == "synthetic/q1:1:T")
        self.assertIsNone(rejected["grade"])
        self.assertEqual(rejected["reason"], "answer_boundary_unconfirmed")
        self.assertIsNone(updated["records"][1]["actions"]["T"]["error"])
        self.assertEqual(updated["records"][1]["actions"]["T"]["grading_status"], "needs_review")

    def test_source_gold_requires_review_cannot_become_a_decided_label(self):
        updated, grades, changes = supplement.apply_grades(
            self.document, self.decisions, self.grades, fake_grader)
        reviewed = next(row for row in grades if row["id"] == "synthetic/q2:1:E")
        self.assertIsNone(reviewed["grade"])
        self.assertEqual(reviewed["status"], "needs_review")
        self.assertEqual(reviewed["reason"], "source_gold_requires_review")
        self.assertIsNone(updated["records"][2]["actions"]["E"]["error"])
        self.assertIsNone(next(row for row in changes if row["id"] == reviewed["id"])["new_grade"])

    def test_missing_extra_and_duplicate_identities_fail_before_grading(self):
        for target in ("decisions", "grades"):
            for corruption in ("missing", "extra", "duplicate"):
                with self.subTest(target=target, corruption=corruption):
                    decisions, grades = deepcopy(self.decisions), deepcopy(self.grades)
                    rows = decisions if target == "decisions" else grades
                    if corruption == "missing":
                        rows.pop()
                    elif corruption == "duplicate":
                        rows.append(deepcopy(rows[0]))
                    else:
                        rows.append({**rows[0], "id": "not-a-saved-answer:1:E"})
                    grader = Mock()
                    with self.assertRaisesRegex(ValueError, "identities"):
                        supplement.apply_grades(self.document, decisions, grades, grader)
                    grader.assert_not_called()

    def test_raw_text_hash_and_original_grade_text_mismatch_are_rejected(self):
        for target in ("boundary_hash", "original_grade_text"):
            with self.subTest(target=target):
                decisions, grades = deepcopy(self.decisions), deepcopy(self.grades)
                if target == "boundary_hash":
                    decisions[0]["raw_text_sha256"] = "0" * 64
                else:
                    grades[0]["answer_text"] += "different text"
                grader = Mock()
                with self.assertRaisesRegex(ValueError, "Answer identity changed"):
                    supplement.apply_grades(self.document, decisions, grades, grader)
                grader.assert_not_called()

    def test_original_analysis_label_disagreement_is_rejected(self):
        for field, replacement in (("error", 1), ("grading_status", "incorrect")):
            with self.subTest(field=field):
                document = deepcopy(self.document)
                document["records"][0]["actions"]["E"][field] = replacement
                grader = Mock()
                with self.assertRaisesRegex(ValueError, "labels disagree"):
                    supplement.apply_grades(document, self.decisions, self.grades, grader)
                grader.assert_not_called()

    def test_structural_decisions_ignore_costs_features_and_correctness_labels(self):
        changed = deepcopy(self.document)
        for record in changed["records"]:
            record["extra_probe_ms"] = 999999
            record["short"]["confidence"] = .01
            for action in record["actions"].values():
                action["error"], action["grading_status"] = 0, "correct"
                action["remaining_ms"] = 1
        self.assertEqual(supplement.structural_decisions(changed), self.decisions)

    def test_structural_stage_rejects_malformed_or_duplicate_saved_records(self):
        for corruption in ("schema", "missing_action", "failed_record", "duplicate"):
            with self.subTest(corruption=corruption):
                document = deepcopy(self.document)
                if corruption == "schema":
                    document["schema_version"] = "unexpected"
                elif corruption == "missing_action":
                    del document["records"][0]["actions"]["T"]
                elif corruption == "failed_record":
                    document["records"][0]["status"] = "failed"
                else:
                    document["records"].append(deepcopy(document["records"][0]))
                with self.assertRaises(ValueError):
                    supplement.structural_decisions(document)


class FrozenSupplementTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.pilot, self.prepared, self.output = (self.base / name for name in ("original", "prepared", "supplement"))
        self.pilot.mkdir()
        self.document, self.grades = synthetic_fixture()
        self.versions = {"fake-evaluator": "pinned-version"}
        self.sources = {"synthetic-source": "a" * 64}
        write_json(self.pilot / "analysis-input.json", self.document)
        write_json(self.pilot / "grading.json", {"completed": True, "dependencies": self.versions,
                                                 "results": self.grades})
        names = ("src/math_grading.py", "src/entropy_value_analysis.py", "scripts/analyze_entropy_value.py")
        write_json(self.pilot / "manifest.json", {"code_sha256": {name: supplement.sha(ROOT / name) for name in names}})
        self.original_bytes = {path.name: path.read_bytes() for path in self.pilot.iterdir()}

    def freeze(self):
        with patch.object(supplement, "source_hashes", return_value=self.sources):
            return supplement.prepare(self.pilot, self.prepared)

    def run_grade(self, grader=fake_grader, versions=None, sources=None):
        with patch.object(supplement, "source_hashes", return_value=self.sources if sources is None else sources), \
                patch.object(math_grading, "_check_dependencies", return_value=self.versions if versions is None else versions), \
                patch.object(math_grading, "grade_record", side_effect=grader):
            return supplement.grade(self.prepared, self.output)

    def assert_original_bytes_unchanged(self):
        self.assertEqual({path.name: path.read_bytes() for path in self.pilot.iterdir()}, self.original_bytes)

    def test_prepare_freezes_all_answers_without_reading_old_grades_as_json(self):
        with patch.object(supplement, "read", wraps=supplement.read) as reader, \
                patch.object(math_grading, "grade_record", side_effect=AssertionError("Preparation cannot grade")) as grader:
            result = self.freeze()
        self.assertNotIn("grading.json", [Path(call.args[0]).name for call in reader.call_args_list])
        grader.assert_not_called()
        self.assertEqual(result["answer_count"], 6)
        self.assertEqual(result["status_counts"], {"unchanged": 3, "accepted_prefix": 2, "rejected": 1})
        frozen = supplement.read(self.prepared / "boundary-decisions.json")
        self.assertEqual(frozen["planned_questions"], 4)
        self.assertEqual(frozen["role"], "post_hoc_exploratory_supplement")
        self.assertEqual(frozen["input_sha256"], {name: hashlib.sha256(data).hexdigest()
                                                  for name, data in self.original_bytes.items()})
        self.assert_original_bytes_unchanged()

    def test_complete_supplement_writes_new_files_and_preserves_original_evidence(self):
        self.freeze()
        frozen_bytes = (self.prepared / "boundary-decisions.json").read_bytes()
        summary = self.run_grade()
        self.assertEqual(summary["answer_count"], 6)
        self.assertEqual(summary["reused_original_grades"], 3)
        self.assertEqual(summary["changed_previously_decided_labels"], [])
        self.assertTrue(summary["original_inputs_unchanged"])
        self.assertEqual(summary["role"], "post_hoc_exploratory_supplement")
        new_input = supplement.read(self.output / "analysis-input.json")
        self.assertEqual(supplement.strip_labels(new_input), supplement.strip_labels(self.document))
        self.assertEqual(supplement.read(self.output / "summary.json"), summary)
        self.assertTrue(supplement.read(self.output / "grading.json")["completed"])
        self.assertEqual((self.prepared / "boundary-decisions.json").read_bytes(), frozen_bytes)
        self.assert_original_bytes_unchanged()

    def test_prepare_rejects_original_evaluator_source_mismatch(self):
        manifest = supplement.read(self.pilot / "manifest.json")
        manifest["code_sha256"]["src/math_grading.py"] = "0" * 64
        write_json(self.pilot / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "source no longer matches"):
            self.freeze()
        self.assertFalse(self.prepared.exists())

    def test_changed_original_input_after_freeze_is_rejected_before_grading(self):
        self.freeze()
        path = self.pilot / "analysis-input.json"
        path.write_bytes(path.read_bytes() + b"\n")
        grader = Mock(side_effect=AssertionError("Changed input cannot be graded"))
        with self.assertRaisesRegex(ValueError, "Frozen original inputs changed"):
            self.run_grade(grader)
        grader.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_tampered_decision_is_rejected_even_when_raw_input_hash_is_unchanged(self):
        self.freeze()
        path = self.prepared / "boundary-decisions.json"
        frozen = supplement.read(path)
        frozen["decisions"][2]["answer_text"] = r"\boxed{999}"
        write_json(path, frozen)
        grader = Mock(side_effect=AssertionError("Altered decision cannot be graded"))
        with self.assertRaisesRegex(ValueError, "altered boundary decisions"):
            self.run_grade(grader)
        grader.assert_not_called()
        self.assertFalse(self.output.exists())
        self.assert_original_bytes_unchanged()

    def test_changed_source_or_protocol_is_rejected_before_grading(self):
        self.freeze()
        grader = Mock(side_effect=AssertionError("Changed source cannot be graded"))
        with self.assertRaisesRegex(ValueError, "Protocol/source changed"):
            self.run_grade(grader, sources={"synthetic-source": "b" * 64})
        path = self.prepared / "boundary-decisions.json"
        frozen = supplement.read(path)
        frozen["protocol"] = "other-protocol"
        write_json(path, frozen)
        with self.assertRaisesRegex(ValueError, "Protocol/source changed"):
            self.run_grade(grader)
        grader.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_evaluator_version_drift_is_rejected_before_grading(self):
        self.freeze()
        grader = Mock(side_effect=AssertionError("Changed evaluator cannot be used"))
        with self.assertRaisesRegex(ValueError, "versions differ"):
            self.run_grade(grader, versions={"fake-evaluator": "different-version"})
        grader.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_incomplete_original_grading_is_rejected_even_with_matching_hash(self):
        original = supplement.read(self.pilot / "grading.json")
        original["completed"] = False
        write_json(self.pilot / "grading.json", original)
        self.freeze()
        grader = Mock(side_effect=AssertionError("Incomplete original grading cannot be supplemented"))
        with self.assertRaisesRegex(ValueError, "Incomplete original grading"):
            self.run_grade(grader)
        grader.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_input_or_freeze_changed_during_grading_prevents_success_artifacts(self):
        for changed_name in ("original", "freeze"):
            with self.subTest(changed_name=changed_name):
                # Give each run an independently frozen directory; no evidence is restored in place.
                prepared = self.base / ("prepared-" + changed_name)
                output = self.base / ("supplement-" + changed_name)
                with patch.object(supplement, "source_hashes", return_value=self.sources):
                    supplement.prepare(self.pilot, prepared)
                target = self.pilot / "analysis-input.json" if changed_name == "original" else prepared / "boundary-decisions.json"
                def mutate_then_grade(payload):
                    target.write_bytes(target.read_bytes() + b"\n")
                    return fake_grader(payload)
                with patch.object(supplement, "source_hashes", return_value=self.sources), \
                        patch.object(math_grading, "_check_dependencies", return_value=self.versions), \
                        patch.object(math_grading, "grade_record", side_effect=mutate_then_grade):
                    with self.assertRaisesRegex(ValueError, "Evidence changed while grading"):
                        supplement.grade(prepared, output)
                self.assertFalse((output / "grading.json").exists())
                self.assertFalse((output / "analysis-input.json").exists())
                self.assertFalse((output / "summary.json").exists())

    def test_existing_output_directories_are_not_overwritten(self):
        self.freeze()
        saved = (self.prepared / "boundary-decisions.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.freeze()
        self.assertEqual((self.prepared / "boundary-decisions.json").read_bytes(), saved)
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_text("prior supplement", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.run_grade()
        self.assertEqual(sentinel.read_text(), "prior supplement")
        self.assertEqual(list(self.output.iterdir()), [sentinel])
        self.assert_original_bytes_unchanged()


class SupplementalEntryPointTests(unittest.TestCase):
    def test_standalone_help_works_without_test_path_injections(self):
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        result = subprocess.run([sys.executable, str(ROOT / "scripts/supplement_entropy_boundaries.py"), "--help"],
                                cwd=tempfile.gettempdir(), env=environment, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("prepare", result.stdout)
        self.assertIn("grade", result.stdout)


if __name__ == "__main__":
    unittest.main()
