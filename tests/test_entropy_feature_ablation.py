"""Synthetic CPU checks for the fixed g-by-f comparison and frozen targets."""
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from scripts import analyze_entropy_compact as cli
from src import entropy_feature_ablation as ablation
from src import entropy_value_analysis as original
from test_entropy_value_analysis import fixture


class FixedAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = fixture(questions=20, candidates=2)
        cls.roster, cls.rows, _ = original.prepare_records(cls.document)

    def test_original_and_compact_roles_route_to_their_separate_feature_views(self):
        row = self.rows[0]
        for role, source in (("f_B", "B"), ("f_H", "H"), ("g_short", "H"), ("g_long", "long_H")):
            self.assertIs(ablation.view(row, role, "original"), row[source])
        expected_dimensions = {"f_B": 5, "f_H": 7, "g_short": 9, "g_long": 15}
        for role, size in expected_dimensions.items():
            with self.subTest(role=role):
                self.assertEqual(ablation.matrix(self.rows[:3], role, "compact").shape, (3, size))
        for role, kind in (("unknown", "compact"), ("f_B", "tuned")):
            with self.assertRaises(ValueError):
                ablation.view(row, role, kind)

    def test_each_g_lambda_builds_one_cache_shared_by_both_f_families(self):
        with patch.object(ablation, "build_target_cache", wraps=ablation.build_target_cache) as builder:
            report = ablation.analyze_ablation(self.document, bootstrap_replicates=8)
        self.assertEqual(builder.call_count, 6)
        self.assertEqual([(call.args[2], call.args[3]) for call in builder.call_args_list],
                         [(g, penalty) for g in ablation.KINDS for penalty in original.ERROR_COSTS_MS])
        self.assertEqual(len(report["target_caches"]), 6)
        self.assertEqual(len(report["cells"]), 12)
        for cache in report["target_caches"]:
            cells = [c for c in report["cells"] if c["g_features"] == cache["g_features"]
                     and c["lambda_error_ms"] == cache["lambda_error_ms"]]
            self.assertEqual({c["f_features"] for c in cells}, {"original", "compact"})
            self.assertEqual({c["target_cache_sha256"] for c in cells}, {ablation.digest(cache)})
            keys = ("row", "question_id", "candidate_index", "outer_fold", "z_ms", "short_action",
                    "long_action", "short_delta_ms", "long_delta_ms", "loss_E_ms", "loss_T_ms",
                    "constant_prediction_ms", "extra_probe_ms")
            self.assertEqual([[tuple(p[k] for k in keys) for p in c["held_out_predictions"]] for c in cells][0],
                             [[tuple(p[k] for k in keys) for p in c["held_out_predictions"]] for c in cells][1])

    def test_value_prediction_never_refits_g_or_changes_its_target_cache(self):
        for kind in ablation.KINDS:
            with self.subTest(g_kind=kind):
                cache = ablation.build_target_cache(self.roster, self.rows, kind, 60000)
                before = deepcopy(cache)
                with patch.object(ablation, "fit_actions", side_effect=AssertionError("f must reuse g targets")):
                    for f_kind in ablation.KINDS:
                        predictions = ablation.predict_values(self.rows, cache, f_kind)
                        self.assertEqual(len(predictions), len(self.rows))
                self.assertEqual(cache, before)

    def test_outer_test_outcomes_cannot_change_g_or_f_predictions_for_that_fold(self):
        outer = original.grouped_folds(self.roster)
        changed = deepcopy(self.rows)
        for row in changed:
            if outer[row["question_id"]] == 0:
                row["E_error"] = 1 - row["E_error"]
                row["T_error"] = 1 - row["T_error"]
                row["E_ms"] += 12345
                row["T_ms"] += 54321
                row["extra_probe_ms"] += 91
        for g_kind in ablation.KINDS:
            with self.subTest(g_kind=g_kind):
                before = ablation.build_target_cache(self.roster, self.rows, g_kind, 60000)
                after = ablation.build_target_cache(self.roster, changed, g_kind, 60000)
                self.assertEqual(before["folds"][0]["training_targets"], after["folds"][0]["training_targets"])
                for f_kind in ablation.KINDS:
                    old = [r for r in ablation.predict_values(self.rows, before, f_kind) if r["outer_fold"] == 0]
                    new = [r for r in ablation.predict_values(changed, after, f_kind) if r["outer_fold"] == 0]
                    self.assertEqual(len(old), 10)  # Five held-out questions, two candidates each.
                    for left, right in zip(old, new):
                        for key in ("question_id", "candidate_index", "short_delta_ms", "long_delta_ms",
                                    "short_action", "long_action", "B_prediction_ms", "H_prediction_ms",
                                    "constant_prediction_ms"):
                            self.assertEqual(left[key], right[key], (g_kind, f_kind, key))
                    self.assertTrue(any(left["z_ms"] != right["z_ms"] for left, right in zip(old, new)))

    def test_cross_fitted_g_scores_do_not_use_inner_held_outcomes(self):
        held_ids = set(self.roster[:4])
        train = [r for r in self.rows if r["question_id"] not in held_ids]
        held = [r for r in self.rows if r["question_id"] in held_ids]
        changed = deepcopy(held)
        for row in changed:
            row["E_error"] = 1 - row["E_error"]
            row["T_error"] = 1 - row["T_error"]
            row["E_ms"] += 9876
            row["extra_probe_ms"] += 17
        for kind in ablation.KINDS:
            old = ablation.scored_targets(train, held, kind, 60000)
            new = ablation.scored_targets(train, changed, kind, 60000)
            for left, right in zip(old, new):
                for key in ("short_delta_ms", "long_delta_ms", "short_action", "long_action", "constant_delta_ms"):
                    self.assertEqual(left[key], right[key])
            self.assertTrue(any(left["z_ms"] != right["z_ms"] for left, right in zip(old, new)))

    def test_original_g_original_f_reproduces_original_nested_predictions(self):
        cells, reference = [], {"results": []}
        for penalty in original.ERROR_COSTS_MS:
            expected, _ = original.nested_predictions(self.roster, self.rows, penalty)
            cache = ablation.build_target_cache(self.roster, self.rows, "original", penalty)
            actual = ablation.predict_values(self.rows, cache, "original")
            cells.append({"g_features": "original", "f_features": "original", "lambda_error_ms": penalty,
                          "held_out_predictions": actual})
            reference["results"].append({"lambda_error_ms": penalty, "held_out_predictions": expected})
        checked = ablation.check_reference(cells, reference)
        self.assertTrue(checked["passed"])
        self.assertGreater(checked["field_checks"], 1000)
        self.assertLess(checked["max_absolute_numeric_difference"], 1e-6)

    def test_reference_check_rejects_coverage_identity_and_score_changes(self):
        reference = {"results": [{"lambda_error_ms": penalty,
                      "held_out_predictions": [{"question_id": "q", "candidate_index": 1, "z_ms": 2.}]}
                     for penalty in original.ERROR_COSTS_MS]}
        cells = [{"g_features": "original", "f_features": "original", "lambda_error_ms": penalty,
                  "held_out_predictions": deepcopy(reference["results"][0]["held_out_predictions"])}
                 for penalty in original.ERROR_COSTS_MS]
        for field, value in (("question_id", "other"), ("candidate_index", 2), ("z_ms", 3.)):
            changed = deepcopy(cells)
            changed[0]["held_out_predictions"][0][field] = value
            with self.assertRaises(ValueError):
                ablation.check_reference(changed, reference)
        cells[0]["held_out_predictions"] = []
        with self.assertRaisesRegex(ValueError, "coverage changed"):
            ablation.check_reference(cells, reference)

    def test_reference_requires_all_three_unique_fixed_penalties(self):
        for penalties in ([], [60000], [60000, 30000, 30000], [60000, 30000, 90000]):
            reference = {"results": [{"lambda_error_ms": value} for value in penalties]}
            with self.assertRaisesRegex(ValueError, "three unique fixed"):
                ablation.check_reference([], reference)

    def test_unknown_and_nonextended_pairs_have_identical_coverage_in_all_cells(self):
        document = deepcopy(self.document)
        for row in document["records"]:
            if row["question_id"] == document["question_ids"][0]:
                row["actions"]["E"]["error"] = None
        document["records"][2]["long"] = deepcopy(document["records"][2]["short"])
        report = ablation.analyze_ablation(document, bootstrap_replicates=1)
        self.assertEqual(report["coverage"]["valid_paired_records"], 38)
        self.assertEqual(report["coverage"]["extension_eligible_records"], 37)
        self.assertEqual(report["coverage"]["excluded_reason_counts"], {"unknown_or_invalid_answer_grade": 2})
        identities = [{(r["question_id"], r["candidate_index"]) for r in c["held_out_predictions"]}
                      for c in report["cells"]]
        self.assertTrue(all(ids == identities[0] for ids in identities))
        self.assertEqual(len(identities[0]), 37)
        self.assertNotIn((document["records"][2]["question_id"], 1), identities[0])

    def test_insufficient_questions_remain_not_estimable_for_every_cell(self):
        report = ablation.analyze_ablation(fixture(questions=4, candidates=1), bootstrap_replicates=1)
        self.assertEqual(len(report["cells"]), 12)
        for cell in report["cells"]:
            self.assertEqual(cell["summary"]["status"], "not_estimable")
            self.assertEqual(cell["held_out_predictions"], [])
            for policy in cell["pipeline_policies"].values():
                self.assertEqual(policy["status"], "not_estimable")
                self.assertIsNone(policy["total_loss_ms"])
        for cache in report["target_caches"]:
            self.assertTrue(all(fold["status"].startswith("not_estimable") for fold in cache["folds"]))

    def test_policy_costs_include_only_selected_additional_probes(self):
        rows = [{"row": 0, "question_id": "q0", "candidate_index": 1,
                 "E_error": 0, "T_error": 0, "E_ms": 100., "T_ms": 200., "extra_probe_ms": 10.},
                {"row": 1, "question_id": "q1", "candidate_index": 1,
                 "E_error": 1, "T_error": 0, "E_ms": 100., "T_ms": 300., "extra_probe_ms": 20.}]
        targets = [{"row": 0, "question_id": "q0", "candidate_index": 1, "loss_E_ms": 100., "loss_T_ms": 200.},
                   {"row": 1, "question_id": "q1", "candidate_index": 1, "loss_E_ms": 60100., "loss_T_ms": 300.}]
        result = ablation.decision_metrics(rows, targets, ["E", "T"], [False, True])
        self.assertEqual(result["correct"], 2)
        self.assertEqual(result["remaining_ms"], 400.)
        self.assertEqual(result["additional_probes"], 1)
        self.assertEqual(result["extra_probe_ms"], 20.)
        self.assertEqual(result["total_loss_ms"], 420.)
        self.assertEqual(result["regret_to_per_pair_oracle_ms"], 20.)
        self.assertEqual(result["optimal_action_agreements"], 2)
        self.assertEqual(result["avoidable_answer_errors"], 0)
        wrong = ablation.decision_metrics(rows, targets, ["E", "E"], [False, False])
        self.assertEqual(wrong["avoidable_answer_errors"], 1)
        with self.assertRaisesRegex(ValueError, "align"):
            ablation.decision_metrics(rows, targets, ["E"], [False, False])
        changed = deepcopy(targets)
        changed[1]["question_id"] = "unrelated"
        with self.assertRaisesRegex(ValueError, "identity"):
            ablation.decision_metrics(rows, changed, ["E", "E"], [False, False])

    def test_cell_detects_mutation_of_shared_cache(self):
        cache = ablation.build_target_cache(self.roster, self.rows, "compact", 60000)
        def bad_predictor(rows, cache, kind):
            cache["folds"][0]["training_targets"][0]["z_ms"] += 1
            return []
        with patch.object(ablation, "predict_values", side_effect=bad_predictor):
            with self.assertRaisesRegex(AssertionError, "mutated"):
                ablation.evaluate_cell(self.rows, cache, "compact", 1)


class FixedAblationCliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.input, self.reference, self.output = (self.base / name for name in ("input.json", "reference.json", "out"))
        self.input.write_text(json.dumps(fixture(questions=4, candidates=1)), encoding="utf-8")
        self.reference.write_text(json.dumps({"input_sha256": hashlib.sha256(self.input.read_bytes()).hexdigest(),
            "source_sha256": {"src/entropy_value_analysis.py": hashlib.sha256((ROOT / "src/entropy_value_analysis.py").read_bytes()).hexdigest()}}), encoding="utf-8")
        self.arguments = ["--input", str(self.input), "--reference-analysis", str(self.reference),
                          "--output-dir", str(self.output)]

    def invoke(self, analyzer):
        with patch.object(cli, "SOURCES", ()), patch.object(cli, "analyze_ablation", side_effect=analyzer), \
                redirect_stdout(io.StringIO()):
            return cli.main(self.arguments)

    def test_cli_rejects_reference_for_a_different_input_before_analysis(self):
        self.reference.write_text(json.dumps({"input_sha256": "0" * 64}), encoding="utf-8")
        analyzer = Mock(side_effect=AssertionError("Mismatched input must not run"))
        with self.assertRaisesRegex(ValueError, "different input"):
            self.invoke(analyzer)
        analyzer.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_cli_preserves_existing_result_without_running_analysis(self):
        self.output.mkdir()
        result = self.output / "feature-ablation.json"
        result.write_text("preserve prior evidence", encoding="utf-8")
        analyzer = Mock(side_effect=AssertionError("Existing result must not rerun"))
        with self.assertRaises(FileExistsError):
            self.invoke(analyzer)
        analyzer.assert_not_called()
        self.assertEqual(result.read_text(), "preserve prior evidence")
        self.assertFalse((self.output / "feature-plan.json").exists())

    def test_cli_rejects_changed_original_analyzer_source(self):
        reference = json.loads(self.reference.read_text())
        reference["source_sha256"]["src/entropy_value_analysis.py"] = "0" * 64
        self.reference.write_text(json.dumps(reference), encoding="utf-8")
        analyzer = Mock(side_effect=AssertionError("Changed source must not run"))
        with self.assertRaisesRegex(ValueError, "original analyzer has changed"):
            self.invoke(analyzer)
        analyzer.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_cli_locks_protocol_before_analysis_and_preserves_input_bytes(self):
        before = (self.input.read_bytes(), self.reference.read_bytes())
        def analyzer(document, reference):
            plan = json.loads((self.output / "feature-plan.json").read_text())
            self.assertEqual(plan["g_families"], ["original", "compact"])
            self.assertEqual(plan["f_families"], ["original", "compact"])
            self.assertEqual(plan["lambda_error_ms"], [60000, 30000, 120000])
            self.assertEqual(plan["ridge_alpha"], 1.0)
            self.assertEqual(plan["minimum_training_questions"], 4)
            self.assertEqual(plan["bootstrap_replicates"], 2000)
            self.assertTrue(plan["no_parameter_search"])
            self.assertTrue(plan["no_generation"])
            return {"cells": [None] * 12, "reference_reproduction": {"passed": True}}
        self.assertEqual(self.invoke(analyzer), 0)
        result = json.loads((self.output / "feature-ablation.json").read_text())
        self.assertEqual(result["plan_sha256"], hashlib.sha256((self.output / "feature-plan.json").read_bytes()).hexdigest())
        self.assertEqual((self.input.read_bytes(), self.reference.read_bytes()), before)

    def test_cli_rejects_input_changes_during_analysis_without_success_result(self):
        def analyzer(document, reference):
            self.input.write_bytes(self.input.read_bytes() + b"\n")
            return {"cells": [], "reference_reproduction": {"passed": True}}
        with self.assertRaisesRegex(ValueError, "Input changed"):
            self.invoke(analyzer)
        self.assertTrue((self.output / "feature-plan.json").exists())
        self.assertFalse((self.output / "feature-ablation.json").exists())

    def test_cli_rejects_frozen_plan_changes_during_analysis(self):
        def analyzer(document, reference):
            plan = self.output / "feature-plan.json"
            plan.write_bytes(plan.read_bytes() + b"\n")
            return {"cells": [], "reference_reproduction": {"passed": True}}
        with self.assertRaisesRegex(ValueError, "Frozen feature plan changed"):
            self.invoke(analyzer)
        self.assertFalse((self.output / "feature-ablation.json").exists())


if __name__ == "__main__":
    unittest.main()
