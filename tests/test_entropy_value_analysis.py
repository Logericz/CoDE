"""Synthetic CPU checks for question-level validation and value arithmetic."""
from pathlib import Path
from copy import deepcopy
import json
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.entropy_value_analysis import (
    analyze, curve_summary, feature_views, fit_ridge, grouped_folds,
    matched_budget, nested_predictions, observation_targets, prepare_records,
    question_bootstrap, regression_metrics,
)


def fixture(questions=20, candidates=3):
    """Synthetic measurements only; these rows contain no experiment evidence."""
    ids = [f"q{i:02}" for i in range(questions)]
    records = []
    for i, qid in enumerate(ids):
        for j in range(1, candidates + 1):
            p = [float(.55 + .12 * np.sin((i + j + k) / 7)) for k in range(42)]
            h = [float(1.0 + .2 * np.cos((i + 2 * j + k) / 9)) for k in range(42)]
            short = {"confidence": float(np.exp(np.log(p[:21]).mean())), "D": float(j - 1),
                     "ended_with_think": False, "max_probabilities": p[:21], "entropies_nats": h[:21]}
            long = {"confidence": float(np.exp(np.log(p).mean())), "D": float(j - 1),
                    "ended_with_think": True, "max_probabilities": p, "entropies_nats": h}
            records.append({"question_id": qid, "candidate_index": j, "status": "ok",
                            "prefix_tokens": 200 * j + i,
                            "history": {"valid_count": j - 1, "previous_confidence": .5 if j > 1 else None,
                                        "previous_D": 0.0 if j > 1 else None, "candidate_gap": 1, "token_gap": 200},
                            "short": short, "long": long,
                            "actions": {"E": {"remaining_ms": 500. + i, "error": int(i % 3 == 0)},
                                        "T": {"remaining_ms": 3000. + j, "error": int(i % 4 == 0)}},
                            "extra_probe_ms": 150. + j})
    return {"schema_version": "entropy-value-v1", "question_ids": ids, "records": records}


class AnalysisPrimitiveTests(unittest.TestCase):
    def test_folds_are_question_level_order_independent_and_include_empty_questions(self):
        ids = [f"q{i:02}" for i in range(20)]
        result = grouped_folds(ids)
        self.assertEqual(result, grouped_folds(list(reversed(ids)) + [ids[0]]))
        self.assertEqual(set(result), set(ids))
        self.assertEqual([list(result.values()).count(i) for i in range(4)], [5] * 4)

    def test_probability_curve_features_capture_order_hidden_by_mean(self):
        rising, falling = curve_summary([.2, .4, .6, .8]), curve_summary([.8, .6, .4, .2])
        self.assertAlmostEqual(rising["mean"], falling["mean"])
        self.assertGreater(rising["slope"], 0)
        self.assertLess(falling["slope"], 0)
        self.assertGreater(falling["largest_decrease"], 0)

    def test_ridge_imputation_and_standardization_use_only_training_rows(self):
        x = np.array([[0., np.nan], [1., 4.], [2., 6.], [3., 8.]])
        model = fit_ridge(x, np.array([0., 1., 2., 3.]))
        self.assertTrue(np.allclose(model.feature_mean, [1.5, 6.]))
        before = model.feature_mean.copy()
        prediction = model.predict(np.array([[1e6, np.nan]]))
        self.assertTrue(np.isfinite(prediction).all())
        self.assertTrue(np.array_equal(model.feature_mean, before))

    def test_unknown_target_is_rejected_not_replaced_by_zero(self):
        with self.assertRaises(ValueError):
            fit_ridge(np.ones((2, 1)), np.array([1., np.nan]))

    def test_metrics_use_ms_and_squared_ms(self):
        result = regression_metrics([10., 20.], [12., 16.])
        self.assertEqual(result["mae_ms"], 3.)
        self.assertEqual(result["mse_ms2"], 10.)


class ProtocolTests(unittest.TestCase):
    def test_entropy_and_future_answers_cannot_enter_B_features(self):
        record = fixture(1, 1)["records"][0]
        original = feature_views(record)
        changed = deepcopy(record)
        changed["short"]["entropies_nats"] = [value + 2 for value in changed["short"]["entropies_nats"]]
        changed["actions"]["E"]["error"] = 1 - changed["actions"]["E"]["error"]
        changed["extra_probe_ms"] = 999999
        after = feature_views(changed)
        self.assertEqual(sorted(original[0]), sorted(after[0]))
        np.testing.assert_allclose(list(original[0].values()), list(after[0].values()), equal_nan=True)
        self.assertNotEqual(original[1]["short_entropy_mean"], after[1]["short_entropy_mean"])
        self.assertTrue(set(original[0]) <= set(original[1]) <= set(original[2]))
        self.assertFalse(any("entropy" in name for name in original[0]))
        self.assertFalse(any("tail_mass_entropy" in name for view in original for name in view))

    def test_unknown_grade_is_counted_and_excluded_without_changing_roster(self):
        document = fixture(20, 1)
        document["records"][0]["actions"]["E"]["error"] = None
        document["records"][1] = {"question_id": "q01", "status": "no_candidate"}
        roster, rows, coverage = prepare_records(document)
        self.assertEqual(len(roster), 20)
        self.assertEqual(len(rows), 18)
        self.assertEqual(coverage["excluded_reason_counts"]["unknown_or_invalid_answer_grade"], 1)
        self.assertEqual(coverage["planned_questions_without_valid_pairs"], ["q00", "q01"])

    def test_long_curve_must_extend_short_curve(self):
        document = fixture(1, 1)
        document["records"][0]["long"]["max_probabilities"][0] += .01
        with self.assertRaisesRegex(ValueError, "exact short"):
            prepare_records(document)

    def test_saved_prefix_rejects_even_tiny_numerical_drift(self):
        document = fixture(1, 1)
        document["records"][0]["long"]["entropies_nats"][0] += 1e-12
        with self.assertRaisesRegex(ValueError, "exact short"):
            prepare_records(document)

    def test_completed_short_does_not_create_fake_information_value(self):
        document = fixture(1, 1)
        row = document["records"][0]
        row["short"]["ended_with_think"] = True
        row["long"] = deepcopy(row["short"])
        row["extra_probe_ms"] = 0.
        _, records, coverage = prepare_records(document)
        self.assertEqual(len(records), 1)  # It can still train an E/T selector.
        self.assertFalse(records[0]["extension_eligible"])
        self.assertEqual(coverage["extension_eligible_records"], 0)

    def test_targets_evaluate_predicted_actions_even_when_long_choice_is_wrong(self):
        _, rows, _ = prepare_records(fixture(1, 1))
        row = rows[0]
        row.update(E_ms=10., T_ms=100., E_error=0, T_error=1, extra_probe_ms=5.)
        # Short predicts E, long predicts T.  The long choice must lose 60095ms;
        # selecting the lower gold loss here would invent an oracle benefit.
        result = observation_targets([row], [-1.], [1.], 60000.)[0]
        self.assertEqual(result["z_ms"], -60095.)
        self.assertEqual(result["short_action"], "E")
        self.assertEqual(result["long_action"], "T")
        same = observation_targets([row], [0.], [0.], 60000.)[0]
        self.assertEqual(same["z_ms"], -5.)

    def test_outer_test_labels_do_not_change_that_folds_predictions(self):
        roster, rows, _ = prepare_records(fixture())
        before, audits = nested_predictions(roster, rows, 60000.)
        held_ids = set(audits[0]["test_question_ids"])
        changed = deepcopy(rows)
        for row in changed:
            if row["question_id"] in held_ids:
                row["E_error"] = 1 - row["E_error"]
                row["T_ms"] += 123456.
        after, _ = nested_predictions(roster, changed, 60000.)
        before = [p for p in before if p["outer_fold"] == 0]
        after = [p for p in after if p["outer_fold"] == 0]
        self.assertEqual(len(before), 15)
        for left, right in zip(before, after):
            for key in ("B_prediction_ms", "H_prediction_ms", "constant_prediction_ms",
                        "short_delta_ms", "long_delta_ms"):
                self.assertAlmostEqual(left[key], right[key])
        for audit in audits:
            self.assertTrue(set(audit["train_question_ids"]).isdisjoint(audit["test_question_ids"]))
            for inner in audit["inner_folds"]:
                self.assertTrue(set(inner["g_train_question_ids"]).isdisjoint(inner["z_question_ids"]))
                self.assertTrue(set(inner["g_train_question_ids"]).isdisjoint(audit["test_question_ids"]))

    def test_budget_ranking_uses_predictions_not_observed_value(self):
        rows = [{"question_id": f"q{i}", "candidate_index": 1, "z_ms": z,
                 "B_prediction_ms": b, "H_prediction_ms": h}
                for i, (z, b, h) in enumerate(((100., 1., 3.), (-100., 3., 1.), (1., 2., 2.), (9999., -1., -1.)))]
        result = matched_budget(rows, .25)
        self.assertEqual(result["matched_count"], 1)
        self.assertEqual(result["budget_unit"], "additional_probe_count")
        self.assertFalse(result["elapsed_ms_matched"])
        self.assertEqual(result["B"]["selected_ids"], [["q1", 1]])
        self.assertEqual(result["H"]["selected_ids"], [["q0", 1]])
        self.assertEqual(result["H_minus_B_net_utility_ms"], 200.)

    def test_small_sample_reports_not_estimable_and_sensitivities_are_labeled(self):
        result = analyze(fixture(2, 1), bootstrap_replicates=10)
        self.assertEqual(result["coverage"]["planned_questions"], 2)
        self.assertEqual([item["role"] for item in result["results"]], ["primary", "sensitivity_only", "sensitivity_only"])
        self.assertTrue(all(item["summary"]["status"] == "not_estimable" for item in result["results"]))
        json.dumps(result, allow_nan=False)

    def test_bootstrap_clusters_questions_and_needs_four_groups(self):
        rows = [{"question_id": f"q{i}", "candidate_index": j, "z_ms": float(i),
                 "B_prediction_ms": 1., "H_prediction_ms": 2., "constant_prediction_ms": 0.}
                for i in range(4) for j in range(1, 4)]
        result = question_bootstrap(rows, 20)
        self.assertEqual(result["resampling_unit"], "question")
        self.assertEqual(result["replicates"], 20)
        self.assertEqual(result, question_bootstrap(rows, 20))
        self.assertEqual(question_bootstrap(rows[:9], 20)["status"], "not_estimable")

    def test_cli_preserves_existing_output_and_records_input_identity(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        from analyze_entropy_value import main
        with tempfile.TemporaryDirectory() as temporary:
            source, output = Path(temporary) / "input.json", Path(temporary) / "analysis.json"
            source.write_text(json.dumps(fixture(2, 1)), encoding="utf-8")
            self.assertEqual(main(["--input", str(source), "--output", str(output)]), 0)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(result["input_sha256"]), 64)
            before = output.read_bytes()
            with self.assertRaises(FileExistsError):
                main(["--input", str(source), "--output", str(output)])
            self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
