"""Synthetic CPU checks for the fixed short-observation fusion comparison."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from src import entropy_short_fusion as fusion
from src import entropy_feature_ablation as previous
from src import entropy_value_analysis as original
from test_entropy_value_analysis import fixture


def fusion_fixture(questions=20, candidates=2):
    document = fixture(questions=questions, candidates=candidates)
    for index, record in enumerate(document["records"]):
        record["short_probe_ms"] = 700. + index
    return document


def saved_decision(question, candidate, e_error, t_error, e_ms, t_ms, c_action, ch_action, h_action, short_ms):
    scores = {name: 0. for name in ("Z", "C", "H", "CH", "trainmean", "late_avg")}
    actions = {name: "E" for name in fusion.METHODS}
    actions.update(C=c_action, CH=ch_action, H=h_action, always_T="T")
    loss_e, loss_t = e_ms + 60000 * e_error, t_ms + 60000 * t_error
    return {"question_id": question, "candidate_index": candidate, "E_error": e_error, "T_error": t_error,
            "E_ms": e_ms, "T_ms": t_ms, "short_probe_ms": short_ms, "loss_E_ms": loss_e,
            "loss_T_ms": loss_t, "delta_ms": loss_e - loss_t, "scores": scores, "actions": actions}


class FusionDecisionRuleTests(unittest.TestCase):
    @staticmethod
    def scores(c, h):
        return {"Z": 0.0, "C": c, "H": h, "CH": 2.0, "trainmean": -2.0}

    def test_fixed_arm_set_and_base_score_signs(self):
        actions = fusion.combine_scores(self.scores(3., -1.))
        self.assertEqual(set(actions), {"Z", "C", "H", "CH", "trainmean", "always_E", "always_T",
                                        "late_avg", "late_agree", "late_either"})
        self.assertEqual({name: actions[name] for name in ("Z", "C", "H", "CH", "trainmean", "always_E", "always_T")},
                         {"Z": "E", "C": "T", "H": "E", "CH": "T", "trainmean": "E",
                          "always_E": "E", "always_T": "T"})

    def test_all_exact_zero_scores_choose_E_except_always_T(self):
        actions = fusion.combine_scores({name: 0. for name in ("Z", "C", "H", "CH", "trainmean")})
        self.assertEqual({name for name, action in actions.items() if action == "T"}, {"always_T"})

    def test_disagreement_distinguishes_average_and_exit_rules(self):
        for c, h, expected_average in ((3., -1., "T"), (-3., 1., "E"), (1., -1., "E")):
            with self.subTest(C=c, H=h):
                actions = fusion.combine_scores(self.scores(c, h))
                self.assertEqual(actions["late_avg"], expected_average)
                self.assertEqual(actions["late_agree"], "T")
                self.assertEqual(actions["late_either"], "E")

    def test_zero_is_an_exit_vote_in_agreement_rules(self):
        for c, h in ((0., 1.), (1., 0.)):
            actions = fusion.combine_scores(self.scores(c, h))
            self.assertEqual(actions["late_agree"], "T")
            self.assertEqual(actions["late_either"], "E")
        for c, h in ((0., -1.), (-1., 0.)):
            actions = fusion.combine_scores(self.scores(c, h))
            self.assertEqual(actions["late_agree"], "E")
            self.assertEqual(actions["late_either"], "E")

    def test_consensus_gives_same_action_for_all_three_fusion_rules(self):
        for c, h, expected in ((1., 2., "T"), (-1., -2., "E")):
            actions = fusion.combine_scores(self.scores(c, h))
            self.assertEqual([actions[name] for name in ("late_avg", "late_agree", "late_either")], [expected] * 3)

    def test_nonfinite_scores_are_rejected_instead_of_becoming_exit_votes(self):
        for name in ("Z", "C", "H", "CH", "trainmean"):
            for value in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(name=name, value=value):
                    scores = self.scores(1., 2.)
                    scores[name] = value
                    with self.assertRaisesRegex(ValueError, "finite"):
                        fusion.combine_scores(scores)


class ShortOnlyProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = fusion_fixture()
        cls.roster, cls.rows, _ = fusion.prepare(cls.document)
        cls.report = fusion.analyze(cls.document, bootstrap_replicates=0)

    def test_legal_long_probe_length_and_extra_cost_do_not_change_primary_or_predictions(self):
        changed = deepcopy(self.document)
        for record in changed["records"]:
            # Both forms remain valid archived paired artifacts.  This is not a
            # contract to accept malformed long evidence through the old parser.
            record["long"] = deepcopy(record["short"])
            record["extra_probe_ms"] = 99999.
        result = fusion.analyze(changed, bootstrap_replicates=0)
        self.assertEqual(result["primary_records"], self.report["primary_records"])
        for old, new in zip(self.report["results"], result["results"]):
            self.assertEqual(old["predictions"], new["predictions"])
            self.assertEqual(old["scopes"]["primary"], new["scopes"]["primary"])

    def test_primary_uses_short_length_and_termination_and_all_includes_other_states(self):
        document = deepcopy(self.document)
        document["records"][0]["short"]["ended_with_think"] = True
        for field in ("max_probabilities", "entropies_nats"):
            document["records"][1]["short"][field] = document["records"][1]["short"][field][:7]
        result = fusion.analyze(document, bootstrap_replicates=0)
        self.assertEqual(result["primary_records"], 38)
        for setting in result["results"]:
            self.assertEqual(setting["scopes"]["primary"]["methods"]["C"]["records"], 38)
            self.assertEqual(setting["scopes"]["all"]["methods"]["C"]["records"], 40)
            primary_rows = {r["row"] for r in setting["predictions"] if r["primary_eligible"]}
            self.assertNotIn(0, primary_rows)
            self.assertNotIn(1, primary_rows)
            self.assertEqual(set(setting["scopes"]), {"primary", "all", "other_short"})
            self.assertEqual(setting["scopes"]["other_short"]["methods"]["C"]["records"], 2)

    def test_predict_fold_needs_no_future_long_view_or_extension_cost(self):
        train, held = self.rows[:30], self.rows[30:]
        expected = fusion.predict_fold(train, held, 60000)
        short_train, short_held = deepcopy(train), deepcopy(held)
        for row in short_train + short_held:
            row.pop("long_H")
            row.pop("extra_probe_ms")
            row.pop("extension_eligible")
        self.assertEqual(fusion.predict_fold(short_train, short_held, 60000), expected)

    def test_outer_test_answers_and_costs_do_not_leak_into_scores_or_actions(self):
        folds = self.report["outer_question_folds"]
        changed = deepcopy(self.document)
        for record in changed["records"]:
            if folds[record["question_id"]] == 0:
                for action in ("E", "T"):
                    record["actions"][action]["error"] = 1 - record["actions"][action]["error"]
                    record["actions"][action]["remaining_ms"] += 30000 if action == "E" else 5000
                record["short_probe_ms"] += 50
        result = fusion.analyze(changed, bootstrap_replicates=0)
        for old_setting, new_setting in zip(self.report["results"], result["results"]):
            old = [r for r in old_setting["predictions"] if r["outer_fold"] == 0]
            new = [r for r in new_setting["predictions"] if r["outer_fold"] == 0]
            self.assertEqual(len(old), 10)
            self.assertEqual({r["question_id"] for r in old}, {q for q, fold in folds.items() if fold == 0})
            for before, after in zip(old, new):
                self.assertEqual(before["scores"], after["scores"])
                self.assertEqual(before["actions"], after["actions"])
                self.assertEqual(before["primary_eligible"], after["primary_eligible"])
            self.assertTrue(any(before["delta_ms"] != after["delta_ms"] for before, after in zip(old, new)))
        for audit in result["results"][0]["fold_audit"]:
            self.assertFalse(set(audit["train_question_ids"]) & set(audit["test_question_ids"]))

    def test_one_extreme_held_feature_cannot_change_another_held_predictions(self):
        train, held = self.rows[:30], self.rows[30:]
        original_predictions = fusion.predict_fold(train, held, 60000)
        changed = deepcopy(held)
        for view in ("B", "H"):
            changed[0][view]["prefix_tokens"] = 1e12
        predictions = fusion.predict_fold(train, changed, 60000)
        self.assertEqual(predictions[1:], original_predictions[1:])
        self.assertNotEqual(predictions[0]["scores"], original_predictions[0]["scores"])

    def test_CH_reproduces_previous_compact_short_selector_for_all_fixed_penalties(self):
        reference = {"target_caches": [previous.build_target_cache(self.roster, self.rows, "compact", penalty)
                                        for penalty in original.ERROR_COSTS_MS]}
        checked = fusion.check_reference(self.report["results"], reference)
        self.assertTrue(checked["passed"])
        self.assertEqual(checked["score_checks"], 120)
        self.assertLess(checked["max_absolute_score_difference_ms"], 1e-6)
        tampered = deepcopy(self.report["results"])
        tampered[0]["predictions"][0]["scores"]["CH"] += 1.
        with self.assertRaisesRegex(ValueError, "short score"):
            fusion.check_reference(tampered, reference)

    def test_only_four_learned_models_per_fold_and_rules_reuse_the_same_scores(self):
        with patch.object(original, "fit_ridge", wraps=original.fit_ridge) as fitter:
            result = fusion.analyze(self.document, bootstrap_replicates=0)
        self.assertEqual(fitter.call_count, 4 * 4 * 3)
        self.assertTrue(all(call.args[2] == 1. for call in fitter.call_args_list))
        for setting in result["results"]:
            self.assertEqual(len(setting["predictions"]), 40)
            for row in setting["predictions"]:
                self.assertEqual(row["actions"], fusion.combine_scores(row["scores"]))
                self.assertEqual(row["scores"]["late_avg"], (row["scores"]["C"] + row["scores"]["H"]) / 2)

    def test_unknown_grade_invalid_short_and_unknown_action_cost_are_excluded(self):
        document = deepcopy(self.document)
        for row in document["records"][:2]:
            row["actions"]["E"]["error"] = None
        document["records"][2]["short"]["confidence"] = float("nan")
        document["records"][3]["actions"]["T"]["remaining_ms"] = None
        report = fusion.analyze(document, bootstrap_replicates=0)
        self.assertEqual(report["coverage"]["valid_paired_records"], 36)
        self.assertEqual(report["coverage"]["excluded_reason_counts"], {
            "unknown_or_invalid_answer_grade": 2, "invalid_probe_features": 1, "unknown_or_invalid_cost": 1})
        for setting in report["results"]:
            self.assertEqual({r["row"] for r in setting["predictions"]}, set(range(4, 40)))
            self.assertEqual(set(setting["scopes"]["primary"]["methods"]), set(fusion.METHODS))

    def test_missing_common_short_cost_stays_unknown_and_does_not_change_predictions(self):
        document = deepcopy(self.document)
        del document["records"][0]["short_probe_ms"]
        result = fusion.analyze(document, bootstrap_replicates=0)
        self.assertEqual(result["primary_records"], 40)
        for old_setting, new_setting in zip(self.report["results"], result["results"]):
            for old, new in zip(old_setting["predictions"], new_setting["predictions"]):
                self.assertEqual(old["scores"], new["scores"])
                self.assertEqual(old["actions"], new["actions"])
            for method in fusion.METHODS:
                summary = new_setting["scopes"]["all"]["methods"][method]
                self.assertIsNone(summary["common_short_probe_ms"])
                self.assertIsNone(summary["cached_loss_plus_common_probe_ms"])

    def test_empty_and_insufficient_question_cohorts_remain_not_estimable(self):
        empty = fusion_fixture()
        empty["records"] = []
        for document in (empty, fusion_fixture(questions=4, candidates=1)):
            report = fusion.analyze(document, bootstrap_replicates=0)
            for setting in report["results"]:
                self.assertEqual(setting["predictions"], [])
                self.assertEqual(setting["scopes"]["primary"]["status"], "not_estimable")
                self.assertEqual(setting["scopes"]["all"]["status"], "not_estimable")


class FusionOutcomeAccountingTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            saved_decision("qa", 1, 0, 0, 100., 200., "E", "T", "T", 10.),
            saved_decision("qa", 2, 1, 0, 100., 300., "E", "T", "E", 20.),
            saved_decision("qb", 1, 1, 1, 100., 400., "T", "E", "E", 30.),
        ]

    def test_remaining_error_penalty_and_common_probe_are_counted_once(self):
        result = fusion.decision_summary(self.records, "CH")
        self.assertEqual(result["correct"], 2)
        self.assertEqual(result["incorrect"], 1)
        self.assertEqual(result["avoidable_answer_errors"], 0)
        self.assertEqual(result["optimal_action_agreements"], 2)
        self.assertEqual(result["remaining_ms"], 600.)
        self.assertEqual(result["total_loss_ms"], 60600.)
        self.assertEqual(result["common_short_probe_ms"], 60.)
        self.assertEqual(result["cached_loss_plus_common_probe_ms"], 60660.)
        self.assertEqual(result["regret_to_pair_oracle_ms"], 100.)
        self.assertEqual(result["additional_probes"], 0)
        self.assertEqual(result["mean_loss_ms"], 20200.)
        self.assertEqual(result["question_macro_mean_loss_ms"], 30175.)

    def test_comparison_distinguishes_correctness_changes_from_same_quality_cost_changes(self):
        result = fusion.compare(self.records, "CH", "C", 0)
        self.assertEqual(result["changed_decisions"], 3)
        self.assertEqual(result["correct_to_wrong"], 0)
        self.assertEqual(result["wrong_to_correct"], 1)
        self.assertEqual(result["same_correctness_less_cost"], 1)
        self.assertEqual(result["same_correctness_more_cost"], 1)
        self.assertEqual(result["total_loss_difference_ms"], -60000.)
        self.assertEqual(result["net_utility_ms"], 60000.)
        self.assertEqual(result["mean_loss_difference_ms"], -20000.)
        self.assertEqual(result["question_macro_mean_loss_difference_ms"], -15075.)
        self.assertEqual(result["leave_one_question_out_mean_difference_range"], [-29850., -300.])
        self.assertIsNone(result["conditional_question_bootstrap_mean_difference_95"])

    def test_C_H_selection_oracle_is_not_the_unrestricted_E_T_oracle(self):
        report = fusion.summarize(self.records, 0)
        oracle = report["oracle_diagnostics_only"]
        self.assertEqual(oracle["pair_oracle_loss_ms"], 60500.)
        self.assertEqual(oracle["C_H_selection_oracle_loss_ms"], 120300.)
        self.assertEqual(oracle["C_H_selection_headroom_over_C_ms"], 300.)
        self.assertEqual(oracle["accuracy_upper_bound_correct"], 2)
        self.assertEqual(set(report["regression"]), {"Z", "C", "H", "CH", "trainmean", "late_avg"})
        self.assertNotIn("late_agree", report["regression"])
        self.assertNotIn("late_either", report["regression"])

    def test_bootstrap_groups_candidates_by_question_and_is_repeatable(self):
        one_question = deepcopy(self.records)
        for row in one_question:
            row["question_id"] = "one-question"
        self.assertIsNone(fusion.compare(one_question, "CH", "C", 50)["conditional_question_bootstrap_mean_difference_95"])
        first = fusion.compare(self.records, "CH", "C", 50)
        second = fusion.compare(self.records, "CH", "C", 50)
        self.assertEqual(first, second)
        low, high = first["conditional_question_bootstrap_mean_difference_95"]
        self.assertGreaterEqual(low, -29850.)
        self.assertLessEqual(high, -300.)

    def test_invalid_bootstrap_counts_are_not_silently_coerced(self):
        for count in (-1, True, 2.5):
            with self.assertRaisesRegex(ValueError, "nonnegative integer"):
                fusion.analyze(fusion_fixture(questions=4, candidates=1), bootstrap_replicates=count)


if __name__ == "__main__":
    unittest.main()
