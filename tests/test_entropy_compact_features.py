"""合成观测测试：验证时间可用性与固定特征语义，不读取实验答案或拟合模型。"""
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.entropy_compact_features import (
    FEATURE_DIMENSIONS, FEATURE_METADATA, FEATURE_SPEC, compact_views,
)


def prepared_fixture():
    base = {"prefix_tokens": 701.0, "D_short": 0.125, "short_confidence": 0.75,
            "history_previous_confidence": 0.6, "short_prob_slope": -0.2,
            "short_probe_tokens": 21.0, "short_ended_with_think": 0.0,
            "candidate_index": 2.0, "history_previous_D": 100.0,
            "short_prob_mean": 0.83, "short_prob_mean_log": -0.23}
    entropy = {**base, "short_entropy_mean": 1.5, "short_entropy_slope": 0.3}
    extended = {**entropy, "long_confidence": 0.85, "long_prob_slope": 0.1,
                "long_entropy_mean": 1.1, "long_entropy_slope": -0.5,
                "long_probe_tokens": 42.0, "long_ended_with_think": 1.0}
    return {"B": base, "H": entropy, "long_H": extended,
            "E_error": 1, "T_error": 0, "E_ms": 200., "T_ms": 3000.,
            "extra_probe_ms": 125., "extension_eligible": True}


class CompactFeatureTests(unittest.TestCase):
    def test_exact_fixed_feature_names_dimensions_and_serializable_metadata(self):
        views = compact_views(prepared_fixture())
        self.assertEqual(FEATURE_DIMENSIONS, {"f_B": 5, "f_H": 7, "g_short": 9, "g_long": 15})
        for name, features in views.items():
            self.assertEqual(list(features), FEATURE_SPEC[name])
            self.assertEqual(len(features), FEATURE_DIMENSIONS[name])
        self.assertEqual(json.loads(json.dumps(FEATURE_METADATA, allow_nan=False)), FEATURE_METADATA)

    def test_future_long_observation_changes_only_long_g(self):
        row = prepared_fixture()
        before = compact_views(row)
        for name in row["long_H"]:
            row["long_H"][name] += 7.0
        after = compact_views(row)
        for name in ("f_B", "f_H", "g_short"):
            self.assertEqual(after[name], before[name])
        self.assertNotEqual(after["g_long"], before["g_long"])
        for name, value in after["g_short"].items():
            self.assertEqual(after["g_long"][name], value)

    def test_entropy_changes_never_enter_baseline_f(self):
        row = prepared_fixture()
        before = compact_views(row)
        row["H"]["short_entropy_mean"] += 1.0
        row["H"]["short_entropy_slope"] -= 1.0
        after = compact_views(row)
        self.assertEqual(after["f_B"], before["f_B"])
        for name in ("f_H", "g_short", "g_long"):
            self.assertNotEqual(after[name], before[name])

    def test_future_answers_errors_and_costs_cannot_enter_features(self):
        row = prepared_fixture()
        before = compact_views(row)
        row.update(E_error=0, T_error=1, E_ms=9e9, T_ms=0., extra_probe_ms=0.,
                   actions={"E": {"answer_text": "changed"}}, gold="unused")
        for view in ("B", "H", "long_H"):
            row[view].update(answer_text="unused", error=1, remaining_ms=9e9)
        self.assertEqual(compact_views(row), before)

    def test_short_D_is_copied_unchanged_not_recomputed_from_long_or_history(self):
        row = prepared_fixture()
        row["B"]["D_short"] = 0.123456789
        row["H"]["D_short"] = 80.
        row["long_H"]["D_short"] = 90.
        row["B"]["history_previous_D"] = 100.
        for view in compact_views(row).values():
            self.assertEqual(view["D_short"], 0.123456789)

    def test_confidence_change_is_current_minus_finite_previous(self):
        row = prepared_fixture()
        for previous in (0, 0.6, 0.9):
            row["B"]["history_previous_confidence"] = previous
            for view in compact_views(row).values():
                self.assertAlmostEqual(view["confidence_change"], 0.75 - previous)

    def test_missing_or_nonfinite_previous_remains_nan(self):
        for previous in (None, float("nan"), float("inf"), -float("inf"), "0.6", True):
            row = prepared_fixture()
            row["B"]["history_previous_confidence"] = previous
            for view in compact_views(row).values():
                self.assertTrue(math.isnan(view["confidence_change"]))
        row = prepared_fixture()
        del row["B"]["history_previous_confidence"]
        self.assertTrue(math.isnan(compact_views(row)["f_B"]["confidence_change"]))

    def test_long_retains_all_short_features_without_aliasing_input_or_views(self):
        row = prepared_fixture()
        saved = deepcopy(row)
        views = compact_views(row)
        for name, value in views["g_short"].items():
            self.assertEqual(views["g_long"][name], value)
        views["g_long"]["short_confidence"] = -1.
        views["f_H"]["prefix_tokens"] = -1.
        self.assertEqual(views["g_short"]["short_confidence"], 0.75)
        self.assertEqual(views["f_B"]["prefix_tokens"], 701.)
        self.assertEqual(row, saved)

    def test_short_length_and_termination_belong_to_g_not_f(self):
        row = prepared_fixture()
        before = compact_views(row)
        row["B"].update(short_probe_tokens=5., short_ended_with_think=1.)
        after = compact_views(row)
        for name in ("f_B", "f_H"):
            self.assertEqual(after[name], before[name])
            self.assertNotIn("short_probe_tokens", after[name])
            self.assertNotIn("short_ended_with_think", after[name])
        for name in ("g_short", "g_long"):
            self.assertEqual(after[name]["short_probe_tokens"], 5.)
            self.assertEqual(after[name]["short_ended_with_think"], 1.)

    def test_existing_slope_is_used_verbatim_without_probability_recomputation(self):
        row = prepared_fixture()
        row["B"]["short_prob_slope"] = 123.456
        row["B"]["short_prob_mean"] = 0.
        row["B"]["short_prob_mean_log"] = -100.
        views = compact_views(row)
        for view in views.values():
            self.assertEqual(view["short_prob_slope"], 123.456)
            self.assertNotIn("short_prob_mean", view)
            self.assertNotIn("short_prob_mean_log", view)

    def test_missing_required_observation_fails_instead_of_inventing_a_value(self):
        row = prepared_fixture()
        del row["B"]["short_confidence"]
        with self.assertRaises(KeyError):
            compact_views(row)


if __name__ == "__main__":
    unittest.main()
