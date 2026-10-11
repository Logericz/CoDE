"""合成短观测测试；不读取真实实验结果，不拟合模型。"""
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.entropy_compact_features import compact_views
from src.entropy_short_features import (
    FEATURE_DIMENSIONS, FEATURE_METADATA, FEATURE_SPEC, feature_views,
)
from test_entropy_compact_features import prepared_fixture


class ShortFeatureTests(unittest.TestCase):
    def test_fixed_names_dimensions_and_serializable_metadata(self):
        views = feature_views(prepared_fixture())
        self.assertEqual(FEATURE_DIMENSIONS, {"Z": 3, "C": 7, "H": 5, "CH": 9})
        for name, fields in views.items():
            self.assertEqual(list(fields), FEATURE_SPEC[name])
            self.assertEqual(len(fields), FEATURE_DIMENSIONS[name])
        self.assertEqual(json.loads(json.dumps(FEATURE_METADATA, allow_nan=False)), FEATURE_METADATA)

    def test_CH_exactly_matches_prior_compact_short_including_missing_history(self):
        for previous in (0.6, 0, 0.9, None, float("nan"), float("inf")):
            row = prepared_fixture()
            row["B"]["history_previous_confidence"] = previous
            expected = compact_views(row)["g_short"]
            actual = feature_views(row)["CH"]
            self.assertEqual(set(actual), set(expected))
            for name, value in expected.items():
                if math.isnan(value):
                    self.assertTrue(math.isnan(actual[name]))
                else:
                    self.assertEqual(actual[name], value)

    def test_changing_entropy_does_not_change_C_or_Z(self):
        row = prepared_fixture()
        before = feature_views(row)
        row["H"]["short_entropy_mean"] += 10.
        row["H"]["short_entropy_slope"] -= 10.
        after = feature_views(row)
        self.assertEqual(after["C"], before["C"])
        self.assertEqual(after["Z"], before["Z"])
        self.assertNotEqual(after["H"], before["H"])
        self.assertNotEqual(after["CH"], before["CH"])

    def test_changing_confidence_D_and_probability_does_not_change_H_or_Z(self):
        row = prepared_fixture()
        before = feature_views(row)
        for source in ("B", "H"):
            row[source].update(short_confidence=0.1, history_previous_confidence=0.9,
                               D_short=100., short_prob_slope=20.)
        after = feature_views(row)
        self.assertEqual(after["H"], before["H"])
        self.assertEqual(after["Z"], before["Z"])
        self.assertNotEqual(after["C"], before["C"])
        self.assertNotEqual(after["CH"], before["CH"])
        for name in ("D_short", "confidence_change", "short_confidence", "short_prob_slope"):
            self.assertNotIn(name, after["H"])

    def test_future_outcomes_and_costs_neither_read_nor_required(self):
        row = prepared_fixture()
        before = feature_views(row)
        row.update(long_H={"invalid_future": object()}, E_error=0, T_error=1,
                   E_ms=9e12, T_ms=0., extra_probe_ms=0., gold="unused",
                   actions={"E": {"answer_text": "unused"}})
        self.assertEqual(feature_views(row), before)
        del row["long_H"]
        self.assertEqual(feature_views(row), before)
        self.assertEqual(feature_views({"B": row["B"], "H": row["H"]}), before)

    def test_top_level_access_is_limited_to_B_and_H(self):
        class RestrictedRow(dict):
            def __getitem__(self, key):
                if key not in ("B", "H"):
                    raise AssertionError("Unexpected field access: " + key)
                return super().__getitem__(key)
        row = prepared_fixture()
        self.assertEqual(feature_views(RestrictedRow(row)), feature_views(row))

    def test_missing_previous_is_nan_and_zero_previous_is_valid(self):
        row = prepared_fixture()
        for previous in (None, float("nan"), float("inf"), -float("inf"), "0.6", True):
            row["B"]["history_previous_confidence"] = previous
            views = feature_views(row)
            self.assertTrue(math.isnan(views["C"]["confidence_change"]))
            self.assertTrue(math.isnan(views["CH"]["confidence_change"]))
        del row["B"]["history_previous_confidence"]
        self.assertTrue(math.isnan(feature_views(row)["C"]["confidence_change"]))
        row["B"]["history_previous_confidence"] = 0
        self.assertEqual(feature_views(row)["C"]["confidence_change"], 0.75)

    def test_D_and_curve_summaries_are_preserved_not_recalculated(self):
        row = prepared_fixture()
        row["B"].update(D_short=0.123456789, short_prob_slope=8.7654321,
                        history_previous_D=999., short_prob_mean=999.)
        row["H"]["short_entropy_slope"] = -9.87654321
        views = feature_views(row)
        self.assertEqual(views["C"]["D_short"], 0.123456789)
        self.assertEqual(views["C"]["short_prob_slope"], 8.7654321)
        self.assertEqual(views["H"]["short_entropy_slope"], -9.87654321)

    def test_all_views_share_only_the_declared_context(self):
        row = prepared_fixture()
        row["B"].update(prefix_tokens=123., short_probe_tokens=5., short_ended_with_think=1.)
        views = feature_views(row)
        self.assertEqual(set(views["C"]) & set(views["H"]), set(views["Z"]))
        self.assertEqual(set(views["CH"]), set(views["C"]) | set(views["H"]))
        for view in views.values():
            for name, value in views["Z"].items():
                self.assertEqual(view[name], value)

    def test_returned_views_do_not_alias_one_another_or_the_input(self):
        row = prepared_fixture()
        saved = deepcopy(row)
        views = feature_views(row)
        views["Z"]["prefix_tokens"] = -1.
        views["CH"]["short_confidence"] = -1.
        self.assertEqual(views["C"]["prefix_tokens"], 701.)
        self.assertEqual(views["C"]["short_confidence"], 0.75)
        self.assertEqual(row, saved)

    def test_missing_required_short_observation_fails_explicitly(self):
        row = prepared_fixture()
        del row["H"]["short_entropy_mean"]
        with self.assertRaises(KeyError):
            feature_views(row)


if __name__ == "__main__":
    unittest.main()
