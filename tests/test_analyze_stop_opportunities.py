"""Independent CPU arithmetic/evidence tests; no models or stored benchmark data."""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("stop_opportunities", ROOT / "scripts/analyze_stop_opportunities.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def rows(confidences, complete=True):
    return [{"candidate_j": j, "token_position": j * 10, "confidence": value, "complete": complete}
            for j, value in enumerate(confidences, 1)]


ENDPOINT = {"kind": "natural_eos", "main_generated_tokens": 123,
            "last_candidate_j": 7, "last_candidate_token_position": 70, "crossing_imputed": False}


class StopOpportunityTests(unittest.TestCase):
    def test_fixed_Q_and_ramp_zero_control(self):
        result = analysis.diagnose(rows([.8] * 19), ENDPOINT)
        self.assertEqual(result["fixed_Q"], [1, 2, 3, 7, 11, 15, 19])
        self.assertTrue(all(p["threshold_j"] == p["threshold_k"] for p in result["points"] if p["in_Q"]))
        self.assertTrue(all(p["D_sparse"] is None for p in result["points"] if not p["in_Q"]))

    def test_confidence_opportunity_delay_is_not_history_loss(self):
        result = analysis.diagnose(rows([.8, .8, .8, .99, .8, .8, .99]), ENDPOINT)
        values = result["first_crossing"]["confidence"]
        self.assertEqual([values[v]["candidate_j"] for v in analysis.VARIANTS], [4, 7, 7])
        comparison = result["comparisons"]["confidence"]
        self.assertTrue(comparison["first_A_crossing_was_skipped"])
        self.assertEqual(comparison["A_to_B_opportunity"]["token_delay"], 30)
        self.assertEqual(comparison["B_to_C_history"]["classification"], "same_crossing")

    def test_history_loss_and_opportunity_have_different_delays(self):
        result = analysis.diagnose(rows([.8, .8, .8, .7, .8, .7, .6, .6, .6, .6, .5]), ENDPOINT)
        values = result["first_crossing"]["degeneration"]
        self.assertEqual([values[v]["candidate_j"] for v in analysis.VARIANTS], [6, 7, 11])
        comparison = result["comparisons"]["degeneration"]
        self.assertEqual(comparison["A_to_B_opportunity"]["token_delay"], 10)
        self.assertEqual(comparison["B_to_C_history"]["token_delay"], 40)
        self.assertAlmostEqual(result["points"][6]["D_sparse"], 1)

    def test_skipped_confidence_peak_can_fall_before_next_query(self):
        result = analysis.diagnose(rows([.8, .8, .8, .99, .8, .8, .8]), ENDPOINT)
        comparison = result["comparisons"]["confidence"]
        next_q = comparison["next_Q_at_or_after_first_A"]
        self.assertEqual((next_q["candidate_j"], next_q["token_delay"]), (7, 30))
        self.assertTrue(next_q["confidence_declined_since_first_A"])
        self.assertFalse(next_q["confidence_crossing_at_next_Q"])
        self.assertIsNone(result["first_crossing"]["confidence"]["B"])

    def test_crossing_lost_until_endpoint_stays_null(self):
        result = analysis.diagnose(rows([.8, .8, .8, .7, .8, .7, .8]), ENDPOINT)
        self.assertIsNone(result["first_crossing"]["degeneration"]["C"])
        item = result["comparisons"]["degeneration"]["B_to_C_history"]
        self.assertEqual(item["classification"], "history_no_crossing_before_endpoint")
        self.assertIsNone(item["token_delay"])
        self.assertEqual(item["endpoint_if_later_untriggered"], ENDPOINT)

    def test_no_candidates_or_no_crossing_does_not_manufacture_terminal_probe(self):
        for values in ([], rows([.8] * 3)):
            result = analysis.diagnose(values, ENDPOINT)
            self.assertTrue(all(c is None for branch in result["first_crossing"].values() for c in branch.values()))
            self.assertLessEqual(len(result["points"]), 3)

    def test_strict_thresholds_and_incomplete_valid_D_stop(self):
        point = {"candidate_j": 3, "token_position": 30, "D_dense": 2., "D_sparse": 2.,
                 "complete": True, "confidence": .95, "threshold_j": .95}
        self.assertIsNone(analysis.crossing(point, "A", "or"))
        point.update(complete=False, confidence=.99, D_dense=2.1)
        result = analysis.crossing(point, "A", "or")
        self.assertEqual(result["reasons"], ["degeneration"])
        self.assertEqual(result["generated_main_tokens_including_pending_wait"], 31)

    def test_score_uses_common_terminal_and_epsilon_clipped_log(self):
        self.assertEqual(analysis.score([(10, .9), (20, .8)]), 0)
        self.assertAlmostEqual(analysis.score([(10, .9), (20, .8), (40, .7)]), 2 + math.log(2))
        self.assertEqual(analysis.score([(10, 1e-13), (20, 5e-14), (30, 1e-14)]), 0)

    def test_branch_analysis_continues_after_OR_for_diagnostic_only(self):
        result = analysis.diagnose(rows([.99, .8, .8, .7, .8, .7, .8]), ENDPOINT)
        self.assertEqual(result["first_crossing"]["or"]["A"]["candidate_j"], 1)
        self.assertGreater(result["first_crossing"]["degeneration"]["A"]["candidate_j"], 1)
        self.assertEqual(len(result["points"]), 7)

    def test_invalid_eos_or_false_complete_is_refused_not_repaired(self):
        markers = {"end_think": 9, "eos_ids": [10]}
        saved = {"candidate_j": 1, "token_position": 5,
                 "raw_observation": {"token_ids": [1, 2, 3], "token_probs": [.8] * 3,
                                     "confidence_raw": .8, "ended_with_think": False, "invalid_reason": None}}
        self.assertFalse(analysis.observations([saved], markers)[0]["complete"])
        for mutation in ({"confidence_raw": {"nonfinite_float": "nan"}}, {"token_ids": [1, 2, 10]},
                         {"ended_with_think": True}, {"token_probs": [.8, .8, False]}):
            changed = deepcopy(saved)
            changed["raw_observation"].update(mutation)
            with self.assertRaises(ValueError):
                analysis.observations([changed], markers)

    def test_input_mutation_wrong_anchor_duplicate_keys_and_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text('{"value": 1}')
            evidence = analysis.Evidence()
            evidence.read(path, analysis.digest(path))
            path.write_text('{"value": 2}')
            with self.assertRaises(ValueError):
                evidence.read(path)
            evidence.finish()
            self.assertFalse(evidence.checks[-1]["passed"])
            for raw in ('{"x": 1, "x": 2}', '{"x": NaN}'):
                path.write_text(raw)
                with self.assertRaises(ValueError):
                    analysis.Evidence().read(path)
            with self.assertRaises(ValueError):
                analysis.Evidence().read(path, "0" * 64)

    def test_actual_natural_endpoint_and_stopped_endpoint_are_distinct(self):
        natural = {"stop_reason": "natural_eos", "probes": [], "main_samples": [{"token_id": 10}]}
        evidence = analysis.Evidence()
        analysis.match_actual(evidence, "natural", None, natural, natural)
        broken = {**natural, "stop_reason": "budget"}
        with self.assertRaises(ValueError):
            analysis.match_actual(evidence, "wrong_end", None, broken, natural)

    def test_output_is_new_separate_and_hash_bound(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(analysis, "ROOT", Path(directory)):
            root = Path(directory)
            allowed = root / "runs/development10-analysis-20261009/mechanism/result"
            report = {"status": "failed", "check_count": 1, "questions": [],
                      "limitations": [], "failed_checks": [{"check": "synthetic_failure"}]}
            analysis.write_report(report, allowed, [root / "input"])
            hashes = json.loads((allowed / "OUTPUT_SHA256.json").read_text())
            self.assertTrue(all(analysis.digest(allowed / name) == sha for name, sha in hashes.items()))
            with self.assertRaises(ValueError):
                analysis.write_report(report, allowed, [])
            with self.assertRaises(ValueError):
                analysis.write_report(report, allowed.parent / "new", [allowed.parent])
            with self.assertRaises(ValueError):
                analysis.inside(root, "../escape")


if __name__ == "__main__":
    unittest.main()
