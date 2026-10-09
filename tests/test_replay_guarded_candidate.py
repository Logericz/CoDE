"""Causal cached-trajectory replay tests; no model, GPU, or real answers."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import replay_guarded_candidate as replay

MARKERS = {"end_think": 9, "eos_ids": [10]}


def probes(confidences, *, complete=False):
    return [{"candidate_j": j, "token_position": 10 * j,
             "raw_observation": {"token_ids": [1, 2, 9 if complete else 3], "token_probs": [.5] * 3,
                                 "confidence_raw": value, "ended_with_think": complete,
                                 "confidence_source": "synthetic_cpu_fixture"},
             "probe_elapsed_ms": 2., "decision": {"cost": {"reason_elapsed_ms": 10. if j > 1 else None}}}
            for j, value in enumerate(confidences, 1)]


def run(rows, kind="guarded", endpoint=100):
    return replay.replay(rows, MARKERS, replay.ProtocolConfig(), replay.schedules()[kind], endpoint)


class CandidateReplayTests(unittest.TestCase):
    def test_only_queries_enter_history_skipped_peak_is_not_observed_and_stop_ends_consumption(self):
        rows = probes([.5, .5, .5, .99, .99, .1])
        for row in rows[3:5]:
            row["raw_observation"].update(token_ids=[1, 2, 9], ended_with_think=True)
        dense = run(rows, "dense")
        # A deliberately malformed future record must remain unconsumed after
        # the candidate stops. It is not an actual archived-data fixture.
        rows[-1] = {"future": "must not be consumed"}
        original = replay.saved_observation
        consumed = []
        def observe(row, markers):
            consumed.append(row["candidate_j"])
            return original(row, markers)
        with patch.object(replay, "saved_observation", side_effect=observe):
            guarded = run(rows)
        self.assertEqual(dense["stop_candidate_j"], 4)
        self.assertEqual(guarded["stop_candidate_j"], 5)
        self.assertEqual(consumed, [1, 2, 3, 5])
        self.assertEqual(guarded["valid_history_candidate_j"], consumed)
        self.assertEqual(guarded["skipped_candidate_j"], [4])
        self.assertEqual(guarded["decisions"][-1]["cost"]["reason_elapsed_ms"], 20.)
        self.assertEqual(guarded["partial_tokens"]["main_generated_including_pending_wait"], 51)
        self.assertEqual(guarded["partial_tokens"]["probe_generated"], 12)
        self.assertIsNone(guarded["partial_tokens"]["forced_answer_tokens"])
        self.assertIsNone(guarded["online_latency_estimate_ms"])

    def test_scheduled_query_beyond_endpoint_is_not_invented(self):
        result = run(probes([.5] * 4), endpoint=45)
        self.assertEqual(result["queried_candidate_j"], [1, 2, 3])
        self.assertEqual(result["skipped_candidate_j"], [4])
        self.assertEqual(result["next_scheduled_candidate_j"], 5)
        self.assertFalse(result["stopped"])
        self.assertIsNone(result["stop_candidate_j"])
        self.assertEqual(result["endpoint_kind"], "natural_eos")
        self.assertEqual(result["partial_tokens"]["main_generated_including_pending_wait"], 45)
        empty = run([], endpoint=1)
        self.assertEqual(empty["n_probes"], 0)
        self.assertEqual(empty["skipped_candidate_j"], [])

    def test_missing_skipped_interval_remains_unknown_and_recovers_dense(self):
        rows = probes([.5] * 6)
        rows[3]["decision"]["cost"]["reason_elapsed_ms"] = None
        result = run(rows)
        self.assertEqual(result["queried_candidate_j"], [1, 2, 3, 5, 6])
        fifth = result["decisions"][3]
        self.assertIsNone(fifth["cost"]["reason_elapsed_ms"])
        self.assertIn("timing_state_unavailable_or_invalid", fifth["forced_dense_reasons"])
        self.assertEqual(fifth["h_next"], 1)

    def test_shared_stop_precedes_schedule_even_for_incomplete_valid_probe(self):
        rows = probes([.8, .7, .6])
        for label in replay.LABELS:
            with self.subTest(label=label):
                result = run(rows, label)
                self.assertEqual(result["stop_candidate_j"], 3)
                self.assertEqual(result["stop_reasons"], ["degeneration"])
                self.assertIsNone(result["decisions"][-1]["h_next"])
                self.assertEqual(result["valid_history_candidate_j"], [1, 2, 3])

    def test_old_adaptive_and_fixed_h2_are_distinct_controls(self):
        rows = probes([.5] * 7)
        adaptive, fixed, guarded = (run(rows, kind) for kind in ("adaptive", "fixed-h2", "guarded"))
        self.assertEqual(adaptive["queried_candidate_j"], list(range(1, 8)))
        self.assertEqual(fixed["queried_candidate_j"], [1, 2, 3, 5, 7])
        self.assertEqual(guarded["queried_candidate_j"], fixed["queried_candidate_j"])
        self.assertEqual(adaptive["forced_dense_reason_counts"]["incomplete_probe"], 7)

    def test_answer_availability_requires_exact_endpoint_reasons_and_main_prefix(self):
        rows = probes([.99], complete=True)
        result = run(rows)
        collection = {"stop_reason": "natural_eos", "main_generated_tokens": 100,
                      "main_samples": list(range(100))}
        actual = {"probes": [{"stop_applied": True, "candidate_j": 1, "token_position": 10,
                               "decision": {"stop_reasons": ["confidence"]}}],
                  "main_generated_tokens": 11, "main_samples": list(range(11))}
        available = replay.answer_availability(result, {"codestop-dense": actual}, collection)
        self.assertEqual(available["status"], "existing_answer_available")
        self.assertFalse(available["graded"])
        for mutation in ("reasons", "prefix", "position"):
            changed = deepcopy(actual)
            if mutation == "reasons":
                changed["probes"][0]["decision"]["stop_reasons"] = ["degeneration"]
            elif mutation == "prefix":
                changed["main_samples"][-1] = -1
            else:
                changed["probes"][0]["token_position"] = 12
            self.assertEqual(replay.answer_availability(result, {"old": changed}, collection)["status"], "unknown")

    def test_new_output_is_exclusive_and_source_identity_includes_design(self):
        self.assertIn("docs/GUARDED_SCHEDULER.md", replay.SOURCE_FILES)
        self.assertIn("src/online_methods/guarded.py", replay.SOURCE_FILES)
        with tempfile.TemporaryDirectory() as directory, patch.object(replay, "WORK_ROOT", Path(directory)):
            output = Path(directory) / "new"
            replay.write_report({"status": "synthetic"}, output)
            before = (output / "replay.json").read_bytes()
            with self.assertRaises(ValueError):
                replay.write_report({"status": "replacement"}, output)
            self.assertEqual((output / "replay.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
