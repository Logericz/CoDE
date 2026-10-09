"""Guarded 候选的 CPU 行为测试；不代表 GPU 性能或开发题上的方法收益。"""

from dataclasses import replace
import math
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from online_methods import adaptive, guarded
from online_methods.common import ScheduleContext
from online_protocol import (
    CostObservation, ProbeObservation, ProtocolConfig, ProtocolController,
    ScheduleConfig, ValidObservation,
)


def observation(confidence=0.6, ended=False):
    return ProbeObservation((1, 2, 3), (0.8, 0.8, 0.8), confidence, ended,
                            confidence_source="synthetic_test_fixture")


def context(**changes):
    """已有 3 条有效历史、较低且平稳的置信、计时有效的默认场景。"""
    value = ScheduleContext(
        candidate_j=3, observation=observation(),
        previous=ValidObservation(2, 200, 0.59, 0.0), valid_history_count=3,
        threshold_r=0.95, score=0.0, signal_delta_j=1, rho=0.01,
        timing_diagnostics=(), protocol=ProtocolConfig(),
        schedule=ScheduleConfig(kind="guarded"), previous_interval=1, rng=None)
    return replace(value, **changes)


def advance(controller, confidence=0.6, ended=False):
    j = controller.next_candidate_j
    return controller.observe(j, 100 * j, observation(confidence, ended),
                              CostObservation(0.1, 10.0))


class GuardedSchedulerTests(unittest.TestCase):
    def test_valid_incomplete_can_skip_while_old_adaptive_is_dense(self):
        value = context()
        new = guarded.choose_next(value)
        old = adaptive.choose_next(replace(value, schedule=ScheduleConfig(kind="adaptive")))
        self.assertEqual(new.h_next, 2)
        self.assertEqual(new.forced_dense_reasons, ())
        self.assertIsNone(new.h_cost)
        self.assertEqual(old.h_next, 1)
        self.assertIn("incomplete_probe", old.forced_dense_reasons)

    def test_small_cost_ratio_does_not_disable_signal_based_skipping(self):
        for rho in (0.0, 0.001, 0.01, 1.0, 100.0):
            with self.subTest(rho=rho):
                result = guarded.choose_next(context(rho=rho))
                self.assertEqual(result.h_next, 2)
                self.assertIsNone(result.h_cost)

    def test_interval_is_capped_at_two_independent_of_other_family_options(self):
        for cap in (2, 4, 8):
            value = context(schedule=ScheduleConfig(kind="guarded", h_max=cap),
                            previous_interval=8)
            self.assertEqual(guarded.choose_next(value).h_next, 2)

    def test_confidence_and_D_protect_their_respective_boundaries(self):
        cases = [context(observation=observation(0.92),
                         previous=ValidObservation(2, 200, 0.92, 0.0)),
                 context(score=1.95, previous=ValidObservation(2, 200, 0.59, 1.95))]
        for value in cases:
            with self.subTest(score=value.score):
                result = guarded.choose_next(value)
                self.assertEqual(result.h_next, 1)
                self.assertEqual(result.forced_dense_reasons, ("guarded_near_stop_boundary",))

    def test_fast_rising_confidence_or_D_restores_dense(self):
        cases = [context(observation=observation(0.8),
                         previous=ValidObservation(2, 200, 0.6, 0.0)),
                 context(score=1.1)]
        for value in cases:
            with self.subTest(score=value.score):
                result = guarded.choose_next(value)
                self.assertEqual(result.h_next, 1)
                self.assertEqual(result.forced_dense_reasons, ("guarded_fast_signal_change",))

    def test_signal_rate_uses_candidates_since_previous_actual_query(self):
        slow = context(observation=observation(0.7),
                       previous=ValidObservation(1, 100, 0.5, 0.0), signal_delta_j=2)
        self.assertEqual(guarded.choose_next(slow).h_next, 2)
        self.assertEqual(guarded.choose_next(replace(slow, signal_delta_j=1)).h_next, 1)

    def test_decline_and_missing_or_invalid_signal_restore_dense(self):
        cases = [(context(observation=observation(0.5)), "confidence_decline"),
                 (context(previous=None), "signal_state_unavailable"),
                 (context(signal_delta_j=None), "signal_state_unavailable"),
                 (context(signal_delta_j=0), "signal_state_unavailable"),
                 (context(score=None), "signal_state_unavailable"),
                 (context(observation=observation(None)), "invalid_probe")]
        for value, reason in cases:
            with self.subTest(reason=reason, value=value):
                result = guarded.choose_next(value)
                self.assertEqual(result.h_next, 1)
                self.assertIn(reason, result.forced_dense_reasons)

    def test_warmup_and_invalid_timing_restore_dense(self):
        for count in (1, 2):
            result = guarded.choose_next(context(valid_history_count=count))
            self.assertEqual(result.h_next, 1)
            self.assertIn("fewer_than_three_valid_observations", result.forced_dense_reasons)
        cases = [context(rho=value) for value in (None, math.nan, math.inf)]
        cases.append(context(timing_diagnostics=("missing_or_nonpositive_reason_time",)))
        for value in cases:
            result = guarded.choose_next(value)
            self.assertEqual(result.h_next, 1)
            self.assertIn("timing_state_unavailable_or_invalid", result.forced_dense_reasons)

    def test_controller_skips_and_recovers_without_fabricating_missing_history(self):
        controller = ProtocolController(ProtocolConfig(), ScheduleConfig(kind="guarded"))
        decisions = [advance(controller, c) for c in (0.5, 0.51, 0.52, None, 0.53)]
        self.assertEqual([d.candidate_j for d in decisions], [1, 2, 3, 5, 6])
        self.assertEqual([d.h_next for d in decisions], [1, 1, 2, 1, 2])
        self.assertEqual([entry.candidate_j for entry in controller.history], [1, 2, 3, 6])
        self.assertEqual(decisions[-1].signal_delta_j, 3)
        self.assertEqual(decisions[-1].cost_delta_j, 1)
        self.assertIsNone(decisions[3].D_observed)

    def test_shared_confidence_stop_precedes_warmup_and_scheduler(self):
        controller = ProtocolController(ProtocolConfig(), ScheduleConfig(kind="guarded"))
        with patch.object(guarded, "choose_next", side_effect=AssertionError("must not schedule")):
            decision = advance(controller, 0.99, ended=True)
        self.assertEqual(decision.stop_reasons, ("confidence",))
        self.assertIsNone(decision.h_next)
        self.assertIsNone(controller.next_candidate_j)

    def test_shared_D_stop_accepts_incomplete_before_scheduler(self):
        controller = ProtocolController(ProtocolConfig(), ScheduleConfig(kind="guarded"))
        advance(controller, 0.8)
        advance(controller, 0.7)
        with patch.object(guarded, "choose_next", side_effect=AssertionError("must not schedule")):
            decision = advance(controller, 0.6)
        self.assertEqual(decision.stop_reasons, ("degeneration",))
        self.assertIsNone(decision.h_next)
        self.assertFalse(decision.observation.ended_with_think)

    def test_module_imports_without_an_initial_protocol_import(self):
        code = (f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r}); "
                "import online_methods.guarded; import online_engine")
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
