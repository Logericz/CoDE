"""CPU arithmetic/protocol checks. No test establishes online GPU performance."""

import ast
from dataclasses import FrozenInstanceError
import itertools
import math
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from online_protocol import (
    UPSTREAM_COMMIT, CostObservation, ProbeObservation, ProtocolConfig,
    ProtocolController, ProtocolError, ScheduleConfig, SCHEDULE_RNG_ID,
    candidate_threshold, capped_geometric_interval, degeneration_score,
    legacy_confidence_oracle, stopping_reasons,
)


def observation(confidence=0.6, *, ended=True):
    # A backend-supplied confidence fixture; not recomputed from synthetic tokens.
    return ProbeObservation((1, 2, 9), (0.8, 0.8, 0.8), confidence, ended,
                            confidence_source="synthetic_test_fixture")


def controller(kind="dense", **kwargs):
    return ProtocolController(ProtocolConfig(r_min=1.0, r_max=1.0, tau=100.0),
                              ScheduleConfig(kind=kind, **kwargs))


def advance(instance, confidence=0.6, *, ended=True, probe_ms=10.0, reason_ms=10.0):
    j = instance.next_candidate_j
    return instance.observe(j, 100 * j, observation(confidence, ended=ended),
                            CostObservation(probe_ms, reason_ms))


class PinnedReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Read the fixed Git object, not the user's annotated working copy.
        result = subprocess.run(
            ["git", "-C", str(ROOT / "upstream" / "CoDE-Stop"), "show",
             f"{UPSTREAM_COMMIT}:method_codestop.py"],
            check=True, text=True, capture_output=True)
        names = {"compute_ramping_deer_threshold", "compute_degeneration_score"}
        tree = ast.parse(result.stdout)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in names]
        if {node.name for node in functions} != names:
            raise AssertionError("pinned reference functions missing")
        namespace = {"np": SimpleNamespace(log=math.log, log1p=math.log1p),
                     "itertools": itertools}
        exec(compile(ast.Module(body=functions, type_ignores=[]),
                     f"{UPSTREAM_COMMIT}:method_codestop.py", "exec"), namespace)
        cls.reference_D = staticmethod(namespace["compute_degeneration_score"])
        cls.reference_ramp = staticmethod(namespace["compute_ramping_deer_threshold"])

    def test_log_score_matches_pinned_function(self):
        rng = random.Random(5107)
        for n in range(3, 14):
            for _ in range(20):
                positions = sorted(rng.sample(range(1, 20000), n))
                confidences = [rng.choice((0, 1e-15, 1e-12, rng.random(), 1)) for _ in positions]
                self.assertEqual(degeneration_score(positions, confidences),
                                 self.reference_D(positions, confidences, use_log=True))

    def test_candidate_clock_is_one_based_wrapper_of_pinned_ramp(self):
        for steps in (0, 1, 2, 8):
            config = ProtocolConfig(ramp_steps=steps)
            for j in range(1, 20):
                self.assertEqual(candidate_threshold(j, config),
                                 self.reference_ramp(j - 1, steps, config.r_min, config.r_max))

    def test_sparse_same_endpoint_property_including_clipping(self):
        rng = random.Random(782)
        for _ in range(50):
            n = rng.randrange(3, 10)
            positions = sorted(rng.sample(range(1, 10000), n))
            values = [rng.choice((0, 1e-15, 1e-14, rng.random(), 1)) for _ in positions]
            dense = degeneration_score(positions, values)
            for mask in range(1 << (n - 2)):
                indices = [0] + [i for i in range(1, n - 1) if mask & (1 << (i - 1))] + [n - 1]
                sparse = degeneration_score([positions[i] for i in indices], [values[i] for i in indices])
                self.assertLessEqual(sparse, dense + 1e-12)

    def test_warmup_ramp_clock_zero_effect_and_no_warmup_counterexample(self):
        config = ProtocolConfig()
        for k, j in enumerate((1, 2, 3, 7, 11, 15), 1):
            self.assertEqual(candidate_threshold(j, config), candidate_threshold(k, config))
        self.assertGreater(candidate_threshold(5, config), candidate_threshold(2, config))


class ArithmeticTests(unittest.TestCase):
    def test_legacy_denominator_and_excluded_edges(self):
        self.assertAlmostEqual(legacy_confidence_oracle((0.1, 0.25, 0.8)), 0.5)
        self.assertAlmostEqual(legacy_confidence_oracle((0.9, 0.25, 0.2)), 0.5)
        self.assertAlmostEqual(legacy_confidence_oracle((0.8, 0.5, 0.25, 0.9)),
                               math.exp((math.log(0.5) + math.log(0.25)) / 3))
        self.assertAlmostEqual(legacy_confidence_oracle((0.5,) * 21), 0.5 ** (19 / 20))

    def test_legacy_short_long_invalid_inputs_and_zero_cancellation(self):
        for values in ((), (0.5,), (0.5, 0.5), (0.5,) * 22, (0.8, math.nan, 0.8),
                       (0.8, -0.1, 0.8), (0.8, math.inf, 0.8)):
            with self.assertRaises(ProtocolError):
                legacy_confidence_oracle(values)
        self.assertTrue(math.isnan(legacy_confidence_oracle((0.8, 0.8, 0.0))))
        self.assertEqual(legacy_confidence_oracle((0.8, 0.0, 0.8)), 0.0)

    def test_observation_preserves_invalid_and_explicit_oracle_identity(self):
        short = ProbeObservation.from_probabilities((1, 9), (0.8, 0.8), True)
        self.assertFalse(short.confidence_valid)
        self.assertIn("probe_token_count_le_2", short.invalid_reasons)
        self.assertEqual(short.confidence_source, "python_float_arithmetic_oracle")
        zero = ProbeObservation.from_probabilities((1, 2, 9), (0.8, 0.8, 0), True)
        self.assertTrue(math.isnan(zero.confidence_raw))
        self.assertFalse(zero.confidence_valid)
        incomplete = observation(0.6, ended=False)
        self.assertTrue(incomplete.confidence_valid)

    def test_D_requires_valid_positive_increasing_observations(self):
        for positions, values in (((0, 1, 2), (0.9, 0.8, 0.7)), ((1, 1, 2), (0.9, 0.8, 0.7)),
                                 ((2, 1, 3), (0.9, 0.8, 0.7)), ((1, 2), (0.9,)),
                                 ((1,), (math.nan,)), ((1,), (1.1,))):
            with self.assertRaises(ProtocolError):
                degeneration_score(positions, values)
        self.assertEqual(degeneration_score((1, 2), (0.9, 0.1)), 0)
        self.assertEqual(degeneration_score((1, 2, 3), (1e-15, 1e-16, 1e-17)), 0)
        self.assertAlmostEqual(degeneration_score((1, 2, 3), (0.9, 0.8, 0.7)),
                               2 + math.log(3 / 2))

    def test_strict_OR_stopping_and_incomplete_D_branch(self):
        config = ProtocolConfig()
        self.assertEqual(stopping_reasons(observation(0.95), 2.0, 0.95, config), ())
        self.assertEqual(stopping_reasons(observation(0.96), 2.1, 0.95, config),
                         ("confidence", "degeneration"))
        self.assertEqual(stopping_reasons(observation(0.96, ended=False), 2.1, 0.95, config),
                         ("degeneration",))
        self.assertEqual(stopping_reasons(observation(math.nan), None, 0.95, config), ())
        self.assertEqual(stopping_reasons(observation(0.94), 100, 0.95,
                                         ProtocolConfig(rule="deer")), ())

    def test_frozen_configuration_and_unsupported_scope(self):
        with self.assertRaises(FrozenInstanceError):
            ProtocolConfig().tau = 4
        for kwargs in ({"ramp_steps": -1}, {"tau": math.nan}, {"r_min": 0.99, "r_max": 0.9},
                       {"rule": "paper"}, {"ramp_steps": True}):
            with self.assertRaises(ProtocolError):
                ProtocolConfig(**kwargs)
        for kwargs in ({"kind": "unknown"}, {"kind": "random", "h_max": 4},
                       {"fixed_interval": 10}, {"beta": 0.9}, {"random_p": 0}):
            with self.assertRaises(ProtocolError):
                ScheduleConfig(**kwargs)
        with self.assertRaises(ProtocolError):
            candidate_threshold(0, ProtocolConfig())


class ControllerTests(unittest.TestCase):
    def test_fixed_and_log_candidate_recursion(self):
        instance = controller("fixed", fixed_interval=4)
        self.assertEqual([advance(instance).candidate_j for _ in range(6)], [1, 2, 3, 7, 11, 15])
        instance = controller("log", log_a=1.0, h_max=8)
        self.assertEqual([advance(instance).candidate_j for _ in range(6)], [1, 2, 3, 5, 7, 10])

    def test_invalid_prolongs_warmup_and_does_not_create_D(self):
        instance = controller("fixed")
        first = advance(instance)
        invalid = advance(instance, math.nan)
        third = advance(instance)
        fourth = advance(instance)
        self.assertEqual([x.next_candidate_j for x in (first, invalid, third, fourth)], [2, 3, 4, 8])
        self.assertIsNone(invalid.D_observed)
        self.assertEqual([x.candidate_j for x in instance.history], [1, 3, 4])
        self.assertEqual(third.signal_delta_j, 2)
        self.assertEqual(third.cost_delta_j, 1)

    def test_incomplete_valid_probe_updates_D_and_recovers_adaptive_families(self):
        for kind in ("adaptive", "backoff"):
            instance = controller(kind)
            advance(instance, 0.8)
            advance(instance, 0.8)
            third = advance(instance, 0.7, ended=False)
            self.assertEqual(third.valid_history_count, 3)
            self.assertEqual(third.D_observed, 1)
            self.assertEqual(third.h_next, 1)
            self.assertIn("incomplete_probe", third.forced_dense_reasons)
        instance = ProtocolController(ProtocolConfig(r_min=1, r_max=1, tau=0.5))
        advance(instance, 0.8)
        advance(instance, 0.8)
        self.assertEqual(advance(instance, 0.7, ended=False).stop_reasons, ("degeneration",))

    def test_valid_incomplete_keeps_fixed_log_random_cadence(self):
        for kind in ("fixed", "log", "random"):
            with self.subTest(kind=kind):
                decisions = []
                for ended in (True, False):
                    instance = ProtocolController(
                        ProtocolConfig(r_min=1, r_max=1, tau=100),
                        ScheduleConfig(kind=kind),
                        seed_schedule=0 if kind == "random" else None)
                    advance(instance, 0.8)
                    advance(instance, 0.8)
                    decisions.append(advance(instance, 0.7, ended=ended))
                complete, incomplete = decisions
                self.assertEqual(incomplete.h_next, complete.h_next)
                self.assertGreater(incomplete.h_next, 1)
                self.assertEqual(incomplete.D_observed, 1)
                self.assertNotIn("incomplete_probe", incomplete.forced_dense_reasons)

    def test_invalid_still_forces_every_family_dense_after_warmup(self):
        for kind in ("dense", "fixed", "log", "random", "backoff", "adaptive"):
            with self.subTest(kind=kind):
                instance = ProtocolController(
                    ProtocolConfig(r_min=1, r_max=1, tau=100),
                    ScheduleConfig(kind=kind),
                    seed_schedule=0 if kind == "random" else None)
                for _ in range(3):
                    advance(instance)
                invalid = advance(instance, math.nan)
                self.assertEqual(invalid.valid_history_count, 3)
                self.assertEqual(invalid.h_next, 1)
                self.assertIn("invalid_probe", invalid.forced_dense_reasons)

    def test_short_probe_raw_one_is_retained_but_cannot_stop(self):
        instance = ProtocolController()
        short = ProbeObservation((1, 9), (0.8, 0.8), 1.0, True)
        result = instance.observe(1, 100, short, CostObservation(4, None))
        self.assertEqual(result.observation.confidence_raw, 1.0)
        self.assertFalse(result.should_stop)
        self.assertEqual(result.valid_history_count, 0)
        self.assertEqual(result.ema_probe_ms, 4)
        self.assertEqual(result.next_candidate_j, 2)

    def test_stop_precedes_warmup_and_rejects_later_queries(self):
        instance = ProtocolController()
        result = advance(instance, 0.95)
        self.assertTrue(result.should_stop)
        self.assertEqual(result.threshold_r, 0.9)
        self.assertIsNone(result.h_next)
        self.assertIsNone(instance.next_candidate_j)
        with self.assertRaises(ProtocolError):
            instance.should_probe(2)

    def test_invalid_cost_updates_EMAs_without_adding_confidence(self):
        instance = controller("adaptive")
        advance(instance, 0.5, probe_ms=10, reason_ms=None)
        second = advance(instance, math.nan, probe_ms=20, reason_ms=10)
        third = advance(instance, 0.51, probe_ms=30, reason_ms=20)
        self.assertEqual(second.ema_probe_ms, 12)
        self.assertEqual(second.ema_reason_ms_per_candidate, 10)
        self.assertAlmostEqual(third.ema_probe_ms, 15.6)
        self.assertEqual(third.ema_reason_ms_per_candidate, 12)
        self.assertEqual(third.signal_delta_j, 2)
        self.assertEqual(third.cost_delta_j, 1)
        self.assertEqual(third.valid_history_count, 2)
        self.assertEqual(third.h_next, 1)

    def test_adaptive_min_rule_and_cost_interval_normalization(self):
        for probe_ms, expected_h in ((4, 1), (40, 8)):
            instance = controller("adaptive", beta=0.5, h_max=8)
            advance(instance, 0.5, probe_ms=probe_ms)
            advance(instance, 0.51, probe_ms=probe_ms)
            result = advance(instance, 0.52, probe_ms=probe_ms)
            self.assertAlmostEqual(result.margin_m, 0.48)
            self.assertAlmostEqual(result.activity_u, 0.01)
            self.assertEqual(result.h_signal, 8)
            self.assertEqual(result.h_cost, expected_h)
            self.assertEqual(result.h_next, expected_h)
            if expected_h == 8:
                following = advance(instance, 0.53, probe_ms=40, reason_ms=80)
                self.assertEqual(following.cost_delta_j, 8)
                self.assertEqual(following.ema_reason_ms_per_candidate, 10)

    def test_invalid_current_timing_does_not_reuse_stale_ratio_to_skip(self):
        instance = controller("adaptive")
        advance(instance, probe_ms=40)
        advance(instance, probe_ms=40)
        result = advance(instance, probe_ms=40, reason_ms=0)
        self.assertEqual(result.h_next, 1)
        self.assertIn("timing_state_unavailable_or_invalid", result.forced_dense_reasons)
        self.assertIn("missing_or_nonpositive_reason_time", result.timing_diagnostics)

    def test_backoff_doubles_from_dense_and_resets_near_boundary(self):
        instance = controller("backoff")
        self.assertEqual([advance(instance).h_next for _ in range(6)], [1, 1, 2, 4, 8, 8])
        result = advance(instance, 0.95)
        self.assertEqual(result.h_next, 1)
        self.assertIn("backoff_near_boundary_or_log_decline", result.forced_dense_reasons)

    def test_adaptive_raw_decline_vs_backoff_clipped_log_decline(self):
        for kind, expected in (("adaptive", 1), ("backoff", 2)):
            instance = controller(kind)
            advance(instance, 1e-15)
            advance(instance, 2e-15)
            result = advance(instance, 1e-15)
            self.assertEqual(result.D_observed, 0)
            self.assertEqual(result.h_next, expected)

    def test_no_silent_shift_or_unscheduled_probe(self):
        instance = controller("fixed")
        with self.assertRaises(ProtocolError):
            instance.observe(1, 0, observation())
        self.assertEqual(instance.next_candidate_j, 1)
        advance(instance)
        advance(instance)
        advance(instance)
        self.assertFalse(instance.should_probe(4))
        with self.assertRaises(ProtocolError):
            instance.observe(4, 400, observation())
        with self.assertRaises(ProtocolError):
            instance.should_probe(8)
        with self.assertRaises(ProtocolError):
            instance.observe(7, 300, observation())
        self.assertEqual(instance.next_candidate_j, 7)
        self.assertEqual(len(instance.history), 3)


class RandomScheduleTests(unittest.TestCase):
    def test_capped_geometric_has_tail_mass_at_eight(self):
        rng = random.Random(12908)
        counts = [0] * 9
        total, p = 50000, 0.2
        for _ in range(total):
            counts[capped_geometric_interval(p, rng)] += 1
        for value in range(1, 8):
            self.assertAlmostEqual(counts[value] / total, p * (1 - p) ** (value - 1), delta=0.01)
        self.assertAlmostEqual(counts[8] / total, (1 - p) ** 7, delta=0.01)
        self.assertEqual(capped_geometric_interval(1, rng), 1)

    def test_schedule_rng_is_local_repeatable_and_explicit(self):
        saved_global_state = random.getstate()
        instances = [ProtocolController(ProtocolConfig(r_min=1, r_max=1, tau=100),
                                        ScheduleConfig(kind="random"), seed_schedule=321)
                     for _ in range(2)]
        traces = [[advance(instance) for _ in range(30)] for instance in instances]
        self.assertEqual([x.candidate_j for x in traces[0]], [x.candidate_j for x in traces[1]])
        self.assertEqual(saved_global_state, random.getstate())
        self.assertEqual(traces[0][2].schedule_rng_identity, SCHEDULE_RNG_ID)
        self.assertEqual(traces[0][2].seed_schedule, 321)
        self.assertTrue(all(1 <= x.h_next <= 8 for x in traces[0]))
        with self.assertRaises(ProtocolError):
            ProtocolController(schedule=ScheduleConfig(kind="random"))


if __name__ == "__main__":
    unittest.main()
