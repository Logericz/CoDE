"""Behavior across method-module boundaries; synthetic CPU evidence only.

The candidate traces below were checked against the pre-refactor controller.
They exercise real skipping and recovery, which the incomplete probes in the
ten-question GPU pilot did not exercise for Adaptive. No archived run is needed
to run these tests, and no model or GPU is loaded.
"""
from pathlib import Path
import random
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from online_engine import run_request
from online_protocol import CostObservation, ProtocolConfig, ProtocolController, ScheduleConfig
# Reuse the existing scripted backend, not a copy of any production controller.
from test_online_engine import Backend, END, EOS, VALUE, WAIT, observation


class Clock:
    """Deterministic profiling clock; these durations are not measurements."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        self.now += 0.001
        return self.now


class CostlyProbeBackend(Backend):
    def __init__(self, main, clock, confidence=0.4):
        super().__init__(main, [observation(confidence)] * 30)
        self.clock = clock

    def probe(self, state):
        self.clock.now += 0.080  # Make cost-driven skipping reachable on CPU.
        return super().probe(state)


def advance(controller, confidence=0.5, *, ended=True, probe_ms=40, reason_ms=10):
    j = controller.next_candidate_j
    return controller.observe(j, 100 * j, observation(confidence, ended),
                              CostObservation(probe_ms, reason_ms))


def no_stop_controller(kind, *, seed=None):
    return ProtocolController(ProtocolConfig(r_min=1, r_max=1, tau=100),
                              ScheduleConfig(kind=kind), seed_schedule=seed)


class SharedEngineMethodTests(unittest.TestCase):
    def test_method_modules_can_be_imported_before_protocol_or_engine(self):
        # A fresh interpreter is necessary: the imports above would otherwise
        # hide cycles between online_protocol and its new scheduler modules.
        for module in ("vanilla", "dense", "fixed", "adaptive"):
            with self.subTest(module=module):
                code = (f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r}); "
                        f"import online_methods.{module}; import online_engine")
                result = subprocess.run([sys.executable, "-c", code],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_four_methods_preserve_main_tokens_while_selecting_different_queries(self):
        main = [7] + [token for _ in range(20) for token in (WAIT, 7)] + [END, VALUE, EOS]
        expected = {
            "vanilla": [],
            "dense": list(range(1, 21)),
            "fixed": [1, 2, 3, 7, 11, 15, 19],
            "adaptive": [1, 2, 3, 11, 19],
        }
        for kind, queried in expected.items():
            with self.subTest(kind=kind):
                clock = Clock()
                backend = CostlyProbeBackend(main, clock)
                with patch("online_engine.time.perf_counter", clock):
                    result = run_request(backend, question="Synthetic question", sample_id=kind,
                        rollout_id=0, seed_reason=42, max_new_tokens=len(main),
                        method="vanilla" if kind == "vanilla" else "codestop",
                        protocol_config=ProtocolConfig(r_min=1, r_max=1, tau=100),
                        schedule_config=ScheduleConfig(kind="dense" if kind == "vanilla" else kind))
                self.assertEqual([p["candidate_j"] for p in result["probes"]], queried)
                self.assertEqual(result["output_token_ids"], main)
                self.assertEqual([p["token_id"] for p in result["main_samples"]], main)
                self.assertEqual(result["answer_token_ids"], [VALUE, EOS])
                self.assertEqual(result["n_candidates"], 20)
                self.assertEqual(result["stop_reason"], "natural_eos")
                self.assertEqual(result["discarded_generated_tokens"], 0)
                self.assertEqual(backend.greedy_count, 0)
                # At candidate j, the pending Wait is excluded; earlier skipped
                # Wait tokens have already been accepted into the main prefix.
                self.assertEqual(backend.probe_states,
                                 [[1, 2] + main[:2 * j - 1] for j in queried])

    def test_all_three_codestop_methods_stop_before_warmup_and_discard_same_wait(self):
        outputs = []
        for kind in ("dense", "fixed", "adaptive"):
            clock = Clock()
            backend = CostlyProbeBackend([7, WAIT, 8, EOS], clock, confidence=0.99)
            with patch("online_engine.time.perf_counter", clock):
                result = run_request(backend, question="Synthetic question", sample_id=kind,
                    rollout_id=0, seed_reason=42, max_new_tokens=4, method="codestop",
                    schedule_config=ScheduleConfig(kind=kind))
            decision = result["probes"][0]["decision"]
            self.assertEqual(decision["stop_reasons"], ["confidence"])
            self.assertEqual(decision["forced_dense_reasons"], [])
            self.assertIsNone(decision["h_next"])
            self.assertEqual(result["main_generated_tokens"], 2)
            self.assertEqual(result["discarded_generated_tokens"], 1)
            self.assertNotIn((WAIT,), backend.extended)
            outputs.append(result["output_token_ids"])
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0], outputs[2])


class SchedulerRecoveryTests(unittest.TestCase):
    def test_adaptive_recovers_after_each_guard_without_resetting_observed_history(self):
        controller = no_stop_controller("adaptive")
        rows = [
            (0.50, {}), (0.51, {}), (0.52, {}), (None, {}), (0.53, {}),
            (0.54, {"ended": False}), (0.55, {}), (0.40, {}), (0.41, {}),
            (0.42, {"reason_ms": 0}), (0.43, {}),
        ]
        decisions = [advance(controller, confidence, **kwargs) for confidence, kwargs in rows]
        self.assertEqual([d.candidate_j for d in decisions],
                         [1, 2, 3, 11, 12, 20, 21, 29, 30, 38, 39])
        self.assertEqual([d.h_next for d in decisions], [1, 1, 8, 1, 8, 1, 8, 1, 8, 1, 8])
        self.assertEqual([d.forced_dense_reasons for d in decisions[2:]], [
            (), ("invalid_probe", "signal_state_unavailable"), (),
            ("incomplete_probe",), (), ("confidence_decline",), (),
            ("timing_state_unavailable_or_invalid",), (),
        ])
        # An invalid query still incurs cost, but supplies no confidence signal.
        self.assertEqual(decisions[4].signal_delta_j, 9)
        self.assertEqual(decisions[4].cost_delta_j, 1)
        self.assertIsNone(decisions[3].D_observed)
        self.assertEqual(len(controller.history), 10)
        self.assertEqual(decisions[9].timing_diagnostics,
                         ("missing_or_nonpositive_reason_time",))

    def test_random_recovery_does_not_consume_a_draw_for_invalid_probe(self):
        global_before = random.getstate()
        expected_draws = [1, 1, 2, 1, 5, 6, 4, 1]
        for inject_invalid in (False, True):
            controller = no_stop_controller("random", seed=321)
            advance(controller)
            advance(controller)
            observed_draws = []
            for index in range(len(expected_draws)):
                if inject_invalid and index == 1:
                    invalid = advance(controller, None)
                    self.assertEqual(invalid.forced_dense_reasons, ("invalid_probe",))
                    self.assertEqual(invalid.h_next, 1)
                observed_draws.append(advance(controller).h_next)
            self.assertEqual(observed_draws, expected_draws)
        self.assertEqual(random.getstate(), global_before)

    def test_backoff_invalid_incomplete_and_decline_each_reset_growth(self):
        controller = no_stop_controller("backoff")
        rows = [(0.5, {}), (0.51, {}), (0.52, {}), (None, {}), (0.53, {}),
                (0.54, {"ended": False}), (0.55, {}), (0.4, {}),
                (0.41, {}), (0.42, {"reason_ms": 0}), (0.43, {})]
        decisions = [advance(controller, confidence, **kwargs) for confidence, kwargs in rows]
        self.assertEqual([d.candidate_j for d in decisions], [1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 18])
        self.assertEqual([d.h_next for d in decisions], [1, 1, 2, 1, 2, 1, 2, 1, 2, 4, 8])
        self.assertEqual(decisions[7].forced_dense_reasons,
                         ("backoff_near_boundary_or_log_decline",))
        # Unlike Adaptive, backoff does not gate its interval on timing.
        self.assertEqual(decisions[9].forced_dense_reasons, ())


if __name__ == "__main__":
    unittest.main()
