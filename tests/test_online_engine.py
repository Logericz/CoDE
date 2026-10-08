"""Synthetic CPU state-machine checks; no GPU/KV performance or research results."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from online_contract import TokenMarkers, TokenSample
from online_engine import RequestError, run_request
from online_protocol import ProbeObservation, ProtocolConfig, ProtocolController, ProtocolError, ScheduleConfig


WAIT, END, EOS, PREFIX, VALUE = 11, 12, 13, 14, 15


def observation(confidence=0.2, ended=True):
    return ProbeObservation((31, 32, END if ended else 33), (0.5, 0.5, 0.5),
                            confidence, ended, confidence_source="synthetic_cpu_fixture")


class Backend:
    markers = TokenMarkers(WAIT, END, EOS, (PREFIX,), (END, PREFIX))
    context_limit = 100000
    metadata = {"backend_kind": "synthetic_cpu_fixture", "gpu_validated": False}

    def __init__(self, main, probes=(), answer=(VALUE, EOS)):
        self.main, self.probes, self.answer = list(main), list(probes), list(answer)
        self.sample_count = self.greedy_count = self.probe_count = 0
        self.extended, self.probe_states, self.actions = [], [], []
        self.started = False
        self.fail_extend = None

    def prompt_ids(self, question):
        return [1, 2]

    def start_request(self, seed_reason):
        self.started = True
        self.seed = seed_reason

    def prefill(self, ids):
        self.actions.append("prefill")
        return list(ids)

    def sample(self, state):
        token = self.main[self.sample_count]
        self.sample_count += 1
        self.actions.append(("sample", token))
        return TokenSample(token, 1.0, 0.5)

    def greedy(self, state):
        token = self.answer[self.greedy_count % len(self.answer)]
        self.greedy_count += 1
        self.actions.append(("greedy", token))
        return TokenSample(token, 1.0, 0.5)

    def extend(self, state, ids):
        self.actions.append(("extend", tuple(ids)))
        if self.fail_extend is not None and self.fail_extend in ids:
            raise RuntimeError("synthetic extension failure")
        self.extended.append(tuple(ids))
        return state + list(ids)

    def probe(self, state):
        self.probe_states.append(list(state))
        result = self.probes[self.probe_count]
        self.probe_count += 1
        self.actions.append("probe")
        return result

    def synchronize(self):
        pass

    def decode(self, ids, *, skip_special_tokens=True):
        pieces = {END: "</think>", WAIT: "Wait", EOS: "<eos>", PREFIX: "The final answer is \\boxed", VALUE: "{7}"}
        return "".join(pieces.get(token, f"[{token}]") for token in ids if not (skip_special_tokens and token == EOS))

    def peak_memory_bytes(self):
        return None


def run(backend, **kwargs):
    arguments = {"question": "Synthetic question", "sample_id": "cpu-fixture", "rollout_id": 0,
                 "method": "codestop", "seed_reason": 42, "max_new_tokens": len(backend.main)}
    arguments.update(kwargs)
    return run_request(backend, **arguments)


class EngineTests(unittest.TestCase):
    def test_secondary_eos_terminates_main_and_finalization_without_rescue(self):
        secondary_eos = 16
        for path in ("main", "finalization"):
            with self.subTest(path=path):
                backend = Backend([7, secondary_eos] if path == "main" else [7],
                                  answer=(VALUE, secondary_eos))
                backend.markers = TokenMarkers(WAIT, END, EOS, (PREFIX,), (END, PREFIX),
                                               eos_ids=(EOS, secondary_eos))
                result = run(backend, method="vanilla")
                self.assertEqual(result["output_token_ids"][-1], secondary_eos)
                self.assertNotIn((secondary_eos,), backend.extended)
                if path == "main":
                    self.assertEqual(result["stop_reason"], "natural_eos")
                    self.assertEqual(backend.greedy_count, 0)
                    self.assertFalse(result["answer_boundary_confirmed"])
                else:
                    self.assertEqual(result["finalization_end"], "eos")
                    self.assertEqual(result["finalization_tokens"], 2)

    def test_secondary_eos_invalidates_probe_in_engine(self):
        secondary_eos = 16
        raw = ProbeObservation((31, secondary_eos, END), (0.5, 0.5, 0.5), 0.99, True)
        backend = Backend([7, WAIT, EOS], [raw])
        backend.markers = TokenMarkers(WAIT, END, EOS, (PREFIX,), (END, PREFIX),
                                       eos_ids=(EOS, secondary_eos))
        result = run(backend)
        decision = result["probes"][0]["decision"]
        self.assertEqual(decision["observation"]["invalid_reason"], "probe_contains_eos")
        self.assertFalse(result["probes"][0]["should_stop"])
        self.assertEqual(result["stop_reason"], "natural_eos")

    def test_marker_eos_defaults_deduplication_and_conflicts(self):
        self.assertEqual(Backend.markers.eos_ids, (EOS,))
        markers = TokenMarkers(WAIT, END, EOS, (PREFIX,), (END, PREFIX), eos_ids=(16, EOS, 16))
        self.assertEqual(markers.eos_ids, (EOS, 16))
        for ids in ((WAIT,), (END,), (True,), (-1,), None):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                TokenMarkers(WAIT, END, EOS, (PREFIX,), (END, PREFIX), eos_ids=ids)

    def test_natural_eos_never_rescued_even_on_budget(self):
        backend = Backend([7, EOS])
        result = run(backend)
        self.assertEqual(result["stop_reason"], "natural_eos")
        self.assertEqual(backend.greedy_count, 0)
        self.assertEqual(result["n_probes"], 0)
        self.assertFalse(result["answer_boundary_confirmed"])
        self.assertEqual(result["answer_text"], "")
        self.assertEqual(result["output_token_ids"], [7, EOS])
        self.assertNotIn((EOS,), backend.extended)

    def test_first_natural_close_defines_answer_even_with_wait_and_second_close(self):
        backend = Backend([7, END, PREFIX, VALUE, WAIT, END, EOS])
        result = run(backend)
        self.assertEqual(result["n_candidates"], 0)
        self.assertEqual(result["answer_token_ids"], [PREFIX, VALUE, WAIT, END, EOS])
        self.assertEqual(result["answer_boundary"]["source_token_span"], [1, 2])
        self.assertIn("\\boxed{7}", result["answer_text"])
        self.assertEqual(result["reason_tokens"], 2)
        self.assertEqual(result["natural_answer_tokens"], 5)

    def test_early_stop_discards_pending_wait_and_includes_injected_boxed(self):
        backend = Backend([7, WAIT, 8], [observation(0.99)])
        result = run(backend)
        self.assertEqual(result["stop_reason"], "confidence")
        self.assertTrue(result["probes"][0]["should_stop"])
        self.assertTrue(result["probes"][0]["stop_applied"])
        self.assertEqual(backend.sample_count, 2)
        self.assertEqual(backend.probe_states, [[1, 2, 7]])
        self.assertNotIn((WAIT,), backend.extended)
        self.assertEqual(result["output_token_ids"], [7, END, PREFIX, VALUE, EOS])
        self.assertEqual(result["discarded_generated_tokens"], 1)
        self.assertEqual(result["reason_tokens"], 2)
        self.assertEqual(result["answer_token_ids"], [PREFIX, VALUE, EOS])
        self.assertIn("\\boxed{7}", result["answer_text"])
        self.assertEqual(result["actual_generated_tokens"], 2 + 2 + 3)
        self.assertTrue(result["token_conservation_verified"])

    def test_continuing_wait_is_accepted_once_without_resampling(self):
        backend = Backend([7, WAIT, EOS], [observation(0.2)])
        result = run(backend)
        self.assertEqual(backend.sample_count, 3)
        self.assertEqual(backend.extended.count((WAIT,)), 1)
        self.assertEqual(result["output_token_ids"], [7, WAIT, EOS])

    def test_last_budget_wait_is_probed_before_budget_finalization(self):
        backend = Backend([7, WAIT], [observation(0.2)])
        result = run(backend)
        self.assertEqual(result["n_probes"], 1)
        self.assertEqual(result["stop_reason"], "budget")
        self.assertEqual(result["output_token_ids"][:4], [7, WAIT, END, PREFIX])
        self.assertLess(backend.actions.index("probe"), backend.actions.index(("extend", (WAIT,))))

    def test_budget_at_natural_end_think_continues_without_prefix(self):
        backend = Backend([END])
        result = run(backend)
        self.assertEqual(result["stop_reason"], "budget")
        self.assertEqual(result["injected_prompt_tokens"], 0)
        self.assertEqual(result["output_token_ids"], [END, VALUE, EOS])
        self.assertEqual(result["answer_boundary"]["source"], "natural_first_end_think")

    def test_forced_answer_cap_is_exactly_thirty_and_no_eos_verdict(self):
        backend = Backend([7], answer=(VALUE,))
        result = run(backend, method="vanilla")
        self.assertEqual(backend.greedy_count, 30)
        self.assertEqual(result["finalization_tokens"], 30)
        self.assertEqual(result["finalization_end"], "answer_budget")
        self.assertNotIn("correct", result)

    def test_zero_position_wait_raises_with_pending_evidence(self):
        backend = Backend([WAIT])
        with self.assertRaisesRegex(RequestError, "position=0") as caught:
            run(backend)
        partial = caught.exception.partial
        self.assertEqual(partial["n_main_samples"], 1)
        self.assertEqual(partial["output_token_ids"], [])
        self.assertEqual(partial["pending_sample"]["token_id"], WAIT)
        self.assertEqual(partial["candidates"][0]["token_position"], 0)

    def test_context_capacity_checked_before_start_prefill(self):
        backend = Backend([7])
        backend.context_limit = 34  # 2 prompt + 1 main + 32 final prefix/reserve = 35.
        with self.assertRaisesRegex(RequestError, "Context capacity") as caught:
            run(backend)
        self.assertFalse(backend.started)
        self.assertEqual(caught.exception.partial["context_required"], 35)
        self.assertEqual(backend.actions, [])

    def test_nonfinite_probe_retained_and_invalid_without_fake_score(self):
        raw = ProbeObservation((31, 32, 33), (0.5, float("nan"), 0.5), float("nan"), False)
        backend = Backend([7, WAIT, EOS], [raw])
        result = run(backend)
        probe = result["probes"][0]
        self.assertEqual(probe["raw_observation"]["confidence_raw"], {"nonfinite_float": "nan"})
        self.assertIsNone(probe["decision"]["D_observed"])
        self.assertFalse(probe["stop_applied"])
        json.dumps(result, allow_nan=False)

    def test_probe_end_marker_mismatch_is_error_with_raw_probe(self):
        raw = ProbeObservation((31, 32, 33), (0.5, 0.5, 0.5), 0.99, True)
        with self.assertRaisesRegex(RequestError, "actual last token") as caught:
            run(Backend([7, WAIT], [raw]))
        self.assertEqual(len(caught.exception.partial["probes"]), 1)
        self.assertIn("probe_cache", caught.exception.partial["phase_elapsed_ms"])

    def test_dense_collect_keeps_querying_after_would_stop(self):
        backend = Backend([7, WAIT, 8, WAIT, EOS], [observation(0.99), observation(0.99)])
        result = run(backend, method="dense_collect_no_stop")
        self.assertEqual(result["n_probes"], 2)
        self.assertTrue(all(p["would_stop"] for p in result["probes"]))
        self.assertFalse(any(p["should_stop"] for p in result["probes"]))
        self.assertFalse(any(p["stop_applied"] for p in result["probes"]))
        self.assertEqual(result["stop_reason"], "natural_eos")
        self.assertFalse(result["eligible_for_primary_speed_comparison"])
        self.assertEqual(result["scope"], "dense_trajectory_diagnostic_only")

    def test_controller_rejects_nondense_or_nonbool_disabled_stopping(self):
        with self.assertRaisesRegex(ProtocolError, "restricted to dense"):
            ProtocolController(schedule=ScheduleConfig(kind="fixed"), stop_enabled=False)
        for value in (0, 1, None, "false"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ProtocolError, "boolean"):
                    ProtocolController(stop_enabled=value)

    def test_fixed_schedule_cost_interval_uses_previous_actual_query(self):
        backend = Backend([7, WAIT, 7, WAIT, 7, WAIT, 7, WAIT, 7, WAIT, EOS],
                          [observation()] * 4)
        clock = iter(i * 0.001 for i in range(1000))
        with patch("online_engine.time.perf_counter", side_effect=lambda: next(clock)):
            result = run(backend, schedule_config=ScheduleConfig(kind="fixed", fixed_interval=2))
        self.assertEqual([p["candidate_j"] for p in result["probes"]], [1, 2, 3, 5])
        decisions = [p["decision"] for p in result["probes"]]
        self.assertIsNone(decisions[0]["cost"]["reason_elapsed_ms"])
        self.assertEqual(decisions[-1]["cost_delta_j"], 2)
        self.assertAlmostEqual(decisions[-1]["cost"]["reason_elapsed_ms"],
                               2 * decisions[1]["cost"]["reason_elapsed_ms"])
        phase_total = sum(result["phase_elapsed_ms"].values())
        self.assertAlmostEqual(result["time_total_ms"], phase_total + result["time_other_ms"])
        self.assertGreater(result["time_other_ms"], 0)

    def test_backend_failure_retains_committed_tokens_and_pending_sample(self):
        backend = Backend([7, 8])
        backend.fail_extend = 8
        with self.assertRaisesRegex(RequestError, "synthetic extension failure") as caught:
            run(backend, method="vanilla")
        partial = caught.exception.partial
        self.assertEqual(partial["output_token_ids"], [7])
        self.assertEqual(partial["pending_sample"]["token_id"], 8)
        self.assertEqual(partial["n_accepted_main_tokens"], 1)
        self.assertEqual(partial["n_main_samples"], 2)

    def test_method_deer_uses_deer_threshold_not_codestop_ramp(self):
        backend = Backend([7, WAIT, EOS], [observation(0.93)])
        result = run(backend, method="deer")
        self.assertEqual(result["stop_reason"], "natural_eos")
        self.assertEqual(result["probes"][0]["decision"]["threshold_r"], 0.95)
        self.assertEqual(result["protocol_config"]["rule"], "deer")


if __name__ == "__main__":
    unittest.main()
