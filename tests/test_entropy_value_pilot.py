"""纯fake状态机测试：无torch、模型、SSH或GPU，不作为实验结果。"""
from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import run_entropy_value_pilot as pilot
from online_contract import TokenMarkers, TokenSample

WAIT, END, EOS, PREFIX, VALUE = 11, 12, 13, 14, 15


@dataclass
class RngValue:
    index: int

    def clone(self):
        return RngValue(self.index)


class FakeGenerator:
    def __init__(self):
        self.index = 0

    def get_state(self):
        return RngValue(self.index)

    def set_state(self, value):
        self.index = value.index


class FakeBackend:
    """extend故意原地修改list，使丢失clone的回归无法被不可变fake掩盖。"""
    markers = TokenMarkers(WAIT, END, EOS, (PREFIX,), (END, PREFIX))
    context_limit = 100000
    metadata = {"scope": "synthetic_cpu_fixture"}

    def __init__(self, main=(EOS,), final=(VALUE, EOS)):
        self.main, self.final = list(main), list(final)
        self._generator = FakeGenerator()
        self.sample_calls = 0
        self.extensions, self.final_input_prefixes = [], []

    def start_request(self, seed):
        self.seed = seed
        self._generator.index = 0

    def prompt_ids(self, question):
        return [101, 102]

    def prefill(self, ids):
        return list(ids)

    def sample(self, state):
        token = self.main[self._generator.index]
        self._generator.index += 1
        self.sample_calls += 1
        return TokenSample(token, .7, .6)

    def greedy(self, state):
        start = len(state) - 1 - state[::-1].index(PREFIX)
        index = len(state) - start - 1
        return TokenSample(self.final[index % len(self.final)], .8, .8)

    def extend(self, state, tokens):
        if tuple(tokens) == self.markers.final_prefix:
            self.final_input_prefixes.append(list(state))
        self.extensions.append(tuple(tokens))
        state.extend(tokens)
        return state

    def synchronize(self):
        pass

    def decode(self, ids, *, skip_special_tokens=False):
        pieces = {WAIT: "Wait", END: "</think>", EOS: "<eos>",
                  PREFIX: "The final answer is \\boxed", VALUE: "{7}"}
        return "".join(pieces.get(token, f"[{token}]") for token in ids)

    def peak_memory_bytes(self):
        return None


class FakeLog:
    def __init__(self):
        self.events = []

    def emit(self, name, **fields):
        self.events.append((name, fields))


def clone_patch():
    return patch("entropy_probe.clone_state", side_effect=lambda backend, state: list(state))


def row(identifier, **extra):
    return {"id": identifier, "problem": f"problem {identifier}", "answer": "7",
            "problem_sha256": "0" * 64, **extra}


def snapshot():
    return {"token_ids": [31, 32, END], "token_probs": [.8, .8, .8],
            "confidence_raw": .8, "entropy_raw_fp32_nats": [.4, .3, .2],
            "ended_with_think": True, "ended_with_eos": False,
            "invalid_reason": None, "actual_length": 3}


class PairedEndpointTests(unittest.TestCase):
    def test_E_finalizer_clones_main_and_retains_boxed_prefix_without_trial_text(self):
        backend, state = FakeBackend(), [101, 102, 7]
        with clone_patch():
            result = pilot.final_answer(backend, state, lambda: None)
        self.assertEqual(state, [101, 102, 7])
        self.assertEqual(backend._generator.index, 0)
        self.assertEqual(backend.final_input_prefixes, [[101, 102, 7]])
        self.assertEqual(result["answer_text"], "The final answer is \\boxed{7}")
        self.assertEqual(result["token_ids"], [VALUE, EOS])
        self.assertEqual(result["answer_token_ids"], [PREFIX, VALUE, EOS])
        self.assertNotIn("</think>", result["answer_text"])

    def test_T_budget_counts_pending_wait_once_and_restores_rng_main(self):
        backend, state = FakeBackend(main=[7] * 300), [101, 102]
        with clone_patch():
            result = pilot.continue_then_answer(backend, state, WAIT, lambda: None)
        self.assertEqual(result["main_generated_tokens"], 256)
        self.assertEqual(result["main_token_ids"], [WAIT] + [7] * 255)
        self.assertEqual(backend.sample_calls, 255)
        self.assertEqual(backend.extensions.count((WAIT,)), 1)
        self.assertEqual(state, [101, 102])
        self.assertEqual(backend._generator.index, 0)
        self.assertEqual(backend.final_input_prefixes[0], state + [WAIT] + [7] * 255)
        self.assertEqual(backend.extensions.count(backend.markers.final_prefix), 1)
        self.assertEqual(result["main_termination"], "chunk")

    def test_T_natural_boundary_is_pending_so_finalizer_does_not_double_close(self):
        for boundary in (END, EOS):
            with self.subTest(boundary=boundary):
                backend, state = FakeBackend(main=[7, boundary]), [101, 102]
                with clone_patch():
                    result = pilot.continue_then_answer(backend, state, WAIT, lambda: None)
                self.assertEqual(result["main_token_ids"], [WAIT, 7, boundary])
                self.assertEqual(result["main_generated_tokens"], 3)
                self.assertEqual(result["main_termination"], "natural_boundary_pending")
                self.assertEqual(backend.final_input_prefixes, [[101, 102, WAIT, 7]])
                self.assertNotIn((boundary,), backend.extensions)
                self.assertEqual(backend.extensions.count(backend.markers.final_prefix), 1)
                self.assertEqual(state, [101, 102])
                self.assertEqual(backend._generator.index, 0)

    def test_T_deadline_restores_rng_even_when_it_fails(self):
        backend, state = FakeBackend(main=[7] * 20), [101, 102]
        calls = 0
        def deadline():
            nonlocal calls
            calls += 1
            if calls == 3:
                raise TimeoutError("synthetic deadline")
        with clone_patch(), self.assertRaisesRegex(TimeoutError, "synthetic"):
            pilot.continue_then_answer(backend, state, WAIT, deadline)
        self.assertEqual(backend.sample_calls, 2)
        self.assertEqual(backend._generator.index, 0)
        self.assertEqual(state, [101, 102])

    def test_E_and_T_use_same_finalizer_contract(self):
        backend, state = FakeBackend(main=[7, END]), [101, 102]
        with clone_patch():
            immediate = pilot.final_answer(backend, state, lambda: None)
            later = pilot.continue_then_answer(backend, state, WAIT, lambda: None)
        for name in ("answer_text", "boundary_policy", "answer_boundary_confirmed", "termination"):
            self.assertEqual(immediate[name], later[name])
        self.assertEqual(backend.final_input_prefixes, [[101, 102], [101, 102, WAIT, 7]])


class SelectionAndCollectionTests(unittest.TestCase):
    def write_split(self, root, name, rows):
        (root / name).write_text("\n".join(json.dumps(value) for value in rows) + "\n")

    def test_selection_excludes_all_pilot_ids_and_keeps_frozen_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pilots = [row(f"p{i}") for i in range(20)]
            eligible = [row(f"d{i}") for i in range(25)]
            selection = pilots[:5] + eligible[:8] + pilots[5:] + eligible[8:]
            self.write_split(root, "pilot20.jsonl", pilots)
            self.write_split(root, "selection120.jsonl", selection)
            with patch.object(pilot, "verify_data", return_value=(pilots[0], {"identity": "frozen"})) as verify:
                selected, identity = pilot.select_questions(root)
            self.assertEqual(selected, eligible[:20])
            self.assertEqual(identity, {"identity": "frozen"})
            verify.assert_called_once_with(root.resolve(), pilot.DATA_HASH, "p0")

    def test_missing_gold_in_selected_twenty_aborts_without_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_split(root, "pilot20.jsonl", [row("pilot")])
            eligible = [row(f"d{i}") for i in range(25)]
            eligible[3]["answer"] = ""
            self.write_split(root, "selection120.jsonl", eligible)
            with patch.object(pilot, "verify_data", return_value=(row("pilot"), {})), self.assertRaisesRegex(ValueError, "do not silently replace"):
                pilot.select_questions(root)

    def test_no_candidate_retained_as_completed_zero_candidate_question(self):
        backend, log = FakeBackend(main=[7, END]), FakeLog()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records, summary, need_gate = pilot.collect_question(
                backend, row("d0"), 0, root, log, time.monotonic(), False)
            self.assertEqual(records, [])
            self.assertEqual(summary["candidates"], 0)
            self.assertEqual(summary["terminal"], "natural_boundary")
            self.assertFalse(need_gate)
            self.assertTrue((root / "question-01/summary.json").exists())

    def test_first_token_wait_fails_instead_of_shifting_candidate(self):
        backend, log = FakeBackend(main=[WAIT, 7, EOS]), FakeLog()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "First-token Wait"):
                pilot.collect_question(backend, row("d0"), 0, root, log, time.monotonic(), False)
            self.assertEqual(backend.sample_calls, 1)
            self.assertEqual(list(root.glob("question-*/candidate-*.json")), [])
            self.assertFalse((root / "question-01/summary.json").exists())

    def test_failed_long_extension_keeps_prefix_and_short_and_closes_session(self):
        backend, log = FakeBackend(main=[7, WAIT, EOS]), FakeLog()
        def extend(session, target):
            if target == pilot.LONG_CAP:
                raise RuntimeError("synthetic long extension failure")
            return snapshot()
        with tempfile.TemporaryDirectory() as temporary, \
                patch("entropy_probe.start_probe", return_value=object()), \
                patch("entropy_probe.extend_probe", side_effect=extend), \
                patch("entropy_probe.close_probe") as close:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "long extension"):
                pilot.collect_question(backend, row("d0"), 0, root, log, time.monotonic(), False)
            partial = root / "question-01/candidate-01-partial"
            self.assertEqual({path.name for path in partial.iterdir()}, {"prefix.json", "short.json"})
            self.assertEqual(json.loads((partial / "prefix.json").read_text())["main_token_ids"], [7])
            self.assertEqual(json.loads((partial / "short.json").read_text())["observation"], snapshot())
            self.assertFalse((root / "question-01/candidate-01.json").exists())
            close.assert_called_once()

    def test_failed_second_action_retains_first_action_without_complete_label(self):
        backend, log = FakeBackend(main=[7, WAIT, EOS]), FakeLog()
        first_action = {"answer_text": "synthetic T answer", "generated_tokens": 2}
        with tempfile.TemporaryDirectory() as temporary, \
                patch("entropy_probe.start_probe", return_value=object()), \
                patch("entropy_probe.extend_probe", side_effect=lambda *args: snapshot()), \
                patch("entropy_probe.close_probe"), \
                patch.object(pilot, "continue_then_answer", return_value=first_action), \
                patch.object(pilot, "final_answer", side_effect=RuntimeError("synthetic E failure")):
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "E failure"):
                pilot.collect_question(backend, row("d0"), 0, root, log, time.monotonic(), False)
            partial = root / "question-01/candidate-01-partial"
            self.assertEqual({path.name for path in partial.iterdir()},
                             {"prefix.json", "short.json", "long.json", "T.json"})
            saved = json.loads((partial / "T.json").read_text())
            self.assertEqual(saved["answer_text"], "synthetic T answer")
            self.assertIsNone(saved["error"])
            self.assertFalse((root / "question-01/candidate-01.json").exists())
            self.assertFalse((root / "question-01/summary.json").exists())

    def test_three_candidate_cap_keeps_main_path_despite_T_rng_consumption(self):
        backend, log = FakeBackend(main=[7, WAIT, 8, WAIT, 9, WAIT, EOS]), FakeLog()
        with tempfile.TemporaryDirectory() as temporary, clone_patch(), \
                patch("entropy_probe.start_probe", side_effect=lambda *args, **kwargs: object()), \
                patch("entropy_probe.extend_probe", side_effect=lambda *args: snapshot()), \
                patch("entropy_probe.close_probe") as close:
            root = Path(temporary)
            records, summary, _ = pilot.collect_question(
                backend, row("d0"), 0, root, log, time.monotonic(), False)
            self.assertEqual(len(records), 3)
            self.assertEqual(summary["terminal"], "candidate_cap")
            self.assertEqual(summary["main_token_ids"], [7, WAIT, 8, WAIT, 9, WAIT])
            self.assertEqual([record["prefix_tokens"] for record in records], [1, 3, 5])
            self.assertEqual(close.call_count, 3)
            self.assertEqual(len(list((root / "question-01").glob("candidate-*.json"))), 3)
            self.assertTrue(all(set(record["actions"]) == {"E", "T"} for record in records))


class FailureAndGradeTests(unittest.TestCase):
    def test_grading_deadline_saves_completed_subset_and_leaves_rest_unknown(self):
        record = {"question_id": "q", "candidate_index": 1, "actions": {
            name: {"answer_text": name, "boundary_policy": "known_injected_final_prefix_v1", "error": None}
            for name in ("E", "T")}}
        calls = 0
        def deadline():
            nonlocal calls
            calls += 1
            if calls == 2:
                raise TimeoutError("synthetic grading deadline")
        with tempfile.TemporaryDirectory() as temporary, \
                patch("math_grading._check_dependencies", return_value={"fake": "fixed"}), \
                patch("math_grading.grade_record", return_value={"id": "q:1:E", "grade": True, "status": "graded"}) as grade:
            root = Path(temporary)
            with self.assertRaisesRegex(TimeoutError, "grading deadline"):
                pilot.grade_saved(root, [row("q")], [record], deadline=deadline)
            saved = json.loads((root / "grading.json").read_text())
            self.assertFalse(saved["completed"])
            self.assertEqual(len(saved["results"]), 1)
            self.assertEqual(saved["results"][0]["id"], "q:1:E")
            self.assertEqual(record["actions"]["E"]["error"], 0)
            self.assertIsNone(record["actions"]["T"]["error"])
            self.assertNotIn("grading_status", record["actions"]["T"])
            self.assertEqual(grade.call_count, 1)

    def test_unknown_and_source_gold_review_stay_null_without_rewriting_candidate(self):
        records = [{"question_id": "q", "candidate_index": 1, "actions": {
            name: {"answer_text": name, "boundary_policy": "known_injected_final_prefix_v1", "error": None}
            for name in ("yes", "no", "unknown")}},
            {"question_id": "review", "candidate_index": 1, "actions": {
                "E": {"answer_text": "yes", "boundary_policy": "known_injected_final_prefix_v1", "error": None}}}]
        def grade(value):
            actual = {"yes": True, "no": False, "unknown": None}[value["answer_text"]]
            return {"id": value["id"], "grade": actual, "status": "needs_review" if actual is None else "graded"}
        with tempfile.TemporaryDirectory() as temporary, \
                patch("math_grading._check_dependencies", return_value={}), \
                patch("math_grading.grade_record", side_effect=grade):
            root = Path(temporary)
            raw = root / "candidate-original.json"
            raw.write_text(json.dumps(records))
            before = raw.read_bytes()
            actual = pilot.grade_saved(root, [row("q"), row("review", needs_review=True)], records)
            self.assertEqual([actual[0]["actions"][name]["error"] for name in ("yes", "no", "unknown")], [0, 1, None])
            self.assertIsNone(actual[1]["actions"]["E"]["error"])
            self.assertEqual(raw.read_bytes(), before)
            saved = json.loads((root / "grading.json").read_text())["results"]
            self.assertEqual(saved[-1]["reason"], "source_gold_requires_review")
            self.assertIsNone(saved[-1]["grade"])

    def test_first_execution_failure_aborts_without_retry_or_next_question(self):
        rows = [row(f"d{i}") for i in range(20)]
        identity, hashes = {"data": "frozen"}, {"source": "fixed"}
        plan = {"question_ids": [value["id"] for value in rows], "code_sha256": hashes,
                "data_identity": identity}
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            run = base / "run"
            durable = {"question_id": "d0", "candidate_index": 1, "actions": {}}
            def fail_after_saved_candidate(backend, row, number, root, log, started, need_gate):
                folder = root / "question-01"
                folder.mkdir()
                (folder / "candidate-01.json").write_text(json.dumps(durable))
                (root / "gpu-acceptance.json").write_text(json.dumps({"passed": True}))
                raise RuntimeError("synthetic later-candidate failure")
            with patch.object(pilot, "make_plan", return_value=(rows, plan)), \
                    patch.object(pilot, "select_questions", return_value=(rows, identity)), \
                    patch.object(pilot, "source_hashes", return_value=hashes), \
                    patch("math_grading._check_dependencies", return_value={}), \
                    patch.object(pilot, "gpu_inventory", return_value={"name": "fake RTX4090", "uuid": "fake"}), \
                    patch.object(pilot, "gpu_lock", return_value=nullcontext()), \
                    patch.object(pilot, "assert_gpu_idle", return_value={}), \
                    patch("torch_online_backend.TorchOnlineBackend", return_value=FakeBackend()), \
                    patch.object(pilot, "collect_question", side_effect=fail_after_saved_candidate) as collect, \
                    patch.object(pilot, "grade_saved", side_effect=lambda root, rows, records, **kwargs: records), \
                    redirect_stdout(io.StringIO()):
                result = pilot.main(["--data-dir", str(base / "data"), "--model-dir", str(base / "model"), "--run-root", str(run)])
            self.assertEqual(result, 1)
            self.assertEqual(collect.call_count, 1)
            self.assertEqual(collect.call_args.args[1]["id"], "d0")
            summary = json.loads((run / "summary.json").read_text())
            self.assertEqual(summary["planned_questions"], 20)
            self.assertEqual(summary["completed_questions"], 0)
            self.assertEqual(summary["saved_candidates"], 1)
            self.assertEqual(summary["status"], "failed")
            # collect_question没有正常返回，主循环need_gate仍True；回执仍应反映已保存验收。
            self.assertTrue(summary["gpu_acceptance_passed"])
            analysis = json.loads((run / "analysis-input.json").read_text())
            self.assertEqual(analysis["question_ids"], plan["question_ids"])
            self.assertEqual(analysis["records"], [durable])


if __name__ == "__main__":
    unittest.main()
