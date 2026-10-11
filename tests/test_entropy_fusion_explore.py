"""新范围/预算/失败恢复测试；使用 fake backend，不执行真实模型。"""
from contextlib import nullcontext, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import run_entropy_fusion_explore as explore
from test_entropy_value_pilot import FakeBackend, row


def write_rows(root, name, rows):
    (root / name).write_text("".join(json.dumps(value) + "\n" for value in rows))


class ScopeTests(unittest.TestCase):
    def test_gpu_guard_accepts_verified_models_and_rejects_other_or_partial_names(self):
        for name in ("NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 5090"):
            explore.check_gpu_model(name)
        for name in ("NVIDIA A100", "RTX 4090D", "fake RTX5090", "RTX 50900"):
            with self.assertRaisesRegex(RuntimeError, "checked RTX"):
                explore.check_gpu_model(name)

    def test_excludes_both_exposed_batches_round_robins_and_does_not_replace_short_subject(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pilot = [row("p", subject="A")]
            previous = [row("old", subject="B")]
            fresh = [row(f"{subject}{i}", subject=subject)
                for subject in "ABCDEFG" for i in range(1 if subject == "G" else 3)]
            write_rows(root, "pilot20.jsonl", pilot)
            write_rows(root, "selection120.jsonl", pilot + previous + fresh)
            with patch.object(explore.old, "select_questions", return_value=(previous, {"frozen": True})):
                selected, identity = explore.select_questions(root)
            self.assertEqual([value["id"] for value in selected],
                [f"{subject}0" for subject in "ABCDEFG"] + [f"{subject}1" for subject in "ABCDEF"])
            self.assertEqual(identity, {"frozen": True})

    def test_selected_missing_gold_fails_without_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fresh = [row(subject, subject=subject) for subject in "ABCDEFG"]
            fresh[0]["answer"] = ""
            write_rows(root, "pilot20.jsonl", [])
            write_rows(root, "selection120.jsonl", fresh)
            with patch.object(explore.old, "select_questions", return_value=([], {})), \
                    self.assertRaisesRegex(ValueError, "do not silently replace"):
                explore.select_questions(root)

    def test_plan_records_exact_frozen_hash_and_budget_without_new_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen = root / "frozen.json"
            frozen.write_text('{"trained": "before_collection"}')
            rows = [row("q", subject="Algebra")]
            base_plan = {"limits": {"main_tokens": 8192, "short_probe_tokens": 21,
                "long_probe_tokens": 42, "T_main_tokens": 256, "final_tokens": 128}}
            with patch.object(explore, "select_questions", return_value=(rows, {"fixed": True})), \
                    patch.object(explore.old, "make_plan", return_value=([], base_plan)), \
                    patch.object(explore, "source_hashes", return_value={"source": "hash"}):
                _, plan = explore.make_plan(root, frozen, 600)
            self.assertEqual(plan["frozen_selector"]["sha256"], explore.old.sha256(frozen))
            self.assertEqual(plan["module_overrides"],
                {"QUESTION_COUNT": 1, "TOTAL_SECONDS": 600, "QUESTION_SECONDS": 900})
            self.assertEqual(plan["limits"]["main_tokens"], 8192)
            self.assertEqual(plan["analysis"]["training"], "none; selectors frozen before collection")
            self.assertEqual(plan["cpu_grading"]["seconds"], 90)

    def test_budget_limits_and_module_values_restore_after_failure(self):
        before = (explore.old.QUESTION_COUNT, explore.old.TOTAL_SECONDS, explore.old.QUESTION_SECONDS)
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with explore.collection_limits(14, 600):
                self.assertEqual((explore.old.QUESTION_COUNT, explore.old.TOTAL_SECONDS,
                    explore.old.QUESTION_SECONDS), (14, 600, 900))
                raise RuntimeError("fixture")
        self.assertEqual((explore.old.QUESTION_COUNT, explore.old.TOTAL_SECONDS,
            explore.old.QUESTION_SECONDS), before)
        self.assertEqual(explore.bounded_seconds("60"), 60)
        self.assertEqual(explore.bounded_seconds("3000"), 3000)
        for value in (59, 3001):
            with self.assertRaises(Exception):
                explore.bounded_seconds(value)


class RecoveryTests(unittest.TestCase):
    def run_fake_failure(self, exception):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            run, frozen = base / "run", base / "frozen.json"
            frozen.write_text('{}')
            rows = [row("q0", subject="Algebra"), row("q1", subject="Geometry")]
            hashes, identity = {"source": "fixed"}, {"data": "fixed"}
            plan = {"question_ids": [value["id"] for value in rows],
                "code_sha256": hashes, "data_identity": identity,
                "frozen_selector": {"sha256": explore.old.sha256(frozen)}}
            durable = {"question_id": "q0", "candidate_index": 1, "actions": {
                "E": {"error": None, "answer_text": "unknown", "boundary_policy": "known_injected_final_prefix_v1"}}}
            def fail_after_save(backend, selected, number, destination, log, started, need_gate):
                self.assertEqual(explore.old.QUESTION_COUNT, 2)
                self.assertEqual(explore.old.TOTAL_SECONDS, 60)
                folder = destination / "question-01"
                folder.mkdir()
                (folder / "candidate-01.json").write_text(json.dumps(durable))
                (destination / "gpu-acceptance.json").write_text('{"passed": true}')
                raise exception
            def grade(destination, selected, records, deadline):
                # GPU timeout does not poison the new CPU clock, and unknown remains null.
                deadline()
                self.assertEqual(records, [durable])
                self.assertIsNone(records[0]["actions"]["E"]["error"])
                return records
            with patch.object(explore, "make_plan", return_value=(rows, plan)), \
                    patch.object(explore, "select_questions", return_value=(rows, identity)), \
                    patch.object(explore, "source_hashes", return_value=hashes), \
                    patch("math_grading._check_dependencies", return_value={}), \
                    patch.object(explore.old, "gpu_inventory", return_value={"name": "NVIDIA GeForce RTX 4090", "uuid": "fake"}), \
                    patch.object(explore.old, "gpu_lock", return_value=nullcontext()), \
                    patch.object(explore.old, "assert_gpu_idle", return_value={}), \
                    patch("torch_online_backend.TorchOnlineBackend", return_value=FakeBackend()), \
                    patch.object(explore.old, "collect_question", side_effect=fail_after_save) as collect, \
                    patch.object(explore.old, "grade_saved", side_effect=grade) as grading, \
                    redirect_stdout(io.StringIO()):
                result = explore.main(["--data-dir", str(base / "data"), "--model-dir", str(base / "model"),
                    "--run-root", str(run), "--frozen-selector", str(frozen), "--max-seconds", "60"])
            self.assertEqual(collect.call_count, 1)
            self.assertEqual(grading.call_count, 1)
            summary = json.loads((run / "summary.json").read_text())
            self.assertEqual(summary["saved_candidates"], 1)
            self.assertEqual(summary["completed_questions"], 0)
            self.assertTrue(summary["source_data_selector_integrity"])
            self.assertTrue(summary["gpu_acceptance_passed"])
            self.assertEqual(json.loads((run / "question-01/candidate-01.json").read_text()), durable)
            return result, summary

    def test_budget_exhaustion_keeps_candidate_and_scores_with_independent_clock(self):
        result, summary = self.run_fake_failure(TimeoutError("Fixed pilot wall-clock budget exhausted"))
        self.assertEqual(result, 0)
        self.assertEqual(summary["status"], "budget_exhausted")
        self.assertIsNone(summary["failure"])
        self.assertIsNone(summary["grading_failure"])

    def test_first_execution_error_keeps_candidate_and_aborts_without_retry(self):
        result, summary = self.run_fake_failure(RuntimeError("synthetic OOM"))
        self.assertEqual(result, 1)
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["failure"]["message"], "synthetic OOM")
        self.assertIsNone(summary["budget_stop"])

    def test_existing_output_is_rejected_without_gpu_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(explore, "make_plan", return_value=([], {})), \
                    patch.object(explore.old, "gpu_inventory") as gpu, \
                    self.assertRaisesRegex(ValueError, "must be new"):
                explore.main(["--data-dir", str(root / "data"), "--model-dir", str(root / "model"),
                    "--run-root", str(root), "--frozen-selector", str(root / "frozen.json")])
            gpu.assert_not_called()


if __name__ == "__main__":
    unittest.main()
