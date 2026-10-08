"""CPU orchestration fixtures for the fixed development matrix, not GPU evidence."""
from contextlib import ExitStack, nullcontext, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_online_development as development

spec = importlib.util.spec_from_file_location("long_context_fixtures", ROOT / "tests/test_run_online_long_context.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
write = fixtures.write


class FakeClock:
    value = 0.0

    def __call__(self):
        return self.value


class DevelopmentTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.LongContextTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, self.data = self.fixture.root, self.fixture.data
        self.metadata = self.fixture.metadata
        # q001/q002 故意不放在前两行，防止错误地使用切片 [2:12]。
        manifest_path = self.data / "manifest.json"
        manifest = development.read_json(manifest_path)
        for name in ("selection120.jsonl", "pilot20.jsonl"):
            path = self.data / name
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["id"] = "selection/0"
            rows[2]["id"], rows[8]["id"] = development.EXCLUDED_IDS
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            manifest["files"][name]["ids"] = [row["id"] for row in rows]
            manifest["files"][name]["sha256"] = development.sha256(path)
        write(manifest_path, manifest)
        self.fixture.data_sha = development.sha256(manifest_path)
        self.a07 = self.root / "a07"
        self.a07.mkdir()
        with patch.object(development.long_run, "SOURCE_SHA256", self.fixture.source_sha):
            _, base = development.long_run.prepare(self.fixture.args())
        write(self.a07 / "manifest.json", base)
        for name in ("integrity-before.json", "integrity-after.json"):
            write(self.a07 / name, base["source_input_sha256"])
        tokens = [7] * 8191 + [22, 15, 24]
        record = {"status": "completed", "method": "vanilla", "sample_id": development.EXCLUDED_IDS[1],
            "max_new_tokens": 32768, "seed_reason": base["seed_reason"], "reason_tokens": 8192,
            "main_generated_tokens": len(tokens), "stop_reason": "natural_eos", "injected_prompt_tokens": 0,
            "finalization_tokens": 0, "answer_boundary": {"source": "natural_first_end_think"},
            "backend_metadata": self.metadata, "output_token_ids": tokens,
            "main_samples": [{"token_id": token} for token in tokens]}
        write(self.a07 / "vanilla.json", record)
        self.manifest_sha, self.request_sha = development.sha256(self.a07 / "manifest.json"), development.sha256(self.a07 / "vanilla.json")
        write(self.a07 / "summary.json", {"status": "completed", "planned_count": 1,
            "executed_count": 1, "unexecuted_count": 0, "source_input_identity_integrity": "unchanged",
            "requests": [{"source_sha256": self.request_sha}]})

    def args(self, **updates):
        f = self.fixture
        args = development.parser().parse_args(["--data-dir", str(f.data), "--data-manifest-sha256", f.data_sha,
            "--gate-summary", str(f.gate), "--capacity-root", str(f.capacity),
            "--natural-run-root", str(f.natural), "--long-context-root", str(self.a07),
            "--model-dir", str(f.model), "--run-root", str(self.root / "development")])
        for name, value in updates.items():
            setattr(args, name, value)
        return args

    def anchors(self, stack):
        stack.enter_context(patch.object(development.long_run, "SOURCE_SHA256", self.fixture.source_sha))
        stack.enter_context(patch.object(development, "A07_MANIFEST_SHA256", self.manifest_sha))
        stack.enter_context(patch.object(development, "A07_REQUEST_SHA256", self.request_sha))

    def prepare(self):
        with ExitStack() as stack:
            self.anchors(stack)
            return development.prepare(self.args())

    def execute(self, backend=None, clock=None, factory=None, **updates):
        backend = backend or fixtures.CPUBackend(self.metadata)
        factory = factory or Mock(return_value=backend)
        gpu = Mock(return_value={"uuid": "GPU-test", "name": "CPU fixture"})
        with ExitStack() as stack:
            self.anchors(stack)
            stack.enter_context(patch.object(development, "gpu_inventory", gpu))
            stack.enter_context(patch.object(development, "gpu_lock", side_effect=lambda _: nullcontext("test-lock")))
            stack.enter_context(patch.object(development, "assert_gpu_idle", return_value={"compute_processes": []}))
            stack.enter_context(redirect_stdout(io.StringIO()))
            code = development.run_development(self.args(**updates), backend_factory=factory,
                                               now=clock or FakeClock())
        return code, backend, factory, gpu

    def test_fixed_original_order_exclusions_rotations_and_paired_seeds(self):
        rows, configs, manifest = self.prepare()
        self.assertEqual([q["source_row_index"] for q in manifest["questions"]], [0, 1, 3, 4, 5, 6, 7, 9, 10, 11])
        self.assertEqual([q["development_id"] for q in manifest["questions"]], [f"dev{i:02d}" for i in range(1, 11)])
        self.assertFalse(set(development.EXCLUDED_IDS) & {row["id"] for row in rows})
        self.assertEqual(len(manifest["requests"]), 40)
        self.assertEqual(configs["codestop-fixed"]["schedule_config"].fixed_interval, 4)
        for i in range(10):
            jobs = manifest["requests"][4*i:4*i+4]
            expected = development.LABELS[i % 4:] + development.LABELS[:i % 4]
            self.assertEqual(tuple(j["configuration"] for j in jobs), expected)
            self.assertEqual(len({j["request_configuration"]["seed_reason"] for j in jobs}), 1)
        self.assertEqual(len({j["request_configuration"]["seed_reason"] for j in manifest["requests"]}), 10)
        self.assertEqual(len({j["relative_directory"] for j in manifest["requests"]}), 40)

    def test_missing_or_tampered_long_trajectory_refuses_before_GPU(self):
        record = development.read_json(self.a07 / "vanilla.json")
        record["reason_tokens"] = 8191
        write(self.a07 / "vanilla.json", record)
        with self.assertRaisesRegex(ValueError, "frozen completed a07"):
            self.execute()
        # 即使 fixture 的外部锚也变化，真实长轨迹条件仍然拒绝。
        self.request_sha = development.sha256(self.a07 / "vanilla.json")
        summary = development.read_json(self.a07 / "summary.json")
        summary["requests"][0]["source_sha256"] = self.request_sha
        write(self.a07 / "summary.json", summary)
        with self.assertRaisesRegex(ValueError, "actual long"):
            self.execute()
        self.assertFalse((self.root / "development").exists())

    def test_inspect_only_has_no_GPU_or_writes_and_import_does_not_load_torch(self):
        code, _, factory, gpu = self.execute(inspect_only=True)
        self.assertEqual(code, 0)
        factory.assert_not_called()
        gpu.assert_not_called()
        self.assertFalse((self.root / "development").exists())
        code = "import sys; sys.path.insert(0,'scripts'); import run_online_development; assert 'torch' not in sys.modules"
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_context_allows_larger_than_old_prompt_without_filtering_and_checks_model_limit(self):
        rows, _, manifest = self.prepare()
        check = development.prompt_preflight(rows, manifest, lambda _: list(range(200)), 40000, (22, 9, 10), (9, 10))
        self.assertTrue(check["all_within_model_context"])
        self.assertEqual(len(check["questions"]), 10)
        self.assertTrue(all(item["exceeds_prior_synthetic_context"] for item in check["questions"]))
        check = development.prompt_preflight(rows, manifest, lambda _: list(range(8000)), 40000, (22, 9, 10), (9, 10))
        self.assertFalse(check["all_within_model_context"])
        self.assertEqual(len(check["questions"]), 10)
        backend = fixtures.CPUBackend(self.metadata)
        backend.prompt_ids = lambda _: list(range(8000))
        code, backend, _, _ = self.execute(backend)
        self.assertEqual((code, backend.requests), (1, 0))
        summary = development.read_json(self.root / "development/summary.json")
        self.assertEqual((summary["executed_count"], summary["unexecuted_count"]), (0, 40))

    def test_complete_matrix_atomic_rows_journal_and_seed_state_reset(self):
        code, backend, factory, _ = self.execute()
        self.assertEqual((code, backend.requests, factory.call_count), (0, 41, 1))
        root = self.root / "development"
        manifest, summary = development.read_json(root / "manifest.json"), development.read_json(root / "summary.json")
        self.assertEqual((summary["planned_count"], summary["executed_count"], summary["failed_count"], summary["unexecuted_count"]), (40, 40, 0, 0))
        self.assertEqual(backend.seeds[1:], [j["request_configuration"]["seed_reason"] for j in manifest["requests"]])
        self.assertNotIn(backend.seeds[0], backend.seeds[1:])
        journal, final = (root / "answers.jsonl").read_bytes(), (root / "final_answers.jsonl").read_bytes()
        self.assertEqual(journal, final)
        self.assertEqual(len(final.splitlines()), 40)
        for line in final.splitlines():
            answer = json.loads(line)
            self.assertEqual(answer["source_sha256"], development.sha256(root / answer["source_file"]))
        self.assertEqual(len(list(root.glob("requests/*/*/request.json"))), 40)
        self.assertEqual(summary["grading_status"], "not_run")

    def test_mid_matrix_OOM_saves_failed_row_and_full_unexecuted_denominator(self):
        backend = fixtures.CPUBackend(self.metadata)
        original = backend.sample
        def fail_second(state):
            if backend.requests == 3 and backend.count == 2:
                raise RuntimeError("synthetic CUDA OOM")
            return original(state)
        backend.sample = fail_second
        code, backend, factory, _ = self.execute(backend)
        root = self.root / "development"
        summary = development.read_json(root / "summary.json")
        self.assertEqual((code, backend.requests, factory.call_count), (1, 3, 1))
        self.assertEqual((summary["executed_count"], summary["failed_count"], summary["unexecuted_count"]), (2, 1, 38))
        self.assertEqual(len((root / "answers.jsonl").read_text().splitlines()), 2)
        failed = development.read_json(root / summary["requests"][-1]["relative_directory"] / "request.json")
        self.assertEqual(len(failed["main_samples"]), 2)
        self.assertFalse(development.read_json(root / "failure.json")["retry_performed"])

    def test_per_request_timeout_keeps_just_sampled_token_pending_and_stops(self):
        clock = FakeClock()
        backend = fixtures.CPUBackend(self.metadata)
        original = backend.sample
        def advance(state):
            token = original(state)
            if backend.requests == 2 and backend.count == 1:
                clock.value = 1801
            return token
        backend.sample = advance
        code, _, _, _ = self.execute(backend, clock=clock)
        root = self.root / "development"
        summary = development.read_json(root / "summary.json")
        self.assertEqual((code, summary["executed_count"], summary["failed_count"], summary["unexecuted_count"]), (1, 1, 1, 39))
        request = development.read_json(root / "requests/dev01/vanilla/request.json")
        self.assertEqual(len(request["main_samples"]), 1)
        self.assertIsNotNone(request["pending_sample"])
        self.assertIn("request_wall_limit", request["error"])

    def test_total_deadline_during_model_load_prevents_warmup_and_all_requests(self):
        clock = FakeClock()
        backend = fixtures.CPUBackend(self.metadata)
        def load(*args, **kwargs):
            clock.value = 7201
            return backend
        code, backend, _, _ = self.execute(backend, clock=clock, factory=Mock(side_effect=load))
        summary = development.read_json(self.root / "development/summary.json")
        self.assertEqual((code, backend.requests, summary["executed_count"], summary["unexecuted_count"]), (1, 0, 0, 40))
        self.assertIn("total_wall_limit", summary["failure"]["message"])

    def test_total_deadline_overrides_later_request_allowance(self):
        clock = FakeClock()
        budget = development.WallBudget(clock)
        clock.value = 7100
        budget.begin_request()
        clock.value = 7200
        with self.assertRaisesRegex(development.BudgetExceeded, "total_wall_limit"):
            development.DeadlineBackend(fixtures.CPUBackend(self.metadata), budget).prefill([1, 2])

    def test_keyboard_interrupt_is_honest_failed_row_not_dropped(self):
        code, _, _, _ = self.execute(fixtures.CPUBackend(self.metadata, failure=KeyboardInterrupt()))
        root = self.root / "development"
        summary = development.read_json(root / "summary.json")
        self.assertEqual((code, summary["executed_count"], summary["failed_count"], summary["unexecuted_count"]), (1, 1, 1, 39))
        record = development.read_json(root / "requests/dev01/vanilla/request.json")
        self.assertFalse(record["partial_evidence_available"])
        self.assertEqual(record["error_type"], "KeyboardInterrupt")

    def test_mutation_and_existing_output_preserve_original_evidence(self):
        path = self.fixture.natural / "third-request-plan.json"
        backend = fixtures.CPUBackend(self.metadata, mutation=lambda: write(path, {"changed": True}))
        code, _, _, _ = self.execute(backend)
        root = self.root / "development"
        self.assertEqual(code, 1)
        self.assertEqual(development.read_json(root / "summary.json")["source_input_identity_integrity"], "changed")
        saved = (root / "manifest.json").read_bytes()
        with self.assertRaises(ValueError):
            self.execute()
        self.assertEqual((root / "manifest.json").read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
