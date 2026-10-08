"""Synthetic CPU orchestration/evidence fixtures, never GPU collection evidence."""
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
import run_online_dense_collection as collection
from online_contract import TokenSample
from online_protocol import ProbeObservation

spec = importlib.util.spec_from_file_location("development_fixtures", ROOT / "tests/test_run_online_development.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
write = fixtures.write


class ProbeBackend(fixtures.fixtures.CPUBackend):
    def sample(self, state):
        if self.requests == 1:
            return super().sample(state)
        self.count += 1
        # Both probes would stop on confidence. Collection must continue anyway.
        return TokenSample((7, 21, 8, 21, 22, 15, 24)[self.count - 1], 1.0, 0.8)

    def probe(self, state):
        return ProbeObservation((7, 8, 22), (0.9, 0.99, 0.99), 0.99, True,
                                confidence_source="synthetic_cpu_collection_fixture")


class DenseCollectionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.DevelopmentTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, self.metadata = self.fixture.root, self.fixture.metadata
        self.a08 = self.root / "a08-development"
        code, _, _, _ = self.fixture.execute(run_root=self.a08)
        self.assertEqual(code, 0)
        self.manifest_sha = collection.sha256(self.a08 / "manifest.json")
        self.summary_sha = collection.sha256(self.a08 / "summary.json")

    def args(self, **updates):
        args = self.fixture.args(run_root=self.root / "dense-collection")
        args.development_run = self.a08
        for key, value in updates.items():
            setattr(args, key, value)
        return args

    def anchors(self, stack):
        self.fixture.anchors(stack)
        stack.enter_context(patch.object(collection, "A08_MANIFEST_SHA256", self.manifest_sha))
        stack.enter_context(patch.object(collection, "A08_SUMMARY_SHA256", self.summary_sha))

    def prepare(self):
        with ExitStack() as stack:
            self.anchors(stack)
            return collection.prepare(self.args())

    def execute(self, backend=None, clock=None, factory=None, **updates):
        backend = backend or ProbeBackend(self.metadata)
        factory = factory or Mock(return_value=backend)
        gpu = Mock(return_value={"uuid": "GPU-test", "name": "CPU fixture"})
        with ExitStack() as stack:
            self.anchors(stack)
            stack.enter_context(patch.object(collection, "gpu_inventory", gpu))
            stack.enter_context(patch.object(collection, "gpu_lock", side_effect=lambda _: nullcontext("synthetic lock")))
            stack.enter_context(patch.object(collection, "assert_gpu_idle", return_value={"compute_processes": []}))
            stack.enter_context(redirect_stdout(io.StringIO()))
            code = collection.run_collection(self.args(**updates), backend_factory=factory,
                                             now=clock or fixtures.FakeClock())
        return code, backend, factory, gpu

    def test_plan_is_exact_a08_questions_seeds_and_no_stop_contract(self):
        rows, config, manifest = self.prepare()
        old = collection.read_json(self.a08 / "manifest.json")
        self.assertEqual(manifest["questions"], old["questions"])
        self.assertEqual((len(rows), manifest["planned_count"], len(manifest["requests"])), (10, 10, 10))
        self.assertEqual(config["method"], "dense_collect_no_stop")
        self.assertEqual(manifest["limits"]["per_request_wall_seconds"], 1800)
        self.assertEqual(manifest["limits"]["total_wall_seconds"], 7200)
        for job in manifest["requests"]:
            prior = next(item for item in old["requests"] if item["sample_id"] == job["sample_id"])
            self.assertEqual(job["request_configuration"]["seed_reason"], prior["request_configuration"]["seed_reason"])
            self.assertFalse(job["request_configuration"]["stopping_enabled"])
            self.assertEqual(job["request_configuration"]["max_new_tokens"], 32768)
        self.assertTrue(set(collection.development.SOURCE_FILES) <= set(manifest["code_sha256"]))
        self.assertIn("scripts/run_online_dense_collection.py", manifest["code_sha256"])
        self.assertEqual(len(manifest["development_evidence"]["paired_sources"]), 10)

    def test_a08_hash_and_raw_request_tampering_are_rejected_before_GPU(self):
        summary_path = self.a08 / "summary.json"
        summary = collection.read_json(summary_path)
        summary["executed_count"] = 39
        write(summary_path, summary)
        with self.assertRaisesRegex(ValueError, "frozen completed a08"):
            self.execute()
        self.summary_sha = collection.sha256(summary_path)
        with self.assertRaisesRegex(ValueError, "completion/source-input"):
            self.execute()
        summary["executed_count"] = 40
        write(summary_path, summary)
        self.summary_sha = collection.sha256(summary_path)
        path = self.a08 / summary["requests"][0]["relative_directory"] / "request.json"
        value = collection.read_json(path)
        value["seed_reason"] += 1
        write(path, value)
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.execute()
        self.assertFalse((self.root / "dense-collection").exists())

    def test_a08_question_order_is_rejected_even_with_fixture_anchor_updated(self):
        path = self.a08 / "manifest.json"
        manifest = collection.read_json(path)
        manifest["questions"] = list(reversed(manifest["questions"]))
        write(path, manifest)
        self.manifest_sha = collection.sha256(path)
        with self.assertRaisesRegex(ValueError, "frozen development plan: questions"):
            self.execute()

    def test_full_collection_retains_would_stop_and_continues_all_candidates(self):
        code, backend, factory, _ = self.execute()
        self.assertEqual((code, backend.requests, factory.call_count), (0, 11, 1))
        root = self.root / "dense-collection"
        summary = collection.read_json(root / "summary.json")
        self.assertEqual((summary["planned_count"], summary["executed_count"], summary["unexecuted_count"]), (10, 10, 0))
        self.assertFalse(summary["eligible_for_primary_speed_comparison"])
        self.assertTrue(summary["collection_cost_only"])
        self.assertEqual((root / "answers.jsonl").read_bytes(), (root / "final_answers.jsonl").read_bytes())
        for job in summary["requests"]:
            directory = root / job["relative_directory"]
            record = collection.read_json(directory / "request.json")
            self.assertEqual(record["method"], "dense_collect_no_stop")
            self.assertEqual((record["n_candidates"], record["n_probes"]), (2, 2))
            self.assertEqual(record["main_generated_tokens"], 7)
            self.assertTrue(all(probe["would_stop"] and not probe["stop_applied"] for probe in record["probes"]))
            self.assertTrue(record["collection_complete_to_termination_or_cap"])
            self.assertEqual(record["scope"], collection.SCOPE)
            answer = collection.read_json(directory / "final-answer.json")
            self.assertEqual(answer["source_sha256"], collection.sha256(directory / "request.json"))
            self.assertEqual(answer["execution_status"], "completed")
        saved = (root / "manifest.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "must be new"):
            self.execute()
        self.assertEqual((root / "manifest.json").read_bytes(), saved)

    def test_second_request_failure_saves_partial_and_eight_unexecuted(self):
        backend = ProbeBackend(self.metadata)
        original = backend.sample
        def failing(state):
            if backend.requests == 3 and backend.count == 2:
                raise RuntimeError("synthetic CUDA OOM")
            return original(state)
        backend.sample = failing
        code, _, _, _ = self.execute(backend)
        root = self.root / "dense-collection"
        summary = collection.read_json(root / "summary.json")
        self.assertEqual((code, summary["executed_count"], summary["failed_count"], summary["unexecuted_count"]), (1, 2, 1, 8))
        path = root / summary["requests"][-1]["relative_directory"] / "request.json"
        partial = collection.read_json(path)
        self.assertEqual(len(partial["main_samples"]), 2)
        self.assertEqual(len(partial["probes"]), 1)
        self.assertFalse(partial["collection_complete_to_termination_or_cap"])
        self.assertEqual(len((root / "final_answers.jsonl").read_text().splitlines()), 2)

    def test_timeout_keeps_pending_token_and_fixed_unexecuted_denominator(self):
        clock = fixtures.FakeClock()
        backend = ProbeBackend(self.metadata)
        original = backend.sample
        def advance(state):
            token = original(state)
            if backend.requests == 2 and backend.count == 1:
                clock.value = 1801
            return token
        backend.sample = advance
        code, _, _, _ = self.execute(backend, clock)
        root = self.root / "dense-collection"
        summary = collection.read_json(root / "summary.json")
        self.assertEqual((code, summary["executed_count"], summary["unexecuted_count"]), (1, 1, 9))
        partial = collection.read_json(root / summary["requests"][0]["relative_directory"] / "request.json")
        self.assertEqual(len(partial["main_samples"]), 1)
        self.assertIsNotNone(partial["pending_sample"])
        self.assertIn("request_wall_limit", partial["error"])

    def test_total_timeout_during_load_prevents_all_requests(self):
        clock = fixtures.FakeClock()
        backend = ProbeBackend(self.metadata)
        def load(*args, **kwargs):
            clock.value = 7201
            return backend
        code, _, _, _ = self.execute(backend, clock, Mock(side_effect=load))
        summary = collection.read_json(self.root / "dense-collection/summary.json")
        self.assertEqual((code, backend.requests, summary["executed_count"], summary["unexecuted_count"]), (1, 0, 0, 10))
        self.assertIn("total_wall_limit", summary["failure"]["message"])

    def test_a08_mutation_during_collection_stops_before_next_request(self):
        backend = ProbeBackend(self.metadata)
        original = backend.sample
        def mutate(state):
            token = original(state)
            if backend.requests == 2 and backend.count == 1:
                write(self.a08 / "environment.json", {"synthetic_mutation": True})
            return token
        backend.sample = mutate
        code, _, _, _ = self.execute(backend)
        summary = collection.read_json(self.root / "dense-collection/summary.json")
        self.assertEqual((code, summary["executed_count"], summary["unexecuted_count"]), (1, 1, 9))
        self.assertEqual(summary["source_input_identity_integrity"], "changed")
        self.assertEqual(summary["failure"]["phase"], "between_requests")

    def test_inspect_and_tokenizer_modes_are_read_only_without_GPU(self):
        code, _, factory, gpu = self.execute(inspect_only=True)
        self.assertEqual(code, 0)
        factory.assert_not_called()
        gpu.assert_not_called()
        with patch.object(collection.development, "tokenizer_preflight", return_value={"all_within_model_context": True}) as tokenizer:
            code, _, factory, gpu = self.execute(preflight_tokenizer_only=True)
            self.assertEqual(code, 0)
            tokenizer.assert_called_once()
            factory.assert_not_called()
            gpu.assert_not_called()
        self.assertFalse((self.root / "dense-collection").exists())
        code = "import sys; sys.path.insert(0,'scripts'); import run_online_dense_collection; assert 'torch' not in sys.modules"
        child = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(child.returncode, 0, child.stderr)


if __name__ == "__main__":
    unittest.main()
