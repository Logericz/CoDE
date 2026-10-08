"""Synthetic CPU gate/orchestration tests; none establish GPU or model quality."""
from contextlib import ExitStack, nullcontext, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_online_long_context as long_run
from online_contract import TokenMarkers, TokenSample

spec = importlib.util.spec_from_file_location("bounded_runner_fixtures", ROOT / "tests/test_run_online_diagnostic.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class CPUBackend:
    """Injected engine fixture; metadata is supplied by synthetic gate fixtures."""
    markers = TokenMarkers(21, 22, 23, (9, 10), (22, 9, 10), eos_ids=(23, 24))
    context_limit = 40000

    def __init__(self, metadata, *, failure=None, mutation=None):
        self.metadata = metadata.copy()
        self.failure, self.mutation = failure, mutation
        self.requests, self.seeds, self.questions = 0, [], []

    def prompt_ids(self, question):
        self.questions.append(question)
        return [1, 2]

    def start_request(self, seed):
        self.requests += 1
        self.seeds.append(seed)
        self.count = self.final_count = 0

    def prefill(self, ids):
        return tuple(ids)

    def sample(self, state):
        self.count += 1
        if self.requests == 2 and self.count == 3:
            if self.mutation:
                self.mutation()
            if self.failure:
                raise self.failure
        return TokenSample(22 if self.count == 65 else 24 if self.count == 66 else 7, 1.0, 0.8)

    def greedy(self, state):
        self.final_count += 1
        return TokenSample(15 if self.final_count == 1 else 24, 1.0, 0.9)

    def extend(self, state, ids):
        return state + tuple(ids)

    def decode(self, ids, *, skip_special_tokens=True):
        return "".join({15: r"\boxed{7}", 22: "</think>", 24: "<eos>"}.get(i, "x") for i in ids)

    def synchronize(self):
        pass

    def peak_memory_bytes(self):
        return None


class LongContextTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.data, self.model = self.root / "data", self.root / long_run.MODEL_REVISION
        self.model.mkdir()
        self.data_sha = fixtures.frozen_fixture(self.data)
        markers = CPUBackend.markers
        self.metadata = {key: None for key in long_run.ENVIRONMENT_FIELDS}
        self.metadata.update(model_id=long_run.MODEL_ID, model_revision=long_run.MODEL_REVISION,
            tokenizer_revision=long_run.MODEL_REVISION, model_parameter_dtype="torch.bfloat16",
            attention_implementation="eager", validation_model=False, device="cuda:0",
            scope="synthetic_cpu_orchestration_fixture", context_limit=40000,
            gpu_total_memory_bytes=24 * 2**30,
            model_config={"num_hidden_layers": 36, "num_key_value_heads": 8, "head_dim": 128,
                          "max_position_embeddings": 40000},
            markers={"wait": markers.wait, "end_think": markers.end_think, "eos": markers.eos,
                     "eos_ids": list(markers.eos_ids), "trial_prefix": list(markers.trial_prefix),
                     "final_prefix": list(markers.final_prefix)})
        self.critical = {name: long_run.sha256(ROOT / name) for name in long_run.CRITICAL}
        self.gate = self.root / "gate/summary.json"
        self.gate_record = {"schema_version": 1, "gate_passed": True,
            "status": "passed_same_KV_contract_bounded_validation",
            "reference_contract": "incremental_main_KV_pinned_original_probe_exact_v1",
            "source_input_identity_integrity": "unchanged",
            "completed_prefixes": [{"case": case, "same_KV_gate": {"status": "passed"}}
                for case in ("q001-probe-0", "q001-probe-1", "q002-probe-0", "q002-probe-1", "q002-probe-2")],
            "isolation": {"case": "q002-probe-0", "status": "passed", "variants": [
                {"inserted_probes": n, "continuation_tokens": 64, "exact": True} for n in (0, 1, 2)]}}
        write(self.gate, self.gate_record)
        write(self.gate.parent / "source-hashes.json", self.critical)
        write(self.gate.parent / "environment.json", self.metadata)
        for name in ("integrity-before.json", "integrity-after.json"):
            write(self.gate.parent / name, {str(ROOT / n): v for n, v in self.critical.items()})
        self.natural = self.root / "natural"
        self.natural_records = {}
        for label, stop in (("early", "degeneration"), ("eos", "natural_eos"), ("answer-budget", "budget")):
            self.natural_records[label] = {"configuration": label, "status": "completed",
                "sample_id": long_run.SAMPLE_ID, "backend_metadata": self.metadata,
                "scope": "single_question_development_gpu_acceptance_not_benchmark",
                "stop_reason": stop, "output_token_ids": [7, 24] if label == "eos" else [7, 22, 15],
                "finalization_tokens": 0 if label == "eos" else 2,
                "probes": [{"stop_applied": True}] if label == "early" else [],
                "injected_prompt_tokens": 0 if label == "answer-budget" else 3,
                "answer_boundary": {"source": "natural_first_end_think" if label == "answer-budget" else "controlled_final_prefix"}}
        write(self.natural / "manifest.json", {"code_sha256": self.critical})
        self.save_natural()
        # Additional derived evidence must be captured even though it is not a
        # required input of the original capacity.validate_natural_runs helper.
        write(self.natural / "third-request-plan.json", {"synthetic_fixture": True})
        self.source = self.root / "original/vanilla.json"
        record = {"sample_id": long_run.SAMPLE_ID, "status": "completed", "method": "vanilla",
            "max_new_tokens": 1024, "backend_metadata": self.metadata, "prompt_token_ids": [1, 2],
            "main_samples": [{"token_id": t, "phase": "reason"} for t in [7, 8] * 512],
            "output_token_ids": [7, 8] * 512 + [22, 9, 10, 15]}
        write(self.source, record)
        self.source_sha = long_run.sha256(self.source)
        self.capacity = self.root / "capacity"
        self.make_capacity()

    def save_natural(self):
        rows = []
        for label, record in self.natural_records.items():
            path = self.natural / f"{label}.json"
            write(path, record)
            rows.append({"configuration": label, "status": "completed", "source_sha256": long_run.sha256(path)})
        write(self.natural / "summary.json", {"status": "completed", "executed_count": len(rows),
            "planned_count": len(rows), "unexecuted_count": 0, "code_unchanged_during_run": True,
            "requests": rows})

    def make_capacity(self):
        gate = long_run.validate_gate(self.gate)
        source = long_run.load_source(self.source, expected_sha256=self.source_sha)
        hashes = {n: long_run.sha256(ROOT / n) for n in long_run.CAPACITY_SOURCE_FILES}
        integrity = {str(ROOT / n): h for n, h in hashes.items()}
        integrity.update({str(self.source): self.source_sha, **gate["input_sha256"]})
        for name in ("integrity-before.json", "integrity-after.json"):
            write(self.capacity / name, integrity)
        write(self.capacity / "source-hashes.json", hashes)
        write(self.capacity / "environment.json", self.metadata)
        write(self.capacity / "source-token-pattern.json", source)
        write(self.capacity / "plan.json", {"prerequisite_gate": gate, "source": {"sha256": self.source_sha}})
        write(self.capacity / "capacity-input.json", {"synthetic_filling": True,
            "prompt_token_ids": [1, 2], "synthetic_main_token_ids": [7, 8] * 16384,
            "reserved_tokens": 33, "context_required": 32803, "context_limit": 40000})
        state = {"length": 32770, "kv": [{}] * 36, "logits": {}, "private_rng": {},
                 "global_cpu_rng": {}, "global_cuda_rng": {}}
        write(self.capacity / "before-probe-state.json", state)
        write(self.capacity / "probe.json", {"state_after": state, "observation": {"token_ids": [9, 10, 22]}})
        write(self.capacity / "finalization.json", {"injected_prefix_ids": [22, 9, 10],
            "samples": [{"token_id": 15}] * 30, "generated_tokens": 30, "end": "answer_budget"})
        progress = []
        for index in range(1, 37):
            filled = min(index - 1, 32) * 1024
            length = 2 + filled + (3 if index >= 35 else 0) + (30 if index == 36 else 0)
            byte_count = 36 * 2 * 8 * length * 128 * 2
            row = {"filled_main_tokens": filled, "kv_lengths_consistent": True,
                "kv_length": length, "committed_state_length": length,
                "kv_layer_shapes": [[[1, 8, length, 128]] * 2 for _ in range(36)],
                "kv_tensor_bytes": byte_count, "allocated_bytes": 8000000000 + byte_count,
                "reserved_bytes": 9000000000 + byte_count, "peak_allocated_bytes": 8500000000 + byte_count,
                "peak_reserved_bytes": 9500000000 + byte_count, "device_total_bytes": 24 * 2**30,
                "device_free_bytes": 1000000000}
            progress.append(row)
            write(self.capacity / f"memory-{index:03d}.json", row)
        write(self.capacity / "summary.json", {"status": "capacity_shape_memory_passed", "scope": long_run.CAPACITY_SCOPE,
            "source_input_identity_integrity": "unchanged", "filled_main_tokens": 32768,
            "main_token_budget": 32768, "probe_state_isolation": "passed", "prerequisite_gate": gate,
            "context_required": 32803, "context_limit": 40000, "probe_tokens": 3,
            "final_KV_length": 32803, "finalization_tokens": 30, "memory_progress": progress})

    def args(self, **changes):
        args = long_run.parser().parse_args(["--data-dir", str(self.data), "--data-manifest-sha256", self.data_sha,
            "--gate-summary", str(self.gate), "--capacity-root", str(self.capacity),
            "--natural-run-root", str(self.natural), "--max-new-tokens", "32768",
            "--model-dir", str(self.model), "--run-root", str(self.root / "new-run")])
        for name, value in changes.items():
            setattr(args, name, value)
        return args

    def execute(self, backend=None, **changes):
        backend = backend or CPUBackend(self.metadata)
        factory = Mock(return_value=backend)
        gpu = Mock(return_value={"uuid": "GPU-test", "name": "CPU fixture"})
        with ExitStack() as stack:
            stack.enter_context(patch.object(long_run, "SOURCE_SHA256", self.source_sha))
            stack.enter_context(patch.object(long_run, "gpu_inventory", gpu))
            stack.enter_context(patch.object(long_run, "gpu_lock", side_effect=lambda _: nullcontext("test-lock")))
            stack.enter_context(patch.object(long_run, "assert_gpu_idle", return_value={"compute_processes": []}))
            stack.enter_context(patch.object(long_run, "memory_snapshot", return_value={"scope": "synthetic_cpu_fixture"}))
            stack.enter_context(redirect_stdout(io.StringIO()))
            code = long_run.run_long_context(self.args(**changes), backend_factory=factory)
        return code, backend, factory, gpu

    def test_inspect_only_pure_validation_and_all_required_evidence(self):
        code, _, factory, gpu = self.execute(inspect_only=True)
        self.assertEqual(code, 0)
        factory.assert_not_called()
        gpu.assert_not_called()
        self.assertFalse((self.root / "new-run").exists())
        with patch.object(long_run, "SOURCE_SHA256", self.source_sha):
            _, manifest = long_run.prepare(self.args())
        self.assertEqual(manifest["seed_reason"], 1937708343865469216)
        self.assertEqual(manifest["max_new_tokens"], 32768)
        self.assertEqual(manifest["max_final_tokens"], 30)
        self.assertIn(str(self.natural / "third-request-plan.json"), manifest["source_input_sha256"])
        self.assertFalse(manifest["eligible_for_primary_speed_comparison"])

    def test_cli_has_no_independent_seed_question_method_or_attention_override(self):
        with redirect_stdout(io.StringIO()), patch("sys.stderr", new_callable=io.StringIO):
            for extra in ("--independent-capacity-diagnostic", "--master-seed", "--sample-id", "--attention-implementation"):
                with self.subTest(option=extra), self.assertRaises(SystemExit):
                    long_run.parser().parse_args([extra, "x"])
        with self.assertRaisesRegex(ValueError, "exactly 32768"):
            self.execute(max_new_tokens=8192)

    def test_missing_branch_refuses_before_GPU_and_does_not_create_output(self):
        del self.natural_records["answer-budget"]
        self.save_natural()
        with patch.object(long_run, "gpu_inventory") as gpu, self.assertRaisesRegex(ValueError, "actual_answer_phase_budget"):
            long_run.run_long_context(self.args())
        gpu.assert_not_called()
        self.assertFalse((self.root / "new-run").exists())

    def test_uncovered_natural_run_and_failed_pairs_refuse(self):
        summary_path = self.natural / "summary.json"
        summary = long_run.read_json(summary_path)
        write(summary_path, {**summary, "status": "uncovered"})
        with self.assertRaisesRegex(ValueError, "complete"):
            self.execute()
        write(summary_path, {**summary, "boundary_pair_passed": True})
        write(self.natural / "boundary-pair-validation.json", {"passed": False})
        with self.assertRaisesRegex(ValueError, "pair evidence"):
            self.execute()

    def test_same_KV_gate_must_have_full_64_token_coverage(self):
        for item in self.gate_record["isolation"]["variants"]:
            item["continuation_tokens"] = 9
        write(self.gate, self.gate_record)
        with self.assertRaisesRegex(ValueError, "64 continuation"):
            self.execute()

    def test_capacity_rechecks_historical_source_hash(self):
        self.source.write_text(self.source.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "Recorded capacity source/input"):
            self.execute()

    def test_capacity_rejects_KV_shape_or_context_reserve_tampering(self):
        path = self.capacity / "memory-033.json"
        memory = long_run.read_json(path)
        memory["kv_layer_shapes"][0][0][2] -= 1
        write(path, memory)
        with self.assertRaisesRegex(ValueError, "memory snapshot"):
            self.execute()
        self.make_capacity()
        path = self.capacity / "capacity-input.json"
        value = long_run.read_json(path)
        value["reserved_tokens"] -= 1
        write(path, value)
        with self.assertRaisesRegex(ValueError, "context reserve"):
            self.execute()

    def test_capacity_requires_actual_30_final_tokens(self):
        path = self.capacity / "finalization.json"
        value = long_run.read_json(path)
        value["samples"].pop()
        write(path, value)
        with self.assertRaisesRegex(ValueError, "30-token"):
            self.execute()

    def test_capacity_environment_and_natural_environment_bind_to_gate(self):
        self.natural_records["eos"]["backend_metadata"] = {**self.metadata, "attention_implementation": "sdpa"}
        self.save_natural()
        with self.assertRaisesRegex(ValueError, "incompatible"):
            self.execute()

    def test_success_keeps_fixed_seed_separate_warmup_progress_and_grading_input(self):
        code, backend, factory, _ = self.execute()
        self.assertEqual(code, 0)
        self.assertEqual(factory.call_count, 1)
        self.assertEqual(backend.requests, 2)
        self.assertEqual(backend.seeds[1], 1937708343865469216)
        self.assertNotEqual(backend.seeds[0], backend.seeds[1])
        directory = self.root / "new-run"
        result = long_run.read_json(directory / "vanilla.json")
        self.assertEqual((result["method"], result["max_new_tokens"], result["n_probes"]), ("vanilla", 32768, 0))
        self.assertEqual(long_run.read_json(directory / "warmup.json")["max_new_tokens"], 16)
        summary = long_run.read_json(directory / "summary.json")
        self.assertEqual((summary["status"], summary["executed_count"], summary["unexecuted_count"]), ("completed", 1, 0))
        answers = [json.loads(line) for line in (directory / "final_answers.jsonl").read_text().splitlines()]
        self.assertEqual(len(answers), 1)
        self.assertEqual(answers[0]["source_sha256"], long_run.sha256(directory / "vanilla.json"))
        self.assertEqual(summary["grading_status"], "not_run")
        self.assertEqual(long_run.read_json(directory / "integrity-before.json"), long_run.read_json(directory / "integrity-after.json"))
        self.assertIn('"main_tokens": 64', (directory / "events.jsonl").read_text())

    def test_actual_backend_mismatch_fails_before_any_prefill(self):
        backend = CPUBackend({**self.metadata, "attention_implementation": "sdpa"})
        code, backend, factory, _ = self.execute(backend)
        self.assertEqual(code, 1)
        self.assertEqual(backend.requests, 0)
        self.assertEqual(factory.call_count, 1)
        summary = long_run.read_json(self.root / "new-run/summary.json")
        self.assertEqual(summary["executed_count"], 0)
        self.assertEqual(summary["failure"]["phase"], "model_load")

    def test_actual_prompt_mismatch_fails_before_warmup(self):
        backend = CPUBackend(self.metadata)
        backend.prompt_ids = lambda _: [1, 3]
        code, backend, _, _ = self.execute(backend)
        self.assertEqual(code, 1)
        self.assertEqual(backend.requests, 0)
        self.assertIn("prompt differs", long_run.read_json(self.root / "new-run/failure.json")["message"])

    def test_OOM_retains_partial_request_and_denominator_without_retry(self):
        code, backend, factory, _ = self.execute(CPUBackend(self.metadata, failure=RuntimeError("synthetic CUDA out of memory")))
        self.assertEqual(code, 1)
        self.assertEqual((factory.call_count, backend.requests), (1, 2))
        directory = self.root / "new-run"
        partial = long_run.read_json(directory / "vanilla.json")
        self.assertEqual(partial["status"], "failed")
        self.assertEqual(len(partial["main_samples"]), 2)
        summary = long_run.read_json(directory / "summary.json")
        self.assertEqual((summary["executed_count"], summary["unexecuted_count"]), (1, 0))
        self.assertFalse(long_run.read_json(directory / "failure.json")["retry_performed"])
        self.assertEqual(len((directory / "final_answers.jsonl").read_text().splitlines()), 1)

    def test_keyboard_interrupt_saves_honest_minimal_failure_row(self):
        code, _, _, _ = self.execute(CPUBackend(self.metadata, failure=KeyboardInterrupt()))
        self.assertEqual(code, 1)
        directory = self.root / "new-run"
        partial = long_run.read_json(directory / "vanilla.json")
        self.assertFalse(partial["partial_evidence_available"])
        self.assertEqual(partial["error_type"], "KeyboardInterrupt")
        self.assertEqual(long_run.read_json(directory / "summary.json")["executed_count"], 1)

    def test_export_failure_preserves_completed_source_and_failed_answer_row(self):
        original = long_run.answer_record
        calls = 0
        def export(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ValueError("synthetic export failure")
            return original(*args, **kwargs)
        with patch.object(long_run, "answer_record", side_effect=export):
            code, _, _, _ = self.execute()
        self.assertEqual(code, 1)
        directory = self.root / "new-run"
        self.assertEqual(long_run.read_json(directory / "vanilla.json")["status"], "completed")
        answer = long_run.read_json(directory / "vanilla.final-answer.json")
        self.assertEqual((answer["source_execution_status"], answer["execution_status"]), ("completed", "failed"))

    def test_input_mutation_during_main_request_invalidates_run(self):
        path = self.natural / "third-request-plan.json"
        code, _, _, _ = self.execute(CPUBackend(self.metadata, mutation=lambda: write(path, {"changed": True})))
        self.assertEqual(code, 1)
        directory = self.root / "new-run"
        summary = long_run.read_json(directory / "summary.json")
        self.assertEqual(summary["source_input_identity_integrity"], "changed")
        self.assertEqual(summary["executed_count"], 1)
        self.assertTrue((directory / "integrity-failure.json").exists())

    def test_existing_or_nested_output_never_overwrites_evidence(self):
        self.assertEqual(self.execute()[0], 0)
        old = (self.root / "new-run/manifest.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "new and outside"):
            self.execute()
        self.assertEqual((self.root / "new-run/manifest.json").read_bytes(), old)
        with self.assertRaisesRegex(ValueError, "new and outside"):
            self.execute(run_root=self.capacity / "nested")


if __name__ == "__main__":
    unittest.main()
