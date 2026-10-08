"""CPU orchestration fixtures for a gated capacity tool; no GPU capacity claim."""
from contextlib import ExitStack, nullcontext, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("capacity_tool", ROOT / "scripts/validate_online_capacity.py")
capacity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capacity)
from online_contract import TokenMarkers, TokenSample
from online_protocol import ProbeObservation


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class FakeCache:
    def __init__(self, length):
        self.length = length

    def get_seq_length(self):
        return self.length


class FakeBackend:
    markers = TokenMarkers(21, 22, 23, (9, 10), (22, 9, 10), eos_ids=(23, 24))
    context_limit = 40000
    metadata = {**{field: None for field in capacity.ENVIRONMENT_FIELDS},
                "model_parameter_dtype": "torch.bfloat16", "validation_model": True,
                "scope": "synthetic_cpu_capacity_test"}

    def __init__(self, *args, **kwargs):
        self.started = False
        self.extensions = []
        self.probes = 0
        self.greedy_calls = 0
        self.corrupt_probe = False
        self.fail_extend = False
        self.main_seed = None

    def start_request(self, seed):
        self.started, self.main_seed = True, seed

    def prefill(self, ids):
        return SimpleNamespace(length=len(ids), cache=FakeCache(len(ids)))

    def extend(self, state, ids):
        if self.fail_extend:
            raise RuntimeError("synthetic out of memory")
        self.extensions.append(list(ids))
        state.cache.length += len(ids)
        return SimpleNamespace(length=state.cache.length, cache=state.cache)

    def probe(self, state):
        self.probes += 1
        if self.corrupt_probe:
            state.cache.length += 1
        return ProbeObservation((9, 10, 22), (0.5, 0.5, 0.5), 0.5, True,
                                confidence_source="synthetic_cpu_test")

    def greedy(self, state):
        self.greedy_calls += 1
        return TokenSample(15 if self.greedy_calls == 1 else 24, 1.0, 1.0)

    def synchronize(self):
        pass


class CapacityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.source = self.root / "source/vanilla.json"
        pattern = [7, 8] * 512
        record = {"sample_id": capacity.SAMPLE_ID, "status": "completed", "method": "vanilla",
                  "max_new_tokens": 1024, "backend_metadata": {"model_revision": capacity.MODEL_REVISION},
                  "prompt_token_ids": [1, 2], "main_samples": [{"token_id": token, "phase": "reason"} for token in pattern],
                  "output_token_ids": pattern + [22, 9, 10, 15]}
        write(self.source, record)
        self.source_hash = capacity.sha256(self.source)
        self.gate = self.root / "gate/summary.json"
        self.gate_record = {"schema_version": 1, "gate_passed": True,
            "source_input_identity_integrity": "unchanged",
            "status": "passed_same_KV_contract_bounded_validation", "reference_contract": capacity.REFERENCE_CONTRACT,
            "completed_prefixes": [{"case": name, "same_KV_gate": {"status": "passed"}} for name in capacity.GATE_CASES],
            "isolation": {"case": "q002-probe-0", "status": "passed", "variants": [
                {"inserted_probes": n, "exact": True, "continuation_tokens": 64} for n in (0, 1, 2)]}}
        write(self.gate, self.gate_record)
        self.hashes = {name: capacity.sha256(ROOT / name) for name in (*capacity.CRITICAL_SOURCES, "src/online_engine.py")}
        write(self.gate.parent / "source-hashes.json", self.hashes)
        write(self.gate.parent / "environment.json", FakeBackend.metadata)
        write(self.gate.parent / "integrity-before.json", self.hashes)
        write(self.gate.parent / "integrity-after.json", self.hashes)
        self.natural = self.root / "natural"
        write(self.natural / "manifest.json", {"code_sha256": self.hashes})
        self.natural_records = {}
        statuses = {"early": "confidence", "eos": "natural_eos", "answer-budget": "budget"}
        for label, stop in statuses.items():
            item = {"status": "completed", "configuration": label,
                "sample_id": capacity.SAMPLE_ID,
                "scope": "single_question_development_gpu_acceptance_not_benchmark",
                "backend_metadata": {"validation_model": False, "model_revision": capacity.MODEL_REVISION,
                    "attention_implementation": "eager", "model_parameter_dtype": "torch.bfloat16",
                    "markers": {"eos_ids": [23, 24]}},
                "stop_reason": stop, "probes": [{"stop_applied": True}] if label == "early" else [],
                "output_token_ids": [7, 24] if label == "eos" else [7, 22, 15],
                "finalization_tokens": 0 if label == "eos" else 2,
                "injected_prompt_tokens": 0 if label == "answer-budget" else 3,
                "answer_boundary": {"source": "natural_first_end_think" if label == "answer-budget" else "controlled_final_prefix"}}
            self.natural_records[label] = item
        self.save_natural()

    def save_natural(self):
        rows = []
        for label, record in self.natural_records.items():
            path = self.natural / f"{label}.json"
            write(path, record)
            rows.append({"configuration": label, "status": "completed", "source_sha256": capacity.sha256(path)})
        write(self.natural / "summary.json", {"status": "completed", "executed_count": len(rows),
            "unexecuted_count": 0, "code_unchanged_during_run": True, "requests": rows})

    def args(self, **updates):
        args = SimpleNamespace(source_record=self.source, gate_summary=self.gate,
            natural_run_root=[self.natural], independent_capacity_diagnostic=False,
            model_dir=self.root / "model", run_root=self.root / "run", chunk_size=128,
            wall_time_seconds=900, seed=872, inspect_only=False)
        for name, value in updates.items():
            setattr(args, name, value)
        return args

    def execute(self, backend=None, **updates):
        backend = backend or FakeBackend()
        original_loader = capacity.load_source
        def fingerprint(instance, state):
            return {"length": state.length, "cache_length": state.cache.length, "rng_seed": instance.main_seed}
        def memory(instance, state):
            return {"allocated_bytes": 1000 + state.cache.length, "reserved_bytes": 2000 + state.cache.length,
                    "peak_allocated_bytes": 1500 + state.cache.length,
                    "peak_reserved_bytes": 2500 + state.cache.length,
                    "kv_length": state.cache.length, "committed_state_length": state.length}
        with ExitStack() as stack:
            stack.enter_context(patch.object(capacity, "load_source", side_effect=lambda p: original_loader(p, expected_sha256=self.source_hash)))
            stack.enter_context(patch.object(capacity, "gpu_inventory", return_value={"uuid": "GPU-test"}))
            stack.enter_context(patch.object(capacity, "gpu_lock", side_effect=lambda _: nullcontext("test-lock")))
            stack.enter_context(patch.object(capacity, "assert_gpu_idle", return_value={"compute_processes": []}))
            stack.enter_context(patch.object(capacity, "state_fingerprint", side_effect=fingerprint))
            stack.enter_context(patch.object(capacity, "memory_snapshot", side_effect=memory))
            stack.enter_context(redirect_stdout(io.StringIO()))
            result = capacity.run_capacity(self.args(**updates), backend_factory=lambda *a, **k: backend)
        return result, backend

    def test_frozen_source_identity_and_pattern(self):
        source = capacity.load_source(self.source, expected_sha256=self.source_hash)
        self.assertEqual(len(source["pattern_ids"]), 1024)
        self.assertEqual(source["prompt_ids"], [1, 2])
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            capacity.load_source(self.source)
        record = capacity.read_json(self.source)
        record["main_samples"][0]["phase"] = "answer"
        write(self.source, record)
        with self.assertRaisesRegex(ValueError, "thinking tokens"):
            capacity.load_source(self.source, expected_sha256=capacity.sha256(self.source))

    def test_gate_requires_five_cases_isolation_and_frozen_sources(self):
        self.assertTrue(capacity.validate_gate(self.gate)["gate_passed"])
        self.gate_record["completed_prefixes"].pop()
        write(self.gate, self.gate_record)
        with self.assertRaisesRegex(ValueError, "five"):
            capacity.validate_gate(self.gate)
        self.gate_record["completed_prefixes"].append({"case": "q002-probe-2", "same_KV_gate": {"status": "passed"}})
        write(self.gate, self.gate_record)
        write(self.gate.parent / "source-hashes.json", {**self.hashes, "src/online_contract.py": "0" * 64})
        with self.assertRaisesRegex(ValueError, "differs"):
            capacity.validate_gate(self.gate)

    def test_shorter_identical_bounded_isolation_is_reported_not_claimed_as_64(self):
        for item in self.gate_record["isolation"]["variants"]:
            item["continuation_tokens"] = 9
        write(self.gate, self.gate_record)
        self.assertEqual(capacity.validate_gate(self.gate)["isolation_continuation_tokens"], 9)
        self.gate_record["isolation"]["variants"][2]["continuation_tokens"] = 8
        write(self.gate, self.gate_record)
        with self.assertRaisesRegex(ValueError, "isolation"):
            capacity.validate_gate(self.gate)

    def test_gate_requires_unchanged_integrity_engine_and_consistent_evidence(self):
        self.gate_record.pop("source_input_identity_integrity")
        write(self.gate, self.gate_record)
        with self.assertRaisesRegex(ValueError, "passed continuation gate"):
            capacity.validate_gate(self.gate)
        self.gate_record["source_input_identity_integrity"] = "unchanged"
        write(self.gate, self.gate_record)
        write(self.gate.parent / "source-hashes.json", {**self.hashes, "src/online_engine.py": "0" * 64})
        with self.assertRaisesRegex(ValueError, "online_engine"):
            capacity.validate_gate(self.gate)
        write(self.gate.parent / "source-hashes.json", self.hashes)
        write(self.gate.parent / "integrity-after.json", {"different": "content"})
        with self.assertRaisesRegex(ValueError, "integrity records differ"):
            capacity.validate_gate(self.gate)

    def test_natural_coverage_and_explicit_independent_mode(self):
        coverage = capacity.validate_natural_runs([self.natural])
        self.assertTrue(coverage["gate_passed"])
        self.assertEqual(set(coverage["covered"]), set(capacity.BRANCHES))
        del self.natural_records["answer-budget"]
        self.save_natural()
        with self.assertRaisesRegex(ValueError, "actual_answer_phase_budget"):
            capacity.validate_natural_runs([self.natural])
        coverage = capacity.validate_natural_runs([self.natural], independent=True)
        self.assertEqual(coverage["missing"], ["actual_answer_phase_budget"])
        self.assertFalse(coverage["gate_passed"])

    def test_forced_or_tampered_requests_cannot_establish_natural_coverage(self):
        self.natural_records["early"]["validation_only"] = True
        self.save_natural()
        with self.assertRaisesRegex(ValueError, "synthetic"):
            capacity.validate_natural_runs([self.natural], independent=True)
        self.natural_records["early"]["validation_only"] = False
        self.save_natural()
        write(self.natural / "early.json", {**self.natural_records["early"], "stop_reason": "budget"})
        with self.assertRaisesRegex(ValueError, "hash"):
            capacity.validate_natural_runs([self.natural])

    def test_real_model_synthetic_question_branches_keep_separate_origin_and_cannot_supply_early_stop(self):
        for label in ("eos", "answer-budget"):
            self.natural_records[label].update(scope=capacity.NATURAL_SYNTHETIC_QUESTION_SCOPE,
                forced_tokens=False, synthetic_question=True, sample_id="synthetic-one-plus-one")
        self.save_natural()
        coverage = capacity.validate_natural_runs([self.natural])
        self.assertTrue(coverage["gate_passed"])
        self.assertEqual(coverage["coverage_by_origin"]["frozen_dataset"], ["actual_early_stop"])
        self.assertEqual(coverage["coverage_by_origin"]["synthetic_question_real_model"],
                         ["actual_answer_phase_budget", "actual_natural_eos"])
        self.natural_records["early"].update(scope=capacity.NATURAL_SYNTHETIC_QUESTION_SCOPE,
            forced_tokens=False, synthetic_question=True, sample_id="synthetic-one-plus-one")
        self.save_natural()
        coverage = capacity.validate_natural_runs([self.natural], independent=True)
        self.assertEqual(coverage["missing"], ["actual_early_stop"])
        self.natural_records["early"].update(scope=capacity.DATASET_SCOPE, sample_id="different-dataset-question")
        self.save_natural()
        self.assertEqual(capacity.validate_natural_runs([self.natural], independent=True)["missing"],
                         ["actual_early_stop"])

    def test_synthetic_question_scope_requires_real_model_and_explicit_unforced_tokens(self):
        record = self.natural_records["eos"]
        record.update(scope=capacity.NATURAL_SYNTHETIC_QUESTION_SCOPE,
                      synthetic_question=True, sample_id="synthetic-one-plus-one")
        for forced, synthetic_model in ((None, False), (True, False), (False, True)):
            with self.subTest(forced=forced, synthetic_model=synthetic_model):
                if forced is None:
                    record.pop("forced_tokens", None)
                else:
                    record["forced_tokens"] = forced
                record["backend_metadata"]["validation_model"] = synthetic_model
                self.save_natural()
                with self.assertRaisesRegex(ValueError, "synthetic/incompatible"):
                    capacity.validate_natural_runs([self.natural], independent=True)

    def test_inspect_only_does_not_touch_gpu_or_create_outputs(self):
        original = capacity.load_source
        with patch.object(capacity, "load_source", side_effect=lambda p: original(p, expected_sha256=self.source_hash)), \
                patch.object(capacity, "gpu_inventory") as gpu, redirect_stdout(io.StringIO()):
            self.assertEqual(capacity.run_capacity(self.args(inspect_only=True)), 0)
        gpu.assert_not_called()
        self.assertFalse((self.root / "run").exists())

    def test_bounded_32k_chunks_probe_isolation_finalization_and_memory_evidence(self):
        result, backend = self.execute()
        self.assertEqual(result, 0)
        summary = capacity.read_json(self.root / "run/summary.json")
        self.assertEqual(summary["filled_main_tokens"], 32768)
        self.assertEqual(summary["finalization_tokens"], 2)
        self.assertEqual(summary["finalization_end"], "eos")
        self.assertEqual(summary["final_KV_length"], 2 + 32768 + 3 + 1)
        self.assertEqual(backend.probes, 1)
        self.assertEqual(len(backend.extensions[:256]), 256)
        self.assertTrue(all(len(tokens) == 128 for tokens in backend.extensions[:256]))
        self.assertNotIn([24], backend.extensions)
        self.assertFalse(summary["natural_32k_generation_validated"])
        self.assertFalse(summary["eligible_for_accuracy_or_speed_comparison"])
        self.assertEqual(len(summary["memory_progress"]), 36)
        self.assertFalse((self.root / "run/failure.json").exists())

    def test_context_reserve_refused_before_prefill(self):
        backend = FakeBackend()
        backend.context_limit = 2 + 32768 + 32
        result, backend = self.execute(backend)
        self.assertEqual(result, 1)
        self.assertFalse(backend.started)
        self.assertIn("reserve insufficient", capacity.read_json(self.root / "run/failure.json")["message"])

    def test_environment_mismatch_fails_before_prefill(self):
        backend = FakeBackend()
        backend.metadata = {**backend.metadata, "attention_implementation": "sdpa"}
        result, backend = self.execute(backend)
        self.assertEqual(result, 1)
        self.assertFalse(backend.started)
        self.assertIn("attention_implementation", capacity.read_json(self.root / "run/failure.json")["message"])

    def test_each_changed_input_blocks_success_after_execution(self):
        for index, path in enumerate((self.source, self.gate, self.natural / "manifest.json",
                                      self.natural / "summary.json", self.natural / "early.json")):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                backend = FakeBackend()
                original_greedy = backend.greedy
                def mutate(state):
                    sample = original_greedy(state)
                    if backend.greedy_calls == 1:
                        path.write_bytes(original + b" ")
                    return sample
                backend.greedy = mutate
                run_root = self.root / f"changed-{index}"
                try:
                    result, _ = self.execute(backend, run_root=run_root)
                    self.assertEqual(result, 1)
                    self.assertFalse((run_root / "summary.json").exists())
                    self.assertEqual(capacity.read_json(run_root / "failure.json")["phase"], "source_input_identity_integrity")
                    self.assertTrue((run_root / "integrity-after.json").is_file())
                finally:
                    path.write_bytes(original)

    def test_keyboard_interrupt_preserves_failure_phase(self):
        backend = FakeBackend()
        backend.extend = lambda *args: (_ for _ in ()).throw(KeyboardInterrupt())
        result, _ = self.execute(backend)
        self.assertEqual(result, 1)
        failure = capacity.read_json(self.root / "run/failure.json")
        self.assertEqual(failure["phase"], "synthetic_chunk_fill")
        self.assertEqual(failure["error_type"], "KeyboardInterrupt")

    def test_one_failure_is_saved_without_retry(self):
        backend = FakeBackend()
        backend.fail_extend = True
        result, backend = self.execute(backend)
        self.assertEqual(result, 1)
        failure = capacity.read_json(self.root / "run/failure.json")
        self.assertEqual(failure["phase"], "synthetic_chunk_fill")
        self.assertEqual(failure["filled_main_tokens"], 0)
        self.assertFalse(failure["retry_performed"])
        self.assertEqual(backend.probes, 0)
        self.assertFalse((self.root / "run/summary.json").exists())

    def test_probe_state_corruption_blocks_finalization(self):
        backend = FakeBackend()
        backend.corrupt_probe = True
        result, backend = self.execute(backend)
        self.assertEqual(result, 1)
        self.assertEqual(backend.greedy_calls, 0)
        self.assertTrue((self.root / "run/first-difference.json").is_file())

    def test_existing_output_and_unsafe_output_are_refused(self):
        (self.root / "run").mkdir()
        with self.assertRaisesRegex(ValueError, "new and outside"):
            self.execute()
        with self.assertRaisesRegex(ValueError, "new and outside"):
            self.execute(run_root=self.source.parent / "new-run")


if __name__ == "__main__":
    unittest.main()
