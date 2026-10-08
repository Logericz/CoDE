"""CPU checks for bounded GPU acceptance orchestration, never GPU evidence."""
from contextlib import nullcontext, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("validate_online_gpu", ROOT / "scripts" / "validate_online_gpu.py")
gpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu)

try:
    import torch
    import transformers
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from test_torch_online_backend import SyntheticTokenizer
    from torch_online_backend import TorchOnlineBackend
    RUNTIME_AVAILABLE = (torch.__version__.split("+")[0] == "2.9.1"
                         and transformers.__version__ == "4.51.3")
except ImportError:
    RUNTIME_AVAILABLE = False


class AcceptanceOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.evidence = self.root / "evidence"
        self.saved = self.evidence / "items" / "q001"
        self.saved.mkdir(parents=True)
        common = {"status": "completed", "metrics": {"model_revision": gpu.MODEL_REVISION}}
        self.base = {**common, "generation_calls": [{"input_token_ids": [[1, 2, 3]]}]}
        self.stage = {**common,
            "method_output": {"prob_checks": [{"stop_token_idx": 2}, {"stop_token_idx": 4}],
                              "sample_response": "saved reasoning"},
            "generation_calls": [{"status": "ok", "input_token_ids": [[1, 2, 3, 10, 11, 12, 13, 98, 99]]}],
            "metadata": {"attention_implementation": "sdpa"}}
        self.save()

    def save(self):
        (self.saved / "base.json").write_text(json.dumps(self.base))
        (self.saved / "codestop.json").write_text(json.dumps(self.stage))

    def test_reconstructs_only_saved_prefixes_and_records_hashes(self):
        before = {path: path.read_bytes() for path in self.saved.iterdir()}
        cases = gpu.load_saved_prefixes(self.evidence, ["q001:0", "q001:1"])
        self.assertEqual(cases[0]["prefix_ids"], [1, 2, 3, 10, 11])
        self.assertEqual(cases[1]["prefix_ids"], [1, 2, 3, 10, 11, 12, 13])
        self.assertEqual(cases[0]["saved_final_prefix_ids"], [98, 99])
        self.assertEqual(len(cases[0]["input_sha256"]), 2)
        self.assertEqual({path: path.read_bytes() for path in self.saved.iterdir()}, before)

    def test_rejects_unbounded_duplicate_and_unrecognized_selectors(self):
        for selectors in ([], ["q001:0"] * 2, ["q003:0"], ["q001:-1"],
                          ["q001:0", "q001:1", "q002:0", "q002:1"]):
            with self.subTest(selectors=selectors), self.assertRaises(ValueError):
                gpu.parse_prefixes(selectors)

    def test_missing_checkpoint_is_not_generated(self):
        with self.assertRaisesRegex(ValueError, "No saved probe"):
            gpu.load_saved_prefixes(self.evidence, ["q001:2"])

    def test_refuses_changed_revision_incomplete_record_and_changed_prompt(self):
        for mutate, message in (
                (lambda: self.stage.update(status="failed"), "did not complete"),
                (lambda: self.stage["metrics"].update(model_revision="wrong"), "different model revision"),
                (lambda: self.stage["generation_calls"][0].update(input_token_ids=[[5, 2, 3, 10, 98]]),
                 "does not preserve")):
            self.stage = json.loads((self.saved / "codestop.json").read_text())
            original = json.loads(json.dumps(self.stage))
            mutate()
            self.save()
            with self.assertRaisesRegex(ValueError, message):
                gpu.load_saved_prefixes(self.evidence, ["q001:0"])
            self.stage = original
            self.save()

    def test_rejects_nonmonotonic_and_boolean_positions(self):
        for positions in ((4, 2), (2, 2), (True, 2)):
            self.stage["method_output"]["prob_checks"] = [{"stop_token_idx": x} for x in positions]
            self.save()
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                gpu.load_saved_prefixes(self.evidence, ["q001:0"])

    def test_tokenizer_must_reproduce_saved_reasoning_final_prefix_and_candidate(self):
        from types import SimpleNamespace
        backend = SimpleNamespace(tokenizer=lambda _: {"input_ids": [10, 11, 12, 13, 3]},
                                  markers=SimpleNamespace(final_prefix=(98, 99), wait=12, eos=3),
                                  _tokens=lambda values: values)
        case = gpu.load_saved_prefixes(self.evidence, ["q001:0"])[0]
        gpu.validate_tokenizer_prefix(case, backend)
        backend.tokenizer = lambda _: {"input_ids": [10, 20, 12, 13, 3]}
        with self.assertRaisesRegex(ValueError, "reasoning prefix"):
            gpu.validate_tokenizer_prefix(case, backend)

    def test_fresh_directory_refuses_existing_even_empty_and_keeps_lock(self):
        run = self.root / "run"
        with gpu.fresh_run(run):
            self.assertTrue((run / ".run.lock").is_file())
            with self.assertRaises(FileExistsError):
                with gpu.fresh_run(run):
                    self.fail("existing directory was accepted")
        with self.assertRaises(FileExistsError):
            with gpu.fresh_run(run):
                self.fail("finished directory was accepted")

    def test_exact_comparison_detects_first_probability_without_tolerance(self):
        expected = {"token_ids": [5, 6], "token_probs": [1.0, 0.5], "ended": False}
        actual = {"token_ids": [5, 6], "token_probs": [1.0, 0.500000001], "ended": True}
        difference = gpu.first_difference(expected, actual)
        self.assertEqual(difference["field"], "token_probs[1]")
        self.assertEqual(gpu.TOLERANCE["atol"], 0)
        self.assertEqual(gpu.TOLERANCE["rtol"], 0)

    def test_matching_nonfinite_raw_values_are_preserved_as_diagnostics(self):
        self.assertIsNone(gpu.first_difference([float("nan")], [float("nan")]))
        path = self.root / "nan.json"
        gpu.write_json(path, {"confidence": float("nan")})
        self.assertEqual(json.loads(path.read_text()), {"confidence": {"nonfinite": "nan"}})
        with self.assertRaises(FileExistsError):
            gpu.write_json(path, {})

    def test_first_difference_is_saved_before_failure_and_not_overwritten(self):
        with self.assertRaises(gpu.AcceptanceFailure):
            gpu.fail_on_difference(self.root, {"token": 1}, {"token": 2}, phase="test", case="q001")
        report = json.loads((self.root / "first-difference.json").read_text())
        self.assertEqual(report["expected"], 1)
        self.assertEqual(report["actual"], 2)
        self.assertEqual(report["phase"], "test")

    def test_reference_export_is_verified_against_actual_pinned_git_object(self):
        raw, function, metadata = gpu.reference_source()
        export = self.root / "upstream"
        export.mkdir()
        (export / gpu.REFERENCE_FILE).write_bytes(raw)
        second, _, exported = gpu.reference_source(export)
        self.assertEqual(second, raw)
        self.assertEqual(function.name, "calcu_max_probs_w_kv")
        self.assertEqual(metadata["sha256"], gpu.REFERENCE_SHA256)
        self.assertEqual(exported["source_kind"], "sha256_verified_git_object_export")
        (export / gpu.REFERENCE_FILE).write_bytes(raw + b"\n")
        with self.assertRaisesRegex(ValueError, "differs from the frozen"):
            gpu.reference_source(export)

    def test_inspect_only_needs_no_model_runtime_and_writes_nothing(self):
        before = set(self.root.rglob("*"))
        output = io.StringIO()
        with redirect_stdout(output):
            code = gpu.main(["--evidence-root", str(self.evidence), "--prefix", "q001:0", "--inspect-only"])
        self.assertEqual(code, 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["prefixes"][0]["prefix_tokens"], 5)
        self.assertEqual(report["probe_max_new_tokens"], 21)
        self.assertEqual(set(self.root.rglob("*")), before)

    def test_limits_fail_before_model_import_or_directory_creation(self):
        for extra in (("--continuation-tokens", "65"), ("--continuation-tokens", "0"), ("--seed", "-1")):
            with self.assertRaisesRegex(ValueError, "Require continuation"):
                gpu.main(["--evidence-root", str(self.evidence), *extra])

    def test_cli_requires_explicit_model_and_new_run_for_execution(self):
        with self.assertRaisesRegex(ValueError, "requires --model-dir"):
            gpu.main(["--evidence-root", str(self.evidence), "--prefix", "q001:0"])

    def test_model_load_failure_is_durable_without_claiming_acceptance(self):
        import torch_online_backend
        run = self.root / "failed-run"
        output = io.StringIO()
        with patch.object(torch_online_backend, "TorchOnlineBackend", side_effect=RuntimeError("no CUDA")), \
                patch.object(gpu, "gpu_inventory", return_value={"uuid": "GPU-test"}), \
                patch.object(gpu, "gpu_lock", return_value=nullcontext("test.lock")), \
                patch.object(gpu, "assert_gpu_idle", return_value={"compute_processes": []}), \
                redirect_stdout(output):
            code = gpu.main(["--evidence-root", str(self.evidence), "--prefix", "q001:0",
                             "--model-dir", str(self.root / "model"), "--run-root", str(run)])
        self.assertEqual(code, 1)
        self.assertEqual(json.loads((run / "failure.json").read_text())["message"], "no CUDA")
        self.assertTrue((run / "plan.json").exists())
        self.assertTrue((run / "inputs.json").exists())
        self.assertFalse((run / "summary.json").exists())

    def test_busy_gpu_is_recorded_and_model_never_loaded(self):
        import torch_online_backend
        run = self.root / "busy-run"
        with patch.object(torch_online_backend, "TorchOnlineBackend") as loader, \
                patch.object(gpu, "gpu_inventory", return_value={"uuid": "GPU-test"}), \
                patch.object(gpu, "gpu_lock", return_value=nullcontext("test.lock")), \
                patch.object(gpu, "assert_gpu_idle", side_effect=RuntimeError("GPU busy")), \
                redirect_stdout(io.StringIO()):
            code = gpu.main(["--evidence-root", str(self.evidence), "--prefix", "q001:0",
                             "--model-dir", str(self.root / "model"), "--run-root", str(run)])
        loader.assert_not_called()
        self.assertEqual(code, 1)
        self.assertEqual(json.loads((run / "failure.json").read_text())["message"], "GPU busy")

    def test_run_root_cannot_write_inside_saved_evidence(self):
        run = self.evidence / "new-run"
        with self.assertRaisesRegex(ValueError, "outside read-only"):
            gpu.main(["--evidence-root", str(self.evidence), "--prefix", "q001:0",
                      "--model-dir", str(self.root / "model"), "--run-root", str(run)])
        self.assertFalse(run.exists())


@unittest.skipUnless(RUNTIME_AVAILABLE, "requires pinned tiny-model CPU environment")
class TinyBF16AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(123)
            config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                head_dim=8, max_position_embeddings=256, bos_token_id=1,
                eos_token_id=2, pad_token_id=0, attention_dropout=0.0,
                tie_word_embeddings=False)
            config._attn_implementation = "eager"
            model = Qwen3ForCausalLM(config).to(dtype=torch.bfloat16).eval()
        self.backend = TorchOnlineBackend.from_test_model(model, SyntheticTokenizer())
        self.case = {"name": "synthetic-cpu", "prompt_ids": [1, 5, 6, 7, 8],
                     "prefix_ids": [1, 5, 6, 7, 8, 15, 16, 17, 18], "saved_observation": {}}

    def test_teacher_forced_prefix_uses_one_token_extends_without_sampling(self):
        self.backend.start_request(171)
        with patch.object(self.backend, "sample", side_effect=AssertionError("must not sample")), \
                patch.object(self.backend, "extend", wraps=self.backend.extend) as extend:
            state = gpu.rebuild_online_prefix(self.backend, self.case, self.root)
        self.assertEqual([call.args[1] for call in extend.call_args_list], [(15,), (16,), (17,), (18,)])
        self.assertEqual(state.length, 9)

    def test_actual_bf16_incremental_probe_matches_pinned_reference_exactly(self):
        _, function, _ = gpu.reference_source()
        reference = gpu.load_reference(function, torch)
        with redirect_stdout(io.StringIO()), patch.object(torch.cuda, "empty_cache"):
            report = gpu.validate_probe(self.backend, reference, self.case, self.root, 171)
        self.assertTrue(report["exact"])
        actual = json.loads((self.root / "synthetic-cpu-online-probe.json").read_text())
        expected = json.loads((self.root / "synthetic-cpu-reference-probe.json").read_text())
        self.assertEqual(actual["probability_bf16_bits"], expected["probability_bf16_bits"])
        self.assertLessEqual(report["probe_tokens"], 21)

    def test_actual_cache_logits_rng_and_0_1_2_probe_continuation_are_exact(self):
        with redirect_stdout(io.StringIO()):
            results = gpu.validate_isolation(self.backend, self.case, self.root, 872, 16)
        self.assertEqual([row["inserted_probes"] for row in results], [0, 1, 2])
        self.assertTrue(all(row["exact"] for row in results))
        self.assertFalse((self.root / "first-difference.json").exists())

    def test_private_rng_corruption_is_caught_before_reference_execution(self):
        original = self.backend.probe
        def corrupted(state):
            result = original(state)
            self.backend._generator.manual_seed(999)
            return result
        with patch.object(self.backend, "probe", side_effect=corrupted), \
                redirect_stdout(io.StringIO()), self.assertRaises(gpu.AcceptanceFailure):
            gpu.validate_probe(self.backend, None, self.case, self.root, 171)
        failure = json.loads((self.root / "first-difference.json").read_text())
        self.assertEqual(failure["phase"], "probe_state_isolation")
        self.assertEqual(failure["field"], "private_rng.sha256")
        self.assertTrue((self.root / "synthetic-cpu-online-probe.json").exists())

    def test_terminal_eos_is_recorded_without_extending_main_kv(self):
        from online_contract import TokenSample
        with patch.object(self.backend, "sample", return_value=TokenSample(2, 1.0, 1.0)), \
                redirect_stdout(io.StringIO()):
            results = gpu.validate_isolation(self.backend, self.case, self.root, 872, 16)
        self.assertTrue(all(row["continuation_tokens"] == 1 for row in results))
        for probes in (0, 1, 2):
            result = json.loads((self.root / f"isolation-{probes}.json").read_text())
            self.assertEqual(result["final_state"]["length"], len(self.case["prefix_ids"]))
            self.assertEqual(result["continuation"][0]["token_id"], 2)


if __name__ == "__main__":
    unittest.main()
