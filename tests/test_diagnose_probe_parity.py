"""Evidence-link and tiny-BF16 CPU tests; no pretrained/GPU claims."""
from contextlib import nullcontext, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_validate_online_gpu as base_tests

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("diagnose_probe_parity", ROOT / "scripts" / "diagnose_probe_parity.py")
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)
validation = diagnostic.validation


class DiagnosticEvidenceTests(unittest.TestCase):
    def setUp(self):
        fixture = base_tests.AcceptanceOrchestrationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root, self.evidence = fixture.root, fixture.evidence
        self.case = validation.load_saved_prefixes(self.evidence, ["q001:0"])[0]
        self.failure = self.root / "previous-failure"
        self.failure.mkdir()
        original = {"token_ids": [10, 11], "token_probs": [1.0, 0.25], "confidence": 0.875, "ended": False}
        actual = {**original, "token_probs": [1.0, 0.5]}
        self.records = {
            "plan.json": {"scope": "bounded_online_gpu_acceptance", "tolerance": validation.TOLERANCE,
                          "reference": {"sha256": validation.REFERENCE_SHA256,
                                        "upstream_commit": validation.UPSTREAM_COMMIT},
                          "seed": 872, "attention_implementation": "eager"},
            "inputs.json": [self.case], "environment.json": {},
            "source-hashes.json": {name: validation.sha((ROOT / name).read_bytes())
                                   for name in diagnostic.CRITICAL_SOURCE_FILES},
            "first-difference.json": {"phase": "full_prefix_probe_numerics", "case": self.case["name"],
                                      **validation.first_difference(original, actual)},
            "q001-probe-0-reference-probe.json": original,
            "q001-probe-0-online-probe.json": actual,
        }
        for filename, value in self.records.items():
            validation.write_json(self.failure / filename, value)

    def change(self, filename, value):
        # These are disposable synthetic fixtures, never original run outputs.
        (self.failure / filename).write_text(json.dumps(value), encoding="utf-8")

    def test_reuses_C_with_hashes_without_changing_prior_outputs(self):
        before = {path: path.read_bytes() for path in self.failure.iterdir()}
        evidence = diagnostic.load_failed_evidence(self.failure, self.case)
        self.assertEqual(evidence["C"]["token_probs"], [1.0, 0.25])
        self.assertEqual(evidence["attention"], "eager")
        self.assertEqual(len(evidence["hashes"]), 7)
        self.assertEqual(before, {path: path.read_bytes() for path in self.failure.iterdir()})

    def test_rejects_changed_raw_prefix(self):
        changed = dict(self.case, prefix_ids=[1, 2, 3, 20, 11])
        with self.assertRaisesRegex(ValueError, "inputs differ: prefix_ids"):
            diagnostic.load_failed_evidence(self.failure, changed)

    def test_rejects_changed_backend_source_identity(self):
        self.change("source-hashes.json", {})
        with self.assertRaisesRegex(ValueError, "source changed"):
            diagnostic.load_failed_evidence(self.failure, self.case)

    def test_rejects_relabelled_first_difference(self):
        changed = dict(self.records["first-difference.json"], actual=0.7)
        self.change("first-difference.json", changed)
        with self.assertRaisesRegex(ValueError, "does not agree"):
            diagnostic.load_failed_evidence(self.failure, self.case)

    def test_rejects_relaxed_tolerance_in_prior_evidence(self):
        plan = dict(self.records["plan.json"], tolerance={**validation.TOLERANCE, "atol": 0.1})
        self.change("plan.json", plan)
        with self.assertRaisesRegex(ValueError, "unchanged pinned exact"):
            diagnostic.load_failed_evidence(self.failure, self.case)

    def test_cache_source_matches_fixed_git_export_and_refuses_modified_helper(self):
        raw, function, source = diagnostic.cache_source()
        exported = self.root / "export"
        exported.mkdir()
        (exported / diagnostic.CACHE_FILE).write_bytes(raw)
        self.assertEqual(diagnostic.cache_source(exported)[0], raw)
        self.assertEqual(source["sha256"], diagnostic.CACHE_SHA256)
        self.assertEqual(function.name, "cache_to_device")
        (exported / diagnostic.CACHE_FILE).write_bytes(raw + b"\n")
        with self.assertRaisesRegex(ValueError, "differs from the frozen"):
            diagnostic.cache_source(exported)

    def test_environment_must_remain_identical_and_bf16(self):
        environment = dict.fromkeys(diagnostic.ENVIRONMENT_FIELDS, "recorded")
        environment["model_parameter_dtype"] = "torch.bfloat16"
        diagnostic.verify_environment(environment, dict(environment))
        with self.assertRaisesRegex(ValueError, "attention_implementation"):
            diagnostic.verify_environment(environment, dict(environment, attention_implementation="sdpa"))
        wrong_dtype = dict(environment, model_parameter_dtype="torch.float32")
        with self.assertRaisesRegex(ValueError, "unchanged BF16"):
            diagnostic.verify_environment(wrong_dtype, wrong_dtype)

    def test_same_token_confidence_cannot_hide_probability_failure(self):
        expected = self.records["q001-probe-0-reference-probe.json"]
        actual = self.records["q001-probe-0-online-probe.json"]
        report = diagnostic.compare_paths(expected, actual, "C", "A")
        self.assertEqual(report["status"], "failed")
        self.assertTrue(report["token_ids_identical"])
        self.assertTrue(report["confidence_identical"])
        self.assertEqual(report["first_difference"]["field"], "token_probs[1]")
        self.assertEqual(report["probability_differences"][0]["absolute_difference"], 0.25)

    def test_inspect_only_has_frozen_three_path_and_64_token_bounds(self):
        output = io.StringIO()
        with redirect_stdout(output), patch.object(diagnostic, "gpu_inventory") as inventory:
            code = diagnostic.main(["--evidence-root", str(self.evidence), "--failure-root", str(self.failure),
                                    "--inspect-only"])
        inventory.assert_not_called()
        self.assertEqual(code, 0)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["new_comparison_paths"], ["A", "D", "B"])
        self.assertEqual(plan["reused_path"], "C")
        self.assertEqual(plan["maximum_new_probe_calls"], 6)
        self.assertEqual(plan["isolation_continuation_cap"], 64)
        self.assertNotIn("atol", vars(diagnostic.parser().parse_args([
            "--evidence-root", "x", "--failure-root", "y"])))

    def test_fullprefix_failure_and_isolation_success_remain_independent(self):
        import torch_online_backend
        run = self.root / "new-diagnostic"
        expected = self.records["q001-probe-0-reference-probe.json"]
        actual = self.records["q001-probe-0-online-probe.json"]
        reports = {"full_prefix_parity": diagnostic.compare_paths(expected, actual, "C", "A"),
                   "same_KV_parity": diagnostic.compare_paths(actual, actual, "D", "A")}
        backend = SimpleNamespace(metadata={}, torch=None)
        with patch.object(diagnostic, "gpu_inventory", return_value={"uuid": "GPU-test"}), \
                patch.object(diagnostic, "gpu_lock", return_value=nullcontext("test.lock")), \
                patch.object(diagnostic, "assert_gpu_idle", return_value={"compute_processes": []}), \
                patch.object(torch_online_backend, "TorchOnlineBackend", return_value=backend), \
                patch.object(diagnostic, "verify_environment"), \
                patch.object(validation, "validate_tokenizer_prefix"), \
                patch.object(diagnostic, "load_cached_reference"), \
                patch.object(diagnostic, "run_paths", return_value=reports), \
                patch.object(validation, "validate_isolation", return_value=[{"exact": True}]) as isolate, \
                redirect_stdout(io.StringIO()):
            code = diagnostic.main(["--evidence-root", str(self.evidence), "--failure-root", str(self.failure),
                                    "--model-dir", str(self.root / "model"), "--run-root", str(run)])
        self.assertEqual(code, 2)
        self.assertEqual(isolate.call_args.args[-1], 64)
        summary = json.loads((run / "summary.json").read_text())
        self.assertEqual(summary["strict_full_prefix_gate"], "failed")
        self.assertEqual(summary["same_KV_parity"]["status"], "passed")
        self.assertEqual(summary["isolation"]["status"], "passed")
        self.assertEqual(summary["overall_GPU_acceptance"], "not_established")
        self.assertEqual((run / "path-C-reused-reference.json").read_bytes(),
                         (self.failure / "q001-probe-0-reference-probe.json").read_bytes())


@unittest.skipUnless(base_tests.RUNTIME_AVAILABLE, "requires the pinned tiny-BF16 CPU environment")
class TinyCachedReferenceTests(unittest.TestCase):
    setUp = base_tests.TinyBF16AcceptanceTests.setUp

    def test_original_cached_helper_and_all_three_new_paths_preserve_main_state(self):
        import torch
        from copy import deepcopy
        _, probe_function, _ = validation.reference_source()
        _, cache_function, _ = diagnostic.cache_source()
        reference = diagnostic.load_cached_reference(probe_function, cache_function, torch)
        self.assertIs(reference.__globals__["deepcopy"], deepcopy)
        self.assertEqual(reference.__globals__["cache_to_device"].__code__.co_filename,
                         f"{validation.UPSTREAM_COMMIT}:cache_utils.py")
        full = self.case["prefix_ids"] + list(self.backend.markers.trial_prefix)
        with patch.object(torch.cuda, "empty_cache"), redirect_stdout(io.StringIO()):
            original = reference(self.backend.model, torch.tensor([full]), None, self.backend.tokenizer, method=0)
            C = {"token_ids": original["token_ids"], "token_probs": original["token_probs"],
                 "confidence": original["total_prob_max"], "ended": bool(original["ended_with_think"])}
            previous_online = {**C, "token_probs": [0.0, *C["token_probs"][1:]]}
            reports = diagnostic.run_paths(self.backend, reference, self.case, self.root, C, 872, previous_online)
        reproducibility = reports.pop("online_replay_reproducibility")
        self.assertEqual(reproducibility["status"], "failed")
        self.assertEqual(reproducibility["first_difference"]["field"], "token_probs[0]")
        self.assertTrue(all(row["exact"] for row in reports.values()))
        A = json.loads((self.root / "path-A.json").read_text())
        D = json.loads((self.root / "path-D.json").read_text())
        self.assertEqual(A["state_before"], D["state_before"])
        self.assertEqual(A["state_before"], A["state_after"])
        self.assertEqual(D["state_before"], D["state_after"])
        self.assertEqual(A["probability_bf16_bits"], D["probability_bf16_bits"])
        self.assertFalse((self.root / "first-difference.json").exists())


if __name__ == "__main__":
    unittest.main()
