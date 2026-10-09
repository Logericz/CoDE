"""Bounded-runner orchestration tests with synthetic CPU data/backends, not GPU evidence."""
from contextlib import contextmanager, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("run_online_diagnostic", ROOT / "scripts/run_online_diagnostic.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
from online_contract import TokenMarkers, TokenSample
from online_protocol import ProbeObservation

WAIT, END, EOS, PREFIX, VALUE = 11, 12, 13, 14, 15


class SyntheticBackend:
    markers = TokenMarkers(WAIT, END, EOS, (PREFIX,), (END, PREFIX))
    context_limit = 10000
    metadata = {"scope": "synthetic_cpu_fixture", "validation_model": True}

    def __init__(self, *args, fail_request=None, **kwargs):
        self.requests = 0
        self.fail_request = fail_request
        self.questions, self.seeds = [], []

    def prompt_ids(self, question):
        self.questions.append(question)
        return [1, 2]

    def start_request(self, seed):
        self.requests += 1
        self.count = self.answer_count = 0
        self.seeds.append(seed)

    def prefill(self, ids):
        return list(ids)

    def sample(self, state):
        self.count += 1
        return TokenSample(WAIT if self.count % 2 == 0 else 7, 1.0, 0.5)

    def greedy(self, state):
        token = (VALUE, EOS)[self.answer_count % 2]
        self.answer_count += 1
        return TokenSample(token, 1.0, 0.5)

    def extend(self, state, ids):
        if self.requests == self.fail_request and self.count >= 3:
            raise RuntimeError("Synthetic KV failure")
        return state + list(ids)

    def probe(self, state):
        return ProbeObservation((31, 32, END), (0.5, 0.5, 0.5), 0.2, True,
                                confidence_source="synthetic_cpu_fixture")

    def decode(self, ids, *, skip_special_tokens=True):
        pieces = {END: "</think>", WAIT: "Wait", EOS: "<eos>", PREFIX: r"\boxed", VALUE: "{7}"}
        return "".join(pieces.get(token, "x") for token in ids
                       if not (skip_special_tokens and token in (END, EOS)))

    def synchronize(self):
        pass

    def peak_memory_bytes(self):
        return None


def frozen_fixture(directory):
    directory.mkdir()
    source = "EleutherAI/hendrycks_math"
    def rows(prefix, count):
        result = []
        for index in range(count):
            problem = f"Synthetic {prefix} problem {index}"
            result.append({"id": f"{prefix}/{index}", "problem": problem, "answer": "7",
                           "solution": r"\boxed{7}", "needs_review": False, "source": source,
                           "subject": "algebra", "level": "1", "problem_sha256": runner.problem_hash(problem)})
        return result
    selection = rows("selection", 120)
    selection[0]["id"] = runner.DEFAULT_SAMPLE
    test = rows("test", 500)
    splits = {"calibration80": rows("calibration", 80), "selection120": selection,
              "pilot20": selection[:20], "math500": test, "analysis100": test[:100], "answer_review": []}
    lock = {name: str(index) * 40 for index, name in enumerate(sorted(runner.DATA_SOURCES), 1)}
    (directory / "sources.lock.json").write_bytes(runner.encoded(lock))
    files = {"sources.lock.json": {"sha256": runner.sha256(directory / "sources.lock.json")}}
    for name, values in splits.items():
        path = directory / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in values), encoding="utf-8")
        files[path.name] = {"sha256": runner.sha256(path), "count": len(values), "ids": [row["id"] for row in values],
                            "problem_sha256": [row["problem_sha256"] for row in values]}
    manifest = directory / "manifest.json"
    manifest.write_bytes(runner.encoded({"schema_version": 1, "sources": lock, "files": files}))
    return runner.sha256(manifest)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.data = self.base / "data"
        self.digest = frozen_fixture(self.data)
        self.model = self.base / runner.MODEL_REVISION
        self.model.mkdir()

    def args(self, *extra):
        return runner.parser().parse_args(["--data-dir", str(self.data), "--data-manifest-sha256", self.digest,
            "--model-dir", str(self.model), "--run-root", str(self.base / "new-run"),
            "--max-new-tokens", "8", *extra])

    @contextmanager
    def mocked_gpu(self):
        gpu = {"uuid": "GPU-synthetic", "name": "synthetic CPU test"}
        original_lock = runner.gpu_lock
        @contextmanager
        def patched_lock(uuid):
            with original_lock(uuid, directory=self.base) as path:
                yield path
        with patch.object(runner, "gpu_inventory", return_value=gpu), \
             patch.object(runner, "assert_gpu_idle", return_value={"compute_processes": []}), \
             patch.object(runner, "gpu_lock", patched_lock), \
             patch.object(runner.importlib.metadata, "version", return_value="synthetic"), \
             redirect_stdout(io.StringIO()):
            yield

    def test_exact_frozen_identity_and_pilot_only_selection(self):
        row, identity = runner.verify_data(self.data, self.digest, runner.DEFAULT_SAMPLE)
        self.assertEqual(row["id"], runner.DEFAULT_SAMPLE)
        self.assertEqual(identity["manifest_sha256"], self.digest)
        with self.assertRaisesRegex(ValueError, "pilot20"):
            runner.verify_data(self.data, self.digest, "test/0")

    def test_manifest_anchor_and_file_tampering_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "manifest SHA"):
            runner.verify_data(self.data, "0" * 64, runner.DEFAULT_SAMPLE)
        path = self.data / "pilot20.jsonl"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            runner.verify_data(self.data, self.digest, runner.DEFAULT_SAMPLE)

    def test_source_lock_must_match_manifest_even_when_hashes_are_valid(self):
        path = self.data / "manifest.json"
        manifest = runner.read_json(path)
        manifest["sources"]["EleutherAI/hendrycks_math"] = "f" * 40
        path.write_bytes(runner.encoded(manifest))
        with self.assertRaisesRegex(ValueError, "revision lock"):
            runner.verify_data(self.data, runner.sha256(path), runner.DEFAULT_SAMPLE)

    def test_seed_domains_question_rollout_and_config_pairing(self):
        baseline = runner.derive_seed(42, "q2", 0, "reason")
        self.assertEqual(baseline, runner.derive_seed(42, "q2", 0, "reason"))
        self.assertEqual(len({baseline, runner.derive_seed(42, "q2", 1, "reason"),
                             runner.derive_seed(42, "q1", 0, "reason"),
                             runner.derive_seed(42, "q2", 0, "schedule"),
                             runner.derive_seed(42, "q2", 0, "warmup")}), 5)

    def test_atomic_new_never_overwrites_prior_result(self):
        path = self.base / "result.json"
        runner.atomic_new(path, {"a": 1})
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            runner.atomic_new(path, {"a": 2})
        self.assertEqual(path.read_bytes(), before)

    def test_physical_gpu_lock_blocks_other_run_and_releases(self):
        with runner.gpu_lock("GPU-same-device", directory=self.base):
            with self.assertRaisesRegex(RuntimeError, "already locked"):
                with runner.gpu_lock("GPU-same-device", directory=self.base):
                    self.fail("Concurrent GPU lock should be impossible")
        with runner.gpu_lock("GPU-same-device", directory=self.base):
            pass

    def test_run_matrix_warmup_pairing_progress_and_atomic_answers(self):
        backend = SyntheticBackend()
        args = self.args()
        with self.mocked_gpu():
            self.assertEqual(runner.run_diagnostic(args, backend_factory=lambda *a, **kw: backend), 0)
        root = args.run_root
        summary = runner.read_json(root / "summary.json")
        self.assertEqual(summary["executed_count"], 3)
        self.assertTrue(summary["warmup_excluded"])
        self.assertEqual(backend.requests, 4)
        self.assertEqual(len(set(backend.seeds[1:])), 1)
        self.assertNotEqual(backend.seeds[0], backend.seeds[1])
        self.assertTrue(all(question == "Synthetic selection problem 0" for question in backend.questions[1:]))
        dense = runner.read_json(root / "codestop-dense.json")
        fixed = runner.read_json(root / "codestop-fixed.json")
        self.assertFalse(runner.read_json(root / "manifest.json")["schedule_rng_used"])
        self.assertEqual(dense["n_probes"], 4)
        self.assertEqual(fixed["n_probes"], 3)
        self.assertEqual(dense["main_samples"], fixed["main_samples"])
        answers = [json.loads(line) for line in (root / "final_answers.jsonl").read_text().splitlines()]
        self.assertEqual(len(answers), 3)
        self.assertEqual(len({row["id"] for row in answers}), 3)
        for row in answers:
            self.assertTrue(row["answer_boundary_confirmed"])
            self.assertEqual(row["answer_text"], r"\boxed{7}")
            self.assertEqual(row["removed_terminal_eos"][0]["token_id"], EOS)
            self.assertEqual(row["source_sha256"], runner.sha256(root / row["source_file"]))
        self.assertIn('"event": "probe_end"', (root / "events.jsonl").read_text())
        before = (root / "manifest.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            runner.run_diagnostic(args, backend_factory=lambda *a, **kw: backend)
        self.assertEqual((root / "manifest.json").read_bytes(), before)

    def test_all_families_keep_actual_configs_hashes_and_independent_random_seed(self):
        args = self.args("--configurations", *runner.SUPPORTED_CONFIGS, "--max-new-tokens", "40",
                         "--h-max", "4", "--beta", "0.25", "--log-a", "2", "--random-p", "0.2",
                         "--margin-m0", "0.1", "--r-max", "0.98", "--tau", "4", "--deer-threshold", "0.97")
        backend = SyntheticBackend()
        with self.mocked_gpu():
            self.assertEqual(runner.run_diagnostic(args, backend_factory=lambda *a, **kw: backend), 0)
        manifest = runner.read_json(args.run_root / "manifest.json")
        self.assertEqual(manifest["planned_count"], len(runner.SUPPORTED_CONFIGS))
        self.assertTrue(manifest["schedule_rng_used"])
        self.assertEqual(len(set(backend.seeds[1:])), 1)
        vanilla = runner.read_json(args.run_root / "vanilla.json")
        for config in manifest["configurations"]:
            result = runner.read_json(args.run_root / f"{config['label']}.json")
            self.assertEqual(result["request_configuration"], config)
            self.assertEqual(result["protocol_config"], config["protocol_config"])
            self.assertEqual(result["schedule_config"], config["schedule_config"])
            declared = {key: value for key, value in config.items() if key != "config_hash"}
            self.assertEqual(result["config_hash"], runner.hashlib.sha256(runner.encoded(declared)).hexdigest())
            self.assertFalse(result["eligible_for_primary_speed_comparison"])
            self.assertEqual(result["main_samples"], vanilla["main_samples"])
            expected_rule = "deer" if config["label"] == "deer-dense" else "codestop"
            self.assertEqual(result["protocol_config"]["rule"], expected_rule)
            self.assertEqual(result["protocol_config"]["r_max"], 0.98)
            self.assertEqual(result["protocol_config"]["deer_threshold"], 0.97)
            if config["label"] == "codestop-random":
                self.assertEqual(result["seed_schedule"], manifest["reserved_schedule_seed"])
                self.assertNotEqual(result["seed_schedule"], result["seed_reason"])
                self.assertEqual(result["schedule_config"]["h_max"], 8)
                self.assertEqual(result["schedule_config"]["random_p"], 0.2)
                self.assertTrue(any(probe["decision"]["h_next"] > 1 for probe in result["probes"]))
            else:
                self.assertIsNone(result["seed_schedule"])
            if config["label"] in ("codestop-log", "codestop-backoff", "codestop-adaptive"):
                self.assertEqual(result["schedule_config"]["h_max"], 4)
            if config["label"] == "codestop-guarded":
                self.assertEqual(result["schedule_config"]["kind"], "guarded")
                self.assertEqual(result["schedule_config"]["margin_m0"], 0.1)
                self.assertTrue(all(p["decision"]["h_next"] in (None, 1, 2) for p in result["probes"]))
                self.assertTrue(all(p["decision"]["h_cost"] is None for p in result["probes"]))
                self.assertTrue(any(p["decision"]["h_next"] == 2 for p in result["probes"]))
        events = [json.loads(line) for line in (args.run_root / "events.jsonl").read_text().splitlines()]
        decisions = [event for event in events if event["event"] == "probe_decision"]
        self.assertTrue(decisions)
        self.assertTrue(all(event["timing"] == "post_request_summary" for event in decisions))
        for label in runner.SUPPORTED_CONFIGS[1:]:
            last_probe = max(index for index, event in enumerate(events)
                             if event.get("configuration") == label and event["event"] == "probe_end")
            first_decision = next(index for index, event in enumerate(events)
                                  if event.get("configuration") == label and event["event"] == "probe_decision")
            self.assertGreater(first_decision, last_probe)

    def test_deer_stops_but_dense_collection_preserves_would_stop_and_continues(self):
        backend = SyntheticBackend()
        backend.probe = lambda state: ProbeObservation((31, 32, END), (0.99, 0.99, 0.99), 0.99, True,
                                                       confidence_source="synthetic_cpu_fixture")
        args = self.args("--configurations", "deer-dense", "dense-collect-no-stop")
        with self.mocked_gpu():
            self.assertEqual(runner.run_diagnostic(args, backend_factory=lambda *a, **kw: backend), 0)
        deer = runner.read_json(args.run_root / "deer-dense.json")
        dense = runner.read_json(args.run_root / "dense-collect-no-stop.json")
        self.assertEqual(deer["method"], "deer")
        self.assertEqual(deer["protocol_config"]["rule"], "deer")
        self.assertEqual(deer["n_probes"], 1)
        self.assertTrue(deer["probes"][0]["stop_applied"])
        self.assertEqual(dense["method"], "dense_collect_no_stop")
        self.assertEqual(dense["n_probes"], 4)
        self.assertEqual(dense["stop_reason"], "budget")
        self.assertTrue(all(probe["would_stop"] and not probe["stop_applied"] for probe in dense["probes"]))
        self.assertFalse(dense["request_configuration"]["stopping_enabled"])
        self.assertFalse(dense["eligible_for_primary_speed_comparison"])

    def test_random_schedule_repeats_queries_without_changing_main_stream(self):
        records = []
        for repeat in range(2):
            args = self.args("--configurations", "codestop-random", "--max-new-tokens", "40",
                             "--random-p", "0.2", "--run-root", str(self.base / f"repeat-{repeat}"))
            with self.mocked_gpu():
                self.assertEqual(runner.run_diagnostic(args, backend_factory=SyntheticBackend), 0)
            records.append(runner.read_json(args.run_root / "codestop-random.json"))
        self.assertEqual(records[0]["main_samples"], records[1]["main_samples"])
        self.assertEqual([probe["decision"]["candidate_j"] for probe in records[0]["probes"]],
                         [probe["decision"]["candidate_j"] for probe in records[1]["probes"]])
        self.assertEqual(records[0]["config_hash"], records[1]["config_hash"])

    def test_new_options_validate_before_gpu_and_original_defaults_stay_bounded(self):
        self.assertEqual(self.args().configurations, ["vanilla", "codestop-dense", "codestop-fixed"])
        for option in (("--h-max", "3"), ("--beta", "nan"), ("--random-p", "0.33"),
                       ("--tau", "0"), ("--r-max", "0.1")):
            with self.subTest(option=option), patch.object(runner, "gpu_inventory") as gpu:
                with self.assertRaises(ValueError):
                    runner.run_diagnostic(self.args(*option))
                gpu.assert_not_called()
        self.assertFalse((self.base / "new-run").exists())

    def test_failed_request_keeps_partial_and_denominator_stops_later_requests(self):
        backend = SyntheticBackend(fail_request=3)
        args = self.args()
        with self.mocked_gpu():
            self.assertEqual(runner.run_diagnostic(args, backend_factory=lambda *a, **kw: backend), 1)
        summary = runner.read_json(args.run_root / "summary.json")
        self.assertEqual((summary["executed_count"], summary["unexecuted_count"]), (2, 1))
        partial = runner.read_json(args.run_root / "codestop-dense.json")
        self.assertEqual(partial["status"], "failed")
        self.assertTrue(partial["output_token_ids"])
        self.assertIsNotNone(partial["pending_sample"])
        self.assertIn("backend_metadata", partial)
        self.assertEqual(partial["schedule_config"]["kind"], "dense")
        answers = [json.loads(line) for line in (args.run_root / "final_answers.jsonl").read_text().splitlines()]
        self.assertEqual(answers[1]["execution_status"], "failed")
        self.assertFalse((args.run_root / "codestop-fixed.json").exists())

    def test_boundary_adapter_preserves_repeated_markers_and_only_removes_terminal_eos(self):
        result = {"sample_id": "q", "rollout_id": 0, "method": "codestop", "status": "completed",
                  "answer_token_ids": [PREFIX, VALUE, END, VALUE, EOS], "answer_boundary_confirmed": True,
                  "answer_boundary": {"confirmed": True, "policy": "token_span_first_reasoning_exit_v1",
                                      "answer_token_span": [10, 15]}}
        path = self.base / "source.json"
        runner.atomic_new(path, result)
        answer = runner.answer_record(result, path, "7", SyntheticBackend(), "codestop-dense")
        self.assertEqual(answer["answer_text"], r"\boxed{7}</think>{7}")
        self.assertEqual(answer["removed_terminal_eos"][0]["output_token_index"], 14)
        result["answer_boundary_confirmed"] = False
        answer = runner.answer_record(result, path, "7", SyntheticBackend(), "codestop-dense")
        self.assertEqual(answer["answer_text"], "")
        self.assertFalse(answer["answer_boundary_confirmed"])

    def test_secondary_eos_is_removed_by_token_identity(self):
        backend = SyntheticBackend()
        backend.markers = TokenMarkers(WAIT, END, EOS, (PREFIX,), (END, PREFIX), eos_ids=(EOS, 99))
        result = {"sample_id": "q", "rollout_id": 0, "method": "vanilla", "status": "completed",
                  "answer_token_ids": [PREFIX, VALUE, 99], "answer_boundary_confirmed": True,
                  "answer_boundary": {"confirmed": True, "answer_token_span": [4, 7]}}
        path = self.base / "source.json"
        runner.atomic_new(path, result)
        answer = runner.answer_record(result, path, "7", backend, "vanilla")
        self.assertEqual(answer["answer_text"], r"\boxed{7}")
        self.assertEqual(answer["removed_terminal_eos"][0]["token_id"], 99)

    def test_answer_export_decode_failure_remains_in_executed_denominator(self):
        backend = SyntheticBackend()
        decode = backend.decode
        def fail_export_only(ids, *, skip_special_tokens=True):
            if backend.requests > 1 and list(ids) == [PREFIX, VALUE] and not skip_special_tokens:
                raise RuntimeError("Synthetic answer export decode failure")
            return decode(ids, skip_special_tokens=skip_special_tokens)
        backend.decode = fail_export_only
        args = self.args()
        with self.mocked_gpu():
            self.assertEqual(runner.run_diagnostic(args, backend_factory=lambda *a, **kw: backend), 1)
        summary = runner.read_json(args.run_root / "summary.json")
        self.assertEqual((summary["executed_count"], summary["unexecuted_count"]), (1, 2))
        source = runner.read_json(args.run_root / "vanilla.json")
        self.assertEqual(source["status"], "completed")
        answer = runner.read_json(args.run_root / "vanilla.final-answer.json")
        self.assertEqual(answer["execution_status"], "failed")
        self.assertEqual(answer["source_execution_status"], "completed")
        self.assertEqual(answer["source_sha256"], runner.sha256(args.run_root / "vanilla.json"))
        self.assertEqual(answer["answer_text"], "")
        self.assertEqual(summary["failure"]["stage"], "final_answer_export")

    def test_interrupted_started_request_is_failed_not_unexecuted(self):
        backend = SyntheticBackend()
        sample = backend.sample
        def interrupted(state):
            if backend.requests == 2:
                raise KeyboardInterrupt("synthetic interrupt")
            return sample(state)
        backend.sample = interrupted
        args = self.args()
        with self.mocked_gpu():
            self.assertEqual(runner.run_diagnostic(args, backend_factory=lambda *a, **kw: backend), 1)
        summary = runner.read_json(args.run_root / "summary.json")
        self.assertEqual((summary["executed_count"], summary["unexecuted_count"]), (1, 2))
        source = runner.read_json(args.run_root / "vanilla.json")
        self.assertEqual(source["status"], "failed")
        self.assertFalse(source["partial_evidence_available"])
        self.assertEqual(source["error_type"], "KeyboardInterrupt")

    def test_busy_gpu_and_multiple_visible_devices_are_refused(self):
        with patch.object(runner, "command_output", return_value="GPU-target, 123, python"):
            with self.assertRaisesRegex(RuntimeError, "compute processes"):
                runner.assert_gpu_idle("GPU-target")
        with patch.dict(runner.os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}):
            with self.assertRaisesRegex(ValueError, "exactly one"):
                runner.gpu_inventory()

    def test_invalid_budget_and_duplicate_configuration_rejected_before_gpu(self):
        with self.assertRaisesRegex(ValueError, "max-new-tokens"):
            runner.run_diagnostic(self.args("--max-new-tokens", "32768"))
        with self.assertRaisesRegex(ValueError, "distinct"):
            runner.run_diagnostic(self.args("--configurations", "vanilla", "vanilla"))

    def test_help_does_not_load_torch_or_touch_gpu(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/run_online_diagnostic.py"), "--help"],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0)
        self.assertIn("--max-new-tokens", result.stdout)


if __name__ == "__main__":
    unittest.main()
