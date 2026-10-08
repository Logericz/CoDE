"""CPU synthetic evidence audits; these fixtures are not GPU measurements."""
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_dense_collection as analysis

spec = importlib.util.spec_from_file_location("dense_analysis_engine_fixtures", ROOT / "tests/test_online_engine.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(analysis.encoded(value))


class DenseAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old, self.new = self.root / "development", self.root / "collection"
        self.samples = [7, fixtures.WAIT, 8, fixtures.WAIT, fixtures.END, fixtures.VALUE, fixtures.WAIT, fixtures.EOS]
        self.code = {k: analysis.sha256(ROOT / k) for k in
                     ("src/online_engine.py", "src/online_protocol.py", "src/online_contract.py", "src/torch_online_backend.py")}
        self.questions = [{"development_id": f"dev{i:02d}", "original_id": f"synthetic/{i}"} for i in range(1, 11)]
        self.make_run(self.old, ("vanilla", "codestop-dense", "codestop-fixed", "codestop-adaptive"))
        self.make_run(self.new, (analysis.LABEL,))
        self.anchor()

    def record(self, job):
        config = job["request_configuration"]
        backend = fixtures.Backend(self.samples, [fixtures.observation(0.99), fixtures.observation(0.2, ended=False)])
        backend.metadata = {**backend.metadata, "markers": asdict(backend.markers)}
        result = fixtures.run(backend, sample_id=job["sample_id"], method=config["method"],
                              max_new_tokens=32768, seed_reason=42,
                              schedule_config=analysis.ScheduleConfig(**config["schedule_config"]))
        result.update(development_id=job["development_id"], configuration=job["configuration"],
                      request_configuration=config, config_hash=config["config_hash"])
        return result

    def make_run(self, root, labels):
        root.mkdir()
        jobs, summaries = [], []
        for q in self.questions:
            for label in labels:
                method = "vanilla" if label == "vanilla" else analysis.METHOD if label == analysis.LABEL else "codestop"
                kind = "fixed" if label == "codestop-fixed" else "adaptive" if label == "codestop-adaptive" else "dense"
                config = {"label": label, "method": method, "seed_reason": 42,
                          "stopping_enabled": method not in ("vanilla", analysis.METHOD),
                          "protocol_config": asdict(analysis.ProtocolConfig()),
                          "schedule_config": asdict(analysis.ScheduleConfig(kind=kind))}
                config["config_hash"] = __import__("hashlib").sha256(analysis.encoded(config)).hexdigest()
                job = {"request_index": len(jobs), "development_id": q["development_id"],
                       "sample_id": q["original_id"], "configuration": label, "request_configuration": config,
                       "relative_directory": f"requests/{q['development_id']}/{label}"}
                path = root / job["relative_directory"] / "request.json"
                write(path, self.record(job))
                summaries.append({**job, "status": "completed", "source_execution_status": "completed",
                                  "source_sha256": analysis.sha256(path)})
                jobs.append(job)
        manifest = {"planned_count": len(jobs), "questions": self.questions, "requests": jobs,
                    "source_input_sha256": {"synthetic-frozen-input": "abc"}, "data_identity": {"synthetic": True},
                    "code_sha256": self.code}
        write(root / "manifest.json", manifest)
        for name in ("integrity-before.json", "integrity-after.json"):
            write(root / name, manifest["source_input_sha256"])
        write(root / "summary.json", {"planned_count": len(jobs), "executed_count": len(jobs),
            "completed_count": len(jobs), "failed_count": 0, "unexecuted_count": 0,
            "requests": summaries, "unexecuted_requests": [], "source_input_identity_integrity": "unchanged"})

    def anchor(self):
        manifest = json.loads((self.new / "manifest.json").read_text())
        summary = json.loads((self.old / "summary.json").read_text())
        pairs = {}
        for row in summary["requests"]:
            if row["configuration"] in ("vanilla", "codestop-dense"):
                pairs.setdefault(row["development_id"], {})[row["configuration"]] = {
                    "sha256": row["source_sha256"], "path": str(self.old / row["relative_directory"] / "request.json")}
        manifest["development_evidence"] = {"manifest_sha256": analysis.sha256(self.old / "manifest.json"),
            "summary_sha256": analysis.sha256(self.old / "summary.json"), "planned_count": 40, "paired_sources": pairs}
        write(self.new / "manifest.json", manifest)

    def alter(self, mutator, *, rehash=True):
        summary = json.loads((self.new / "summary.json").read_text())
        row = summary["requests"][0]
        path = self.new / row["relative_directory"] / "request.json"
        record = json.loads(path.read_text())
        mutator(record)
        write(path, record)
        if rehash:
            row["source_sha256"] = analysis.sha256(path)
            row["source_execution_status"] = record["status"]
            write(self.new / "summary.json", summary)

    def run_analysis(self):
        return analysis.analyze(self.new, self.old)

    def test_complete_tail_and_reason_answer_separation(self):
        result = self.run_analysis()
        self.assertEqual(result["status"], "passed", result["failed_checks"])
        self.assertEqual(result["counts"], {"planned": 10, "executed": 10, "completed": 10, "failed": 0, "unexecuted": 0})
        q = result["questions"][0]
        self.assertEqual((q["main_samples_saved"], q["reason_samples_saved"], q["natural_answer_samples_saved"]), (8, 5, 3))
        self.assertEqual(q["candidate_count"], 2)  # Wait in answer is excluded.
        self.assertEqual(q["first_would_stop"], [1, 1])
        self.assertEqual(q["probes_after_old_dense_stop"]["count"], 1)
        self.assertEqual(q["probes"]["incomplete_fraction"], 0.5)
        self.assertEqual(q["adaptive_replay"]["queried_candidate_j"], [1])
        self.assertIsNone(q["peak_allocated_bytes"])
        self.assertIsNone(q["peak_reserved_bytes"])

    def test_sample_probability_difference_is_not_hidden_by_equal_token_ids(self):
        self.alter(lambda r: r["main_samples"][0].update(raw_probability=0.25))
        result = self.run_analysis()
        self.assertEqual(result["status"], "failed")
        self.assertTrue(any(c["check"] == "dev01:vanilla_main_prefix" for c in result["failed_checks"]))

    def test_shared_probe_raw_probabilities_and_request_hash_are_verified(self):
        self.alter(lambda r: r["probes"][0]["raw_observation"]["token_probs"].__setitem__(0, 0.25), rehash=False)
        result = self.run_analysis()
        names = [c["check"] for c in result["failed_checks"]]
        self.assertTrue(any(c.endswith(":request_hash") for c in names))
        self.assertIn("dev01:shared_dense_raw_observations", names)

    def test_false_would_stop_and_answer_candidate_are_rejected(self):
        self.alter(lambda r: r["probes"][0].update(would_stop=False))
        result = self.run_analysis()
        self.assertEqual(result["status"], "failed")
        self.assertTrue(any(c["check"] == "dev01:first_would_stop_matches_old_dense" for c in result["failed_checks"]))

    def test_failed_partial_stays_failed_and_unexecuted_stays_in_denominator(self):
        summary = json.loads((self.new / "summary.json").read_text())
        manifest = json.loads((self.new / "manifest.json").read_text())
        for row in summary["requests"][1:]:
            (self.new / row["relative_directory"] / "request.json").unlink()
        summary.update(executed_count=1, completed_count=0, failed_count=1, unexecuted_count=9,
                       requests=summary["requests"][:1], unexecuted_requests=manifest["requests"][1:])
        summary["requests"][0]["status"] = "failed"
        write(self.new / "summary.json", summary)
        def partial(r):
            r.update(status="failed", main_samples=r["main_samples"][:1], candidates=[], probes=[],
                     finalization_samples=[], phase_elapsed_ms={p: 0.0 for p in analysis.PHASES},
                     elapsed_ms_before_error_report=5.0)
            for k in ("actual_generated_tokens", "time_total_ms"):
                r.pop(k)
        self.alter(partial)
        result = self.run_analysis()
        self.assertEqual(result["status"], "passed", result["failed_checks"])
        self.assertEqual(result["counts"]["failed"], 1)
        self.assertEqual(result["counts"]["unexecuted"], 9)
        self.assertFalse(result["collection_complete"])
        self.assertEqual(result["questions"][0]["elapsed"]["field"], "elapsed_ms_before_error_report")
        self.assertEqual(sum(q["execution_status"] == "unexecuted" for q in result["questions"]), 9)

    def test_escaped_path_and_protocol_identity_fail_closed(self):
        manifest = json.loads((self.new / "manifest.json").read_text())
        manifest["code_sha256"]["src/online_protocol.py"] = "changed"
        write(self.new / "manifest.json", manifest)
        self.assertEqual(self.run_analysis()["status"], "failed")
        with self.assertRaises(ValueError):
            analysis.inside(self.new, "../outside.json")

    def test_nonfinite_is_invalid_and_not_replaced_by_confidence(self):
        p = {"candidate_j": 1, "token_position": 1, "probe_elapsed_ms": 5.0,
             "raw_observation": asdict(fixtures.observation()), "decision": {"cost": {"reason_elapsed_ms": None}}}
        p["raw_observation"]["confidence_raw"] = {"nonfinite_float": "nan"}
        result = analysis.adaptive_replay([p], asdict(fixtures.Backend.markers), analysis.ProtocolConfig())
        self.assertIn("invalid_probe", result["decisions"][0]["forced_dense_reasons"])
        self.assertEqual(result["decisions"][0]["observation"]["confidence_raw"], {"nonfinite_float": "nan"})
        self.assertIsNone(result["online_latency_estimate_ms"])
        with self.assertRaises(ValueError):
            analysis.strict_json('{"x": NaN}')

    def test_replay_skips_observations_and_sums_only_elapsed_dense_intervals(self):
        probes = []
        for j in range(1, 10):
            probes.append({"candidate_j": j, "token_position": j, "probe_elapsed_ms": 10.0,
                           "raw_observation": asdict(fixtures.observation(0.2)),
                           "decision": {"cost": {"reason_elapsed_ms": 1.0 if j > 1 else None}}})
        # This skipped observation must not trigger a confidence stop.
        probes[3]["raw_observation"] = asdict(fixtures.observation(0.99))
        result = analysis.adaptive_replay(probes, asdict(fixtures.Backend.markers), analysis.ProtocolConfig())
        self.assertEqual(result["queried_candidate_j"], [1, 2, 3])
        self.assertEqual(result["skipped_candidate_j"], [4, 5, 6, 7, 8, 9])
        self.assertFalse(result["stopped"])
        self.assertEqual(result["scheduled_sparse_decisions_h_next_gt_1"], 1)
        # h=8 schedules candidate11; its absence is not counted as a skip.
        self.assertEqual(result["decisions"][-1]["next_candidate_j"], 11)
        for j in (10, 11):
            probes.append({"candidate_j": j, "token_position": j, "probe_elapsed_ms": 10.0,
                           "raw_observation": asdict(fixtures.observation(0.2)),
                           "decision": {"cost": {"reason_elapsed_ms": 1.0}}})
        result = analysis.adaptive_replay(probes, asdict(fixtures.Backend.markers), analysis.ProtocolConfig())
        self.assertEqual(result["decisions"][-1]["cost"]["reason_elapsed_ms"], 8.0)

    def test_changed_input_during_analysis_and_output_isolation(self):
        audit = analysis.Audit()
        p = self.root / "x.json"
        write(p, {"x": 1})
        audit.read(p)
        write(p, {"x": 2})
        audit.finish()
        self.assertFalse(audit.checks[-1]["passed"])
        report = self.run_analysis()
        with patch.object(analysis, "ROOT", self.root):
            out = analysis.write_report(report, self.root / "runs/new-report", [self.new, self.old])
            self.assertTrue((out / "analysis.md").is_file())
            with self.assertRaises(ValueError):
                analysis.write_report(report, out, [self.new, self.old])
            with self.assertRaises(ValueError):
                analysis.write_report(report, self.new / "analysis", [self.new, self.old])


if __name__ == "__main__":
    unittest.main()
