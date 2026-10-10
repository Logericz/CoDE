"""真实追加验收的纯CPU选择/身份/预算测试；不导入torch或连接GPU。"""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import validate_entropy_extension as validator


def snapshot(n, valid=True):
    return {"token_ids": [7] * n, "token_probs": [.7] * n, "confidence_raw": .7,
            "confidence_nonfinite": None, "confidence_valid": valid,
            "entropy_raw_fp32_nats": [.4] * n, "entropy_valid": valid,
            "invalid_reason": None if valid else "invalid_confidence",
            "ended_with_think": n < 21, "ended_with_eos": False, "actual_length": n}


def save_candidate(root, question, candidate, short=21, long=42, valid=True):
    folder = root / f"question-{question}"
    folder.mkdir(exist_ok=True)
    partial = folder / f"candidate-{candidate}-partial"
    partial.mkdir()
    prefix = {"question_id": f"q{question}", "candidate_index": candidate,
              "prompt_token_ids": [101, 102], "main_token_ids": [7, 8], "pending_wait": 11}
    row = {"question_id": f"q{question}", "candidate_index": candidate, "prefix_tokens": 2,
           "prefix_sha256": hashlib.sha256(validator.encoded([101, 102, 7, 8])).hexdigest(),
           "short_raw": snapshot(short, valid), "long_raw": snapshot(long, valid),
           "long_additional_tokens": long - short}
    path = folder / f"candidate-{candidate}.json"
    path.write_text(json.dumps(row))
    (partial / "prefix.json").write_text(json.dumps(prefix))
    return path, row


class ExtensionValidationTests(unittest.TestCase):
    def test_first_eligible_uses_numeric_order_and_ignores_answer_quality(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_candidate(root, 1, 1, short=5, long=5)
            save_candidate(root, 1, 2, valid=False)
            save_candidate(root, 10, 1)
            save_candidate(root, 2, 10)
            path, row = save_candidate(root, 2, 2)
            row["actions"] = {"E": {"error": 1}, "T": {"error": 1}}
            path.write_text(json.dumps(row))
            selected = validator.first_eligible(root)
            self.assertEqual(selected[:2], (2, 2))

    def test_terminated_five_to_five_is_not_extension(self):
        self.assertFalse(validator.substantive_extension(snapshot(5), snapshot(5)))
        self.assertFalse(validator.compare_saved(snapshot(5), snapshot(5),
            {"short_raw": snapshot(5), "long_raw": snapshot(5)})["substantive_extension"])
        self.assertTrue(validator.substantive_extension(snapshot(21), snapshot(22)))

    def test_saved_prefix_hash_identity_and_position_must_match(self):
        for field, value, expected in (("prefix_sha256", "0" * 64, "SHA256"),
                                       ("prefix_tokens", 3, "position"),
                                       ("question_id", "wrong", "ID")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path, row = save_candidate(root, 1, 1)
                row[field] = value
                path.write_text(json.dumps(row))
                selected = validator.first_eligible(root)
                with self.assertRaisesRegex(ValueError, expected):
                    validator.load_prefix(selected, {"question_ids": ["q1"]})

    def test_prefix_replay_budget_cannot_exceed_8192(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_candidate(root, 1, 1)
            path = root / "question-1/candidate-1-partial/prefix.json"
            prefix = json.loads(path.read_text())
            prefix["main_token_ids"] = [7] * 8193
            path.write_text(json.dumps(prefix))
            with self.assertRaisesRegex(ValueError, "over-budget"):
                validator.load_prefix(validator.first_eligible(root), {"question_ids": ["q1"]})

    def test_additional_count_mismatch_fails_without_trying_later_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, row = save_candidate(root, 1, 1)
            row["long_additional_tokens"] = 1
            path.write_text(json.dumps(row))
            save_candidate(root, 2, 1)
            with self.assertRaisesRegex(ValueError, "count contradicts"):
                validator.first_eligible(root)

    def test_archived_source_hashes_are_all_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hashes = {}
            for relative in validator.REQUIRED_SOURCES:
                path = root / relative
                path.parent.mkdir(exist_ok=True)
                path.write_text("# fixed\n")
                hashes[relative] = validator.sha256(path)
            self.assertEqual(validator.verify_sources({"code_sha256": hashes}, root), hashes)
            (root / "src/entropy_probe.py").write_text("# changed\n")
            with self.assertRaisesRegex(ValueError, "Frozen source mismatch"):
                validator.verify_sources({"code_sha256": hashes}, root)

    def test_budget_expires_and_cannot_be_enlarged(self):
        with patch.object(validator.time, "monotonic", return_value=1000):
            with self.assertRaises(TimeoutError):
                validator.check_budget(100, 900)
            validator.check_budget(101, 900)
        for value in (0, -1, 901, True, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validator.check_budget(0, value)

    def test_no_extension_writes_not_covered_without_gpu_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {"protocol": "entropy-value-paired-v1", "model_id": validator.MODEL_ID,
                        "model_revision": validator.MODEL_REVISION}
            (root / "manifest.json").write_text(json.dumps(manifest))
            (root / "summary.json").write_text(json.dumps({"status": "completed", "source_data_integrity": True}))
            save_candidate(root, 1, 1, short=5, long=5)
            output = root / "validation.json"
            with patch.object(validator, "verify_sources", return_value={}), \
                    patch.object(validator, "gpu_inventory", side_effect=AssertionError("GPU accessed")), redirect_stdout(io.StringIO()):
                code = validator.main(["--run-root", str(root), "--model-dir", str(root / "model"), "--output", str(output)])
            result = json.loads(output.read_text())
            self.assertEqual(code, 2)
            self.assertEqual(result["status"], "not_covered")
            self.assertFalse(result["passed"])
            self.assertFalse((root / "validation.same-kv.json").exists())

    def test_existing_output_or_gate_cannot_be_overwritten(self):
        for existing_name in ("validation.json", "validation.same-kv.json"):
            with self.subTest(existing_name=existing_name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                existing = root / existing_name
                existing.write_text("preserve")
                with self.assertRaises(FileExistsError):
                    validator.main(["--run-root", str(root), "--model-dir", str(root / "model"),
                                    "--output", str(root / "validation.json")])
                self.assertEqual(existing.read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
