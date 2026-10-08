"""CPU checks for pending review packets; no review or model-quality evidence."""
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_answer_review as review
import math_grading as grading


def record(identifier="hidden-original-id", answer="\\boxed{2}", **extra):
    return {"id": identifier, "gold": "hidden-gold", "answer_text": answer,
            "method": "hidden-method", "source_sha256": "a" * 64,
            "stop_reason": "hidden-stop", "elapsed_seconds": 123.456, **extra}


class ReviewPacketTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name).resolve()
        self.output = self.folder / "packet"

    def write_rows(self, rows, name="answers.jsonl"):
        path = self.folder / name
        path.write_bytes(b"".join(review.encoded(row) for row in rows))
        return path

    def prepare(self, inputs, **kwargs):
        return review.prepare_review(inputs, self.output, scope=review.SCOPE, **kwargs)

    def test_all_rows_pending_exact_text_and_metadata_isolation(self):
        text = "\r\n答案🙂 e\u0301 \\boxed{2}</think>\n unfinished \\boxed{"
        rows = [record("private-first", text, answer_boundary_confirmed=False),
                record("private-second", "", execution_status="failed"),
                record("private-third", " no boxed answer ", status="incorrect", grade=False)]
        first = self.write_rows(rows[:2])
        second = self.write_rows(rows[2:], "second.jsonl")
        original_bytes = {path: path.read_bytes() for path in (first, second)}
        strict = self.folder / "strict-grades.json"
        strict.write_text("strict report is unchanged", encoding="utf-8")
        # Preparing evidence must never call extraction, grading or dependency checks.
        with ExitStack() as stack:
            for name in ("extract_final_answer", "grade_records", "grade_record", "_run_worker", "_check_dependencies"):
                stack.enter_context(patch.object(grading, name, side_effect=AssertionError("grading forbidden")))
            manifest = self.prepare([first, second])
        cases = grading.read_jsonl(self.output / "reviewer/cases.jsonl")
        mapping = grading.read_jsonl(self.output / "private/mapping.jsonl")
        self.assertEqual(len(cases), len(rows))
        self.assertEqual(len({case["case_id"] for case in cases}), len(rows))
        by_case = {item["case_id"]: item for item in mapping}
        for case in cases:
            self.assertEqual(set(case), {"case_id", "answer_text", "review"})
            self.assertEqual(uuid.UUID(hex=case["case_id"]).version, 4)
            self.assertEqual(case["review"], {"status": "pending", "boundary_decision": None,
                                             "selected_expression_verbatim": None,
                                             "selected_expression_char_span": None, "review_note": None})
            private = by_case[case["case_id"]]
            original = private["original_record"]
            self.assertEqual(case["answer_text"].encode("utf-8"), original["answer_text"].encode("utf-8"))
            self.assertEqual(private["answer_text_sha256"], hashlib.sha256(case["answer_text"].encode()).hexdigest())
            self.assertEqual(private["original_record_canonical_sha256"], review.digest_bytes(review.encoded(original)))
            self.assertNotEqual(case["case_id"], original["id"])
            input_path = Path(private["input_path"])
            self.assertEqual(private["input_sha256"], hashlib.sha256(original_bytes[input_path]).hexdigest())
            self.assertEqual(grading.read_jsonl(input_path)[private["input_row_number"] - 1], original)
        self.assertEqual({item["original_record"]["id"] for item in mapping}, {row["id"] for row in rows})
        reviewer_bytes = (self.output / "reviewer/cases.jsonl").read_bytes()
        for forbidden in (b"hidden-gold", b"hidden-method", b"hidden-stop", b"private-first", b"123.456", str(first).encode()):
            self.assertNotIn(forbidden, reviewer_bytes)
        self.assertEqual(manifest["case_count"], 3)
        self.assertTrue(manifest["all_reviews_pending"])
        self.assertFalse(manifest["adjudication_performed"])
        self.assertFalse(manifest["math_grading_performed"])
        self.assertFalse(manifest["strict_reports_modified"])
        self.assertIn("development", " ".join(manifest["blinding_limitations"]))
        self.assertIn("cannot claim", " ".join(manifest["blinding_limitations"]))
        self.assertIn("assertion", manifest["provenance_limit"])
        for path, before in original_bytes.items():
            self.assertEqual(path.read_bytes(), before)
        self.assertEqual(strict.read_text(), "strict report is unchanged")

    def test_manifest_hashes_and_private_permissions(self):
        source = self.write_rows([record()])
        manifest = self.prepare([source])
        self.assertEqual(json.loads((self.output / "manifest.json").read_text()), manifest)
        self.assertEqual(manifest["inputs"], [{"path": str(source), "sha256": grading.file_sha256(source), "row_count": 1}])
        self.assertEqual(manifest["preparation_source_sha256"], grading.file_sha256(Path(review.__file__)))
        for name, identity in manifest["outputs"].items():
            path = self.output / name
            self.assertEqual(identity, {"sha256": grading.file_sha256(path), "bytes": path.stat().st_size})
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        for path in (self.output, self.output / "private", self.output / "reviewer"):
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE((self.output / "manifest.json").stat().st_mode), 0o600)

    def test_duplicate_ids_across_inputs_and_repeat_input_rejected(self):
        first = self.write_rows([record()])
        second = self.write_rows([record(answer="different answer")], "second.jsonl")
        for inputs in ([first, second], [first, first]):
            with self.subTest(inputs=inputs), self.assertRaisesRegex(ValueError, "duplicate id"):
                self.prepare(inputs)
            self.assertFalse(self.output.exists())

    def test_existing_output_file_directory_and_broken_symlink_are_preserved(self):
        source = self.write_rows([record()])
        for kind in ("file", "directory", "symlink"):
            with self.subTest(kind=kind):
                if kind == "directory":
                    self.output.mkdir()
                    (self.output / "sentinel").write_text("keep")
                elif kind == "file":
                    self.output.write_text("keep")
                else:
                    self.output.symlink_to(self.folder / "missing")
                with self.assertRaisesRegex(ValueError, "already exists"):
                    self.prepare([source])
                if kind == "directory":
                    self.assertEqual((self.output / "sentinel").read_text(), "keep")
                    (self.output / "sentinel").unlink()
                    self.output.rmdir()
                elif kind == "file":
                    self.assertEqual(self.output.read_text(), "keep")
                    self.output.unlink()
                else:
                    self.assertTrue(self.output.is_symlink())
                    self.output.unlink()

    def test_invalid_or_unexecuted_rows_fail_before_publication(self):
        invalid = [[], "not a row", {"id": "x", "gold": "2"},
                   record(execution_status="not_run"), record(answer_boundary_confirmed="true"),
                   record(source_sha256="bad"), record(gold="")]
        for row in invalid:
            with self.subTest(row=row):
                source = self.write_rows([row])
                with self.assertRaises(ValueError):
                    self.prepare([source])
                self.assertFalse(self.output.exists())

    def test_bad_json_empty_input_and_missing_input_rejected(self):
        for content in (b"", b"\n", b'{"id":"first","id":"second"}\n', b'{"x":NaN}\n', b'{"x":\n'):
            with self.subTest(content=content):
                source = self.folder / "input.jsonl"
                source.write_bytes(content)
                with self.assertRaises(ValueError):
                    self.prepare([source])
                self.assertFalse(self.output.exists())
        with self.assertRaises(FileNotFoundError):
            self.prepare([self.folder / "missing.jsonl"])
        with self.assertRaises(ValueError):
            self.prepare([])

    def test_declared_answer_digest_is_verified_before_publication(self):
        source = self.write_rows([record(answer_text_sha256="f" * 64)])
        with self.assertRaisesRegex(ValueError, "answer_text_sha256"):
            self.prepare([source])
        self.assertFalse(self.output.exists())
        source = self.write_rows([record(answer_text_sha256=review.digest_bytes(b"\\boxed{2}"))])
        self.prepare([source])

    def test_only_explicit_development_scope_allowed(self):
        source = self.write_rows([record()])
        with self.assertRaisesRegex(ValueError, "development-exposed"):
            review.prepare_review([source], self.output, scope="confirmatory")
        self.assertFalse(self.output.exists())

    def test_input_change_while_reading_rejected(self):
        source = self.write_rows([record()])
        original_read = review.read_jsonl

        def read_then_change(path):
            rows = original_read(path)
            path.write_bytes(path.read_bytes() + b"\n")
            return rows

        with patch.object(review, "read_jsonl", side_effect=read_then_change):
            with self.assertRaisesRegex(ValueError, "Input changed"):
                self.prepare([source])
        self.assertFalse(self.output.exists())

    def test_input_change_during_staging_rejected_before_output(self):
        source = self.write_rows([record()])
        original_write = review.write_exclusive

        def write_then_change(path, payload):
            original_write(path, payload)
            if path.name == "manifest.json":
                source.write_bytes(source.read_bytes() + b"\n")

        with patch.object(review, "write_exclusive", side_effect=write_then_change):
            with self.assertRaisesRegex(ValueError, "Input changed"):
                self.prepare([source])
        self.assertFalse(self.output.exists())

    def test_input_change_during_publication_leaves_no_complete_manifest(self):
        source = self.write_rows([record()])
        original_link = os.link

        def link_then_change(src, dst):
            original_link(src, dst)
            if Path(dst).name == "cases.jsonl":
                source.write_bytes(source.read_bytes() + b"\n")

        with patch.object(review.os, "link", side_effect=link_then_change):
            with self.assertRaisesRegex(ValueError, "Input changed"):
                self.prepare([source])
        self.assertTrue(self.output.exists())
        self.assertFalse((self.output / "manifest.json").exists())
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.prepare([source])

    def test_racing_output_creation_never_overwrites(self):
        source = self.write_rows([record()])
        original_write = review.write_exclusive

        def write_then_create(path, payload):
            original_write(path, payload)
            if path.name == "manifest.json":
                self.output.mkdir()
                (self.output / "sentinel").write_text("keep")

        with patch.object(review, "write_exclusive", side_effect=write_then_create):
            with self.assertRaises(FileExistsError):
                self.prepare([source])
        self.assertEqual([item.name for item in self.output.iterdir()], ["sentinel"])
        self.assertEqual((self.output / "sentinel").read_text(), "keep")

    def test_interruption_during_publication_has_no_manifest(self):
        source = self.write_rows([record()])
        with patch.object(review.os, "link", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.prepare([source])
        self.assertTrue(self.output.exists())
        self.assertFalse((self.output / "manifest.json").exists())

    def test_changed_published_copy_has_no_complete_manifest(self):
        source = self.write_rows([record()])
        original = source.read_bytes()
        original_link = os.link

        def link_then_change(src, dst):
            original_link(src, dst)
            if Path(dst).name == "cases.jsonl":
                Path(dst).write_bytes(b"changed reviewer copy\n")

        with patch.object(review.os, "link", side_effect=link_then_change):
            with self.assertRaisesRegex(ValueError, "Published output changed"):
                self.prepare([source])
        self.assertFalse((self.output / "manifest.json").exists())
        self.assertEqual(source.read_bytes(), original)

    def test_cli_runs_without_site_packages_and_rejects_reuse(self):
        first = self.write_rows([record("a")])
        second = self.write_rows([record("b")], "second.jsonl")
        command = [sys.executable, "-S", str(ROOT / "scripts/prepare_answer_review.py"),
                   "--input", str(first), "--input", str(second), "--output-dir", str(self.output),
                   "--scope", "development-exposed"]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["case_count"], 2)
        self.assertTrue(json.loads(completed.stdout)["all_reviews_pending"])
        repeated = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(repeated.returncode, 2)
        self.assertIn("already exists", repeated.stderr)


if __name__ == "__main__":
    unittest.main()
