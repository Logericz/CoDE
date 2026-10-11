"""Synthetic-only checks for the old-data model freeze/new-question boundary."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from src import entropy_frozen_fusion as frozen
from src import entropy_short_fusion as fusion
from src import entropy_value_analysis as original
from test_entropy_value_analysis import fixture


def document(prefix, questions=20, candidates=2):
    value = fixture(questions, candidates)
    value["question_ids"] = [prefix + q for q in value["question_ids"]]
    for row in value["records"]:
        row["question_id"] = prefix + row["question_id"]
        row["short_probe_ms"] = 500.
    return value


IDENTITY = {"input_sha256": "a" * 64, "source_sha256": {"source.py": "b" * 64}}


class FrozenModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = document("old/")
        cls.new = document("new/", 7)
        cls.model = frozen.freeze(cls.old, IDENTITY)

    def test_json_roundtrip_predictions_exactly_match_direct_fit_all_penalties(self):
        model = json.loads(json.dumps(self.model, allow_nan=False))
        _, train, _ = fusion.prepare(self.old)
        _, held, _ = fusion.prepare(self.new)
        for penalty in original.ERROR_COSTS_MS:
            self.assertEqual(frozen.predict_frozen(model, held, penalty), fusion.predict_fold(train, held, penalty))

    def test_freeze_fits_exactly_twelve_fixed_models(self):
        with patch.object(original, "fit_ridge", wraps=original.fit_ridge) as fit:
            frozen.freeze(self.old, IDENTITY)
        self.assertEqual(fit.call_count, 12)
        self.assertTrue(all(call.args[2] == 1.0 for call in fit.call_args_list))

    def test_evaluation_does_not_fit_and_outcomes_only_change_metrics(self):
        changed = deepcopy(self.new)
        for row in changed["records"]:
            for a in "ET":
                row["actions"][a]["error"] = 1 - row["actions"][a]["error"]
                row["actions"][a]["remaining_ms"] += 12000 if a == "E" else 3000
        before = deepcopy(self.model)
        with patch.object(original, "fit_ridge", side_effect=AssertionError("evaluation fitted")):
            a = frozen.evaluate(self.new, self.model, 0)
            b = frozen.evaluate(changed, self.model, 0)
        self.assertEqual(self.model, before)
        for left, right in zip(a["results"], b["results"]):
            for old, new in zip(left["predictions"], right["predictions"]):
                self.assertEqual(old["scores"], new["scores"])
                self.assertEqual(old["actions"], new["actions"])
                self.assertNotEqual(old["delta_ms"], new["delta_ms"])

    def test_models_include_only_parameters_and_identity_not_training_text_or_answers(self):
        old = deepcopy(self.old)
        old["question_text"] = "PRIVATE_QUESTION"
        for row in old["records"]:
            row["actions"]["E"]["answer_text"] = "PRIVATE_ANSWER"
        model = frozen.freeze(old, IDENTITY)
        encoded = json.dumps(model)
        self.assertNotIn("PRIVATE_", encoded)
        self.assertNotIn("answer_text", encoded)
        self.assertNotIn("E_error", encoded)
        self.assertEqual(model["training"]["valid_records"], 40)
        self.assertEqual(model["training_identity"], IDENTITY)

    def test_overlap_checks_entire_roster_even_without_valid_rows(self):
        new = deepcopy(self.new)
        new["question_ids"].append(self.old["question_ids"][0])
        with self.assertRaisesRegex(ValueError, "overlap"):
            frozen.evaluate(new, self.model, 0)

    def test_tampering_is_rejected_before_prediction(self):
        model = deepcopy(self.model)
        model["models"][0]["trainmean"] += 1
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            frozen.validate_artifact(model)

    def test_rehashed_wrong_features_dimensions_scales_and_penalties_still_rejected(self):
        for mutation in (lambda m: m["feature_names"]["C"].reverse(),
                         lambda m: m["models"][0]["learned"]["C"]["coefficients"].pop(),
                         lambda m: m["models"][0]["learned"]["C"]["expanded_scale"].__setitem__(0, 0),
                         lambda m: m["lambda_error_ms"].__setitem__(0, 1)):
            model = deepcopy(self.model)
            mutation(model)
            model["payload_sha256"] = frozen.digest(frozen._payload(model))
            with self.assertRaises(ValueError):
                frozen.validate_artifact(model)

    def test_unknowns_remain_excluded_not_zero_and_empty_scope_not_estimable(self):
        new = deepcopy(self.new)
        for row in new["records"]:
            row["actions"]["E"]["error"] = None
        report = frozen.evaluate(new, self.model, 0)
        self.assertEqual(report["coverage"]["valid_paired_records"], 0)
        self.assertEqual(report["coverage"]["excluded_reason_counts"], {"unknown_or_invalid_answer_grade": 14})
        for result in report["results"]:
            self.assertEqual(result["predictions"], [])
            self.assertEqual(result["scopes"]["all"]["status"], "not_estimable")

    def test_held_feature_outlier_does_not_change_other_scores_or_frozen_scaling(self):
        _, rows, _ = fusion.prepare(self.new)
        changed = deepcopy(rows)
        for key in ("B", "H"):
            changed[0][key]["prefix_tokens"] = 1e12
        a = frozen.predict_frozen(self.model, rows, 60000)
        b = frozen.predict_frozen(self.model, changed, 60000)
        self.assertEqual(a[1:], b[1:])
        self.assertNotEqual(a[0]["scores"], b[0]["scores"])

    def test_summaries_reuse_shared_metrics_and_short_primary_membership(self):
        new = deepcopy(self.new)
        new["records"][0]["short"]["ended_with_think"] = True
        report = frozen.evaluate(new, self.model, 0)
        self.assertEqual(report["primary_records"], 13)
        for result in report["results"]:
            self.assertEqual(result["scopes"]["all"], fusion.summarize(result["predictions"], 0))
            self.assertEqual(set(result["scopes"]["all"]["methods"]), set(fusion.METHODS))

    def test_not_enough_training_and_invalid_protocol_options_fail(self):
        with self.assertRaisesRegex(ValueError, "four"):
            frozen.freeze(document("tiny/", 3), IDENTITY)
        for count in (-1, True, 1.5):
            with self.assertRaisesRegex(ValueError, "nonnegative"):
                frozen.evaluate(self.new, self.model, count)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            frozen.freeze(self.old, {"input_sha256": "bad", "source_sha256": {"source": "b" * 64}})


class FrozenCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("freeze_cli", ROOT / "scripts/freeze_entropy_fusion.py")
        cls.cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.cli)

    def test_freeze_then_evaluate_exclusive_outputs_and_source_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            old = document("old/")
            old["records"] = old["records"][:38]
            (path / "old.json").write_text(json.dumps(old))
            (path / "new.json").write_text(json.dumps(document("new/", 7)))
            digest = hashlib.sha256((path / "old.json").read_bytes()).hexdigest()
            freeze_args = ["freeze", "--input", str(path / "old.json"), "--output", str(path / "model.json"), "--expected-input-sha256", digest]
            self.assertEqual(self.cli.main(freeze_args), 0)
            with self.assertRaises(FileExistsError):
                self.cli.main(freeze_args)
            evaluate_args = ["evaluate", "--input", str(path / "new.json"), "--model", str(path / "model.json"), "--output", str(path / "report.json")]
            with patch.object(original, "fit_ridge", side_effect=AssertionError("fit forbidden")):
                self.assertEqual(self.cli.main(evaluate_args), 0)
            report = json.loads((path / "report.json").read_text())
            self.assertEqual(report["new_label_fits"], 0)
            self.assertEqual(report["coverage"]["valid_paired_records"], 14)
            with self.assertRaises(FileExistsError):
                self.cli.main(evaluate_args)
            model = json.loads((path / "model.json").read_text())
            model["training_identity"]["source_sha256"]["src/entropy_short_fusion.py"] = "c" * 64
            model["payload_sha256"] = frozen.digest(frozen._payload(model))
            (path / "model.json").write_text(json.dumps(model))
            evaluate_args[-1] = str(path / "blocked.json")
            with self.assertRaisesRegex(ValueError, "sources differ"):
                self.cli.main(evaluate_args)
            self.assertFalse((path / "blocked.json").exists())

    def test_freeze_wrong_input_digest_or_pair_count_produces_no_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            source, output = path / "old.json", path / "model.json"
            source.write_text(json.dumps(document("old/")))
            args = ["freeze", "--input", str(source), "--output", str(output), "--expected-input-sha256", "0" * 64]
            with self.assertRaisesRegex(ValueError, "digest"):
                self.cli.main(args)
            args[-1] = self.cli.sha(source)
            with self.assertRaisesRegex(ValueError, "38 valid"):
                self.cli.main(args)
            self.assertFalse(output.exists())

    def test_input_changed_during_fit_is_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            source, output = path / "old.json", path / "model.json"
            old = document("old/")
            old["records"] = old["records"][:38]
            source.write_text(json.dumps(old))
            expected = self.cli.sha(source)
            def mutate_after_fit(doc, identity):
                result = frozen.freeze(doc, identity)
                source.write_text(source.read_text() + "\n")
                return result
            with patch.object(self.cli, "freeze", side_effect=mutate_after_fit):
                with self.assertRaisesRegex(ValueError, "changed during"):
                    self.cli.main(["freeze", "--input", str(source), "--output", str(output), "--expected-input-sha256", expected])
            self.assertFalse(output.exists())

    def test_source_changed_during_fit_is_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            source, output = path / "old.json", path / "model.json"
            old = document("old/")
            old["records"] = old["records"][:38]
            source.write_text(json.dumps(old))
            expected = self.cli.sha(source)
            real_sha = self.cli.sha
            changed = [False]
            def mock_sha(item):
                return "f" * 64 if changed[0] and item.name == "entropy_frozen_fusion.py" else real_sha(item)
            def fit_then_mark_source_changed(doc, identity):
                result = frozen.freeze(doc, identity)
                changed[0] = True
                return result
            with patch.object(self.cli, "sha", side_effect=mock_sha), patch.object(self.cli, "freeze", side_effect=fit_then_mark_source_changed):
                with self.assertRaisesRegex(ValueError, "changed during"):
                    self.cli.main(["freeze", "--input", str(source), "--output", str(output), "--expected-input-sha256", expected])
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
