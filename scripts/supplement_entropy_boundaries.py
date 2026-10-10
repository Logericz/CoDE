#!/usr/bin/env python3
"""CPU-only, two-stage boundary supplement; preserve every original artifact.

prepare freezes structural decisions for every answer without consulting gold.
grade verifies that freeze, then applies the unchanged strict math evaluator.
Neither stage loads a language model or changes any prediction feature/cost.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.entropy_answer_boundary import extract_entropy_answer_boundary

PROTOCOL = "entropy-known-finalizer-boundary-supplement-v1"
SOURCES = ("scripts/supplement_entropy_boundaries.py", "src/entropy_answer_boundary.py",
           "src/math_grading.py", "src/entropy_value_analysis.py", "scripts/analyze_entropy_value.py",
           "docs/ENTROPY_BOUNDARY_SUPPLEMENT.md")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def source_hashes():
    return {name: sha(ROOT / name) for name in SOURCES}


def identity(record, action):
    return f"{record['question_id']}:{record['candidate_index']}:{action}"


def structural_decisions(document):
    """Only answer text and its known boundary enter the structural function."""
    decisions, seen = [], set()
    if document.get("schema_version") != "entropy-value-v1":
        raise ValueError("Unexpected input schema")
    for record in document["records"]:
        if record.get("status") != "ok" or set(record["actions"]) != {"E", "T"}:
            raise ValueError("Supplement requires complete saved E/T records")
        for action in ("E", "T"):
            key, result = identity(record, action), record["actions"][action]
            if key in seen:
                raise ValueError("Duplicate answer identity")
            seen.add(key)
            decision = extract_entropy_answer_boundary(result["answer_text"],
                answer_boundary_confirmed=result.get("answer_boundary_confirmed"),
                boundary_policy=result.get("boundary_policy"))
            decisions.append({"id": key, **decision})
    return decisions


def prepare(pilot_dir, output_dir):
    pilot_dir = Path(pilot_dir).resolve()
    paths = {name: pilot_dir / name for name in ("analysis-input.json", "grading.json", "manifest.json")}
    hashes = {name: sha(path) for name, path in paths.items()}
    manifest = read(paths["manifest.json"])
    for name in ("src/math_grading.py", "src/entropy_value_analysis.py", "scripts/analyze_entropy_value.py"):
        if sha(ROOT / name) != manifest["code_sha256"][name]:
            raise ValueError("Original evaluator/analysis source no longer matches: " + name)
    document = read(paths["analysis-input.json"])
    # This step does not read gold or the old correctness labels.
    decisions = structural_decisions(document)
    frozen = {"protocol": PROTOCOL, "role": "post_hoc_exploratory_supplement",
              "created_utc": datetime.now(timezone.utc).isoformat(), "pilot_dir": str(pilot_dir),
              "input_sha256": hashes, "source_sha256": source_hashes(),
              "planned_questions": len(document["question_ids"]), "answer_count": len(decisions),
              "status_counts": dict(Counter(row["status"] for row in decisions)), "decisions": decisions}
    if hashes != {name: sha(path) for name, path in paths.items()}:
        raise ValueError("Input changed during boundary preparation")
    Path(output_dir).mkdir(parents=True, exist_ok=False)
    write_new(Path(output_dir) / "boundary-decisions.json", frozen)
    return {k: frozen[k] for k in ("protocol", "answer_count", "status_counts")}


def strip_labels(document):
    result = deepcopy(document)
    for row in result["records"]:
        for action in row["actions"].values():
            action.pop("error", None)
            action.pop("grading_status", None)
    return result


def apply_grades(document, decisions, original_grades, grader):
    """Align by identity and preserve original text/cost/features in new input."""
    updated, grades, changes = deepcopy(document), [], []
    by_id = {row["id"]: row for row in decisions}
    old = {row["id"]: row for row in original_grades}
    expected = {identity(row, action) for row in document["records"] for action in ("E", "T")}
    if (set(by_id) != expected or set(old) != expected
            or len(by_id) != len(decisions) or len(old) != len(original_grades)):
        raise ValueError("Boundary/grade identities do not match saved answers")
    for record in updated["records"]:
        for action in ("E", "T"):
            key, result = identity(record, action), record["actions"][action]
            boundary, previous = by_id[key], old[key]
            text_hash = hashlib.sha256(result["answer_text"].encode()).hexdigest()
            if text_hash != boundary["raw_text_sha256"] or result["answer_text"] != previous["answer_text"]:
                raise ValueError("Answer identity changed")
            old_error = 0 if previous["grade"] is True else 1 if previous["grade"] is False else None
            if result.get("error") != old_error or result.get("grading_status") != previous["status"]:
                raise ValueError("Original analysis labels disagree with original grading")
            if boundary["status"] == "unchanged":
                # Identical span: preserve the already recorded strict decision,
                # avoiding timeout or evaluator drift on the original 40 labels.
                grade = deepcopy(previous)
                grade["reused_original_grade"] = True
            else:
                grade = grader({"id": key, "gold": previous["gold"],
                    "answer_text": boundary["answer_text"] or "",
                    "execution_status": "completed", "boundary_policy": PROTOCOL,
                    "answer_boundary_confirmed": boundary["status"] != "rejected",
                    "source_sha256": text_hash})
                grade["reused_original_grade"] = False
            if previous.get("reason") == "source_gold_requires_review":
                grade.update(status="needs_review", reason="source_gold_requires_review", grade=None)
            result["error"] = 0 if grade["grade"] is True else 1 if grade["grade"] is False else None
            result["grading_status"] = grade["status"]
            grades.append(grade)
            changes.append({"id": key, "old_status": previous["status"], "old_grade": previous["grade"],
                            "new_status": grade["status"], "new_grade": grade["grade"],
                            "boundary_status": boundary["status"]})
    if strip_labels(updated) != strip_labels(document):
        raise AssertionError("Only grading fields may change")
    return updated, grades, changes


def grade(prepared_dir, output_dir):
    from src.math_grading import _check_dependencies, grade_record
    prepared = Path(prepared_dir) / "boundary-decisions.json"
    frozen, prepared_hash = read(prepared), sha(prepared)
    if frozen["protocol"] != PROTOCOL or frozen["source_sha256"] != source_hashes():
        raise ValueError("Protocol/source changed after boundary freeze")
    pilot = Path(frozen["pilot_dir"])
    hashes = {name: sha(pilot / name) for name in frozen["input_sha256"]}
    if hashes != frozen["input_sha256"]:
        raise ValueError("Frozen original inputs changed")
    document, original = read(pilot / "analysis-input.json"), read(pilot / "grading.json")
    if not original["completed"] or structural_decisions(document) != frozen["decisions"]:
        raise ValueError("Incomplete original grading or altered boundary decisions")
    versions = _check_dependencies()
    if versions != original["dependencies"]:
        raise ValueError("Math evaluator versions differ from original")
    Path(output_dir).mkdir(parents=True, exist_ok=False)
    updated, grades, changes = apply_grades(document, frozen["decisions"], original["results"], grade_record)
    if (hashes != {name: sha(pilot / name) for name in hashes} or sha(prepared) != prepared_hash
            or source_hashes() != frozen["source_sha256"]):
        raise ValueError("Evidence changed while grading")
    summary = {"protocol": PROTOCOL, "role": "post_hoc_exploratory_supplement",
               "original_inputs_unchanged": True, "only_analysis_label_fields_changed": True,
               "prepared_sha256": prepared_hash, "source_sha256": source_hashes(),
               "dependencies": versions, "answer_count": len(grades),
               "status_counts": dict(Counter(g["status"] for g in grades)),
               "reused_original_grades": sum(g["reused_original_grade"] for g in grades),
               "changed_previously_decided_labels": [c for c in changes if c["old_grade"] is not None
                                                     and c["old_grade"] != c["new_grade"]]}
    write_new(Path(output_dir) / "grading.json", {**summary, "completed": True, "results": grades, "changes": changes})
    write_new(Path(output_dir) / "analysis-input.json", updated)
    write_new(Path(output_dir) / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    pre = sub.add_parser("prepare")
    pre.add_argument("--pilot-dir", type=Path, required=True)
    pre.add_argument("--output-dir", type=Path, required=True)
    score = sub.add_parser("grade")
    score.add_argument("--prepared-dir", type=Path, required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare(args.pilot_dir, args.output_dir) if args.stage == "prepare" else grade(args.prepared_dir, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
