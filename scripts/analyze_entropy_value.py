#!/usr/bin/env python3
"""Analyze bounded paired entropy/value records on CPU; never invoke a GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.entropy_value_analysis import analyze


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("Nonstandard JSON number: " + value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="analysis-input.json from the bounded collector")
    parser.add_argument("--output", type=Path, required=True, help="new JSON result path; existing files are never replaced")
    args = parser.parse_args(argv)
    raw = args.input.read_bytes()
    document = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    report = analyze(document)
    report["input_sha256"] = hashlib.sha256(raw).hexdigest()
    report["source_sha256"] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ("src/entropy_value_analysis.py", "scripts/analyze_entropy_value.py")}
    if args.input.read_bytes() != raw:
        raise ValueError("Input changed during analysis")
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
    primary = report["results"][0]["summary"]
    print(json.dumps({"output": str(args.output.resolve()), "status": report["status"],
                      "coverage": {k: report["coverage"][k] for k in ("planned_questions", "valid_paired_records", "extension_eligible_records")},
                      "primary": primary}, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
