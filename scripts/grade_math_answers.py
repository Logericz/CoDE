#!/usr/bin/env python3
"""Grade isolated final answers into stdout or a new, exclusive JSON report."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import math_grading


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSONL with id, gold, answer_text; executed rows only")
    parser.add_argument("--planned-count", type=int, required=True, help="all planned requests, including unexecuted ones")
    parser.add_argument("--timeout-seconds", type=float, default=10.0, help="hard per-row process deadline, including imports")
    parser.add_argument("--output", type=Path, help="new JSON report; omitted means stdout; existing files are refused")
    args = parser.parse_args(argv)
    try:
        if args.output is not None:
            if args.output.resolve() == args.input.resolve():
                raise ValueError("output must not be the input file")
            if args.output.exists() or args.output.is_symlink():
                raise ValueError("output already exists; refusing to overwrite")
        input_hash = math_grading.file_sha256(args.input)
        records = math_grading.read_jsonl(args.input)
        math_grading.validate_records(records, args.planned_count)
        manifest = math_grading.build_manifest(args.input, args.timeout_seconds, Path(__file__))
        if manifest["input"]["sha256"] != input_hash:
            raise ValueError("input changed while being read; no report written")
        report = math_grading.grade_records(records, args.planned_count, args.timeout_seconds)
        if math_grading.file_sha256(args.input) != manifest["input"]["sha256"]:
            raise ValueError("input changed during grading; no report written")
        report["manifest"] = manifest
        encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output is None:
            sys.stdout.write(encoded)
        else:
            # Exclusive creation also prevents a race with another report writer.
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(encoded)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"grading failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
