#!/usr/bin/env python3
"""Run only the fixed 2x2 cached feature ablation on CPU; no tuning flags."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.entropy_feature_ablation import analyze_ablation

SOURCES = ("src/entropy_compact_features.py", "src/entropy_feature_ablation.py",
           "src/entropy_value_analysis.py", "scripts/analyze_entropy_compact.py",
           "docs/ENTROPY_COMPACT_ABLATION.md")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference-analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    document, reference = json.loads(args.input.read_text()), json.loads(args.reference_analysis.read_text())
    inputs = {str(p.resolve()): sha(p) for p in (args.input, args.reference_analysis)}
    if reference["input_sha256"] != sha(args.input):
        raise ValueError("Reference analysis belongs to a different input")
    if reference["source_sha256"]["src/entropy_value_analysis.py"] != sha(ROOT / "src/entropy_value_analysis.py"):
        raise ValueError("Frozen original analyzer has changed")
    sources = {name: sha(ROOT / name) for name in SOURCES}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "feature-ablation.json").exists():
        raise FileExistsError("Preserve existing ablation output")
    plan = {"role": "post_hoc_cached_feature_ablation", "created_utc": datetime.now(timezone.utc).isoformat(),
            "input_sha256": inputs, "source_sha256": sources, "g_families": ["original", "compact"],
            "f_families": ["original", "compact"], "outer_folds": 4, "inner_folds": 4,
            "fold_seed": 20261010, "ridge_alpha": 1.0, "lambda_error_ms": [60000, 30000, 120000],
            "minimum_training_questions": 4, "bootstrap_replicates": 2000,
            "no_generation": True, "no_parameter_search": True}
    write_new(args.output_dir / "feature-plan.json", plan)
    plan_hash = sha(args.output_dir / "feature-plan.json")
    print(json.dumps({"phase": "plan_frozen", "sha256": plan_hash}), flush=True)
    started = time.monotonic()
    report = analyze_ablation(document, reference)
    if inputs != {str(p.resolve()): sha(p) for p in (args.input, args.reference_analysis)}:
        raise ValueError("Input changed during analysis")
    if sources != {name: sha(ROOT / name) for name in SOURCES}:
        raise ValueError("Source changed during analysis")
    if sha(args.output_dir / "feature-plan.json") != plan_hash:
        raise ValueError("Frozen feature plan changed during analysis")
    report.update(input_sha256=inputs, source_sha256=sources, elapsed_seconds=time.monotonic() - started,
                  completed_utc=datetime.now(timezone.utc).isoformat(), plan_sha256=plan_hash)
    write_new(args.output_dir / "feature-ablation.json", report)
    print(json.dumps({"phase": "completed", "cells": len(report["cells"]),
                      "reference_reproduction": report["reference_reproduction"],
                      "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
