#!/usr/bin/env python3
"""固定短试答融合消融的CPU入口：封存计划、核验旧证据、一次运行并独占保存。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.entropy_short_features import FEATURE_METADATA
from src.entropy_short_fusion import analyze, METHODS, BOOTSTRAP_SEED

SOURCES = ("src/entropy_short_features.py", "src/entropy_short_fusion.py",
           "src/entropy_value_analysis.py", "scripts/analyze_entropy_short_fusion.py",
           "docs/ENTROPY_SHORT_FUSION.md")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        handle.write("\n")


def verify_preserved(baseline):
    if not isinstance(baseline, dict) or not baseline:
        raise ValueError("Prior evidence manifest must be a nonempty path-to-hash mapping")
    changed = [name for name, digest in baseline.items()
               if not (ROOT / name).is_file() or sha(ROOT / name) != digest]
    if changed:
        raise ValueError("Previously saved evidence changed: " + ", ".join(changed))
    return {"passed": True, "files": len(baseline)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference-analysis", type=Path, required=True)
    parser.add_argument("--prior-evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    document = json.loads(args.input.read_text())
    reference = json.loads(args.reference_analysis.read_text())
    baseline = json.loads(args.prior_evidence.read_text())
    input_paths = (args.input, args.reference_analysis, args.prior_evidence)
    inputs = {str(path.resolve()): sha(path) for path in input_paths}
    expected_input = reference["input_sha256"].get(str(args.input.resolve()))
    if expected_input != sha(args.input):
        raise ValueError("Reference analysis belongs to a different input")
    if reference["source_sha256"]["src/entropy_value_analysis.py"] != sha(ROOT / "src/entropy_value_analysis.py"):
        raise ValueError("Frozen original analyzer has changed")
    preserved = verify_preserved(baseline)
    sources = {name: sha(ROOT / name) for name in SOURCES}
    # 先提交协议与源码，防止把看过结果后的实现伪装成预先固定的设计。
    subprocess.run(["git", "ls-files", "--error-unmatch", *SOURCES], cwd=ROOT,
                   check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *SOURCES], cwd=ROOT, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "short-fusion.json").exists():
        raise FileExistsError("Preserve existing fusion output")
    plan = {
        "role": "post_hoc_cached_short_fusion", "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit, "input_sha256": inputs, "source_sha256": sources,
        "features": FEATURE_METADATA, "methods": METHODS, "outer_folds": 4,
        "fold_seed": 20261010, "ridge_alpha": 1.0, "lambda_error_ms": [60000, 30000, 120000],
        "primary_comparison": "CH_vs_C; lambda=60000; full21_nonended_short",
        "auxiliary_comparison": "CH_vs_H", "minimum_training_questions": 4,
        "bootstrap_replicates": 2000, "bootstrap_seed": BOOTSTRAP_SEED,
        "no_generation": True, "no_parameter_search": True, "prior_evidence": preserved,
    }
    plan_path = args.output_dir / "fusion-plan.json"
    write_new(plan_path, plan)
    plan_hash = sha(plan_path)
    print(json.dumps({"phase": "plan_frozen", "sha256": plan_hash, "git_commit": commit,
                      "prior_evidence_files": preserved["files"]}), flush=True)
    started = time.monotonic()
    report = analyze(document, reference)
    if inputs != {str(path.resolve()): sha(path) for path in input_paths}:
        raise ValueError("Input changed during analysis")
    if sources != {name: sha(ROOT / name) for name in SOURCES}:
        raise ValueError("Source changed during analysis")
    if sha(plan_path) != plan_hash:
        raise ValueError("Frozen fusion plan changed during analysis")
    report.update(input_sha256=inputs, source_sha256=sources, git_commit=commit,
                  elapsed_seconds=time.monotonic() - started,
                  completed_utc=datetime.now(timezone.utc).isoformat(), plan_sha256=plan_hash,
                  prior_evidence_preserved=verify_preserved(baseline))
    write_new(args.output_dir / "short-fusion.json", report)
    print(json.dumps({"phase": "completed", "primary_records": report["primary_records"],
                      "reference_reproduction": report["reference_reproduction"],
                      "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
