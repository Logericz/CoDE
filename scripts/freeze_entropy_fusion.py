#!/usr/bin/env python3
"""Freeze old 38-pair fusion models or evaluate disjoint new questions, CPU only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.entropy_frozen_fusion import freeze, evaluate, validate_artifact

SOURCES = ("src/entropy_frozen_fusion.py", "src/entropy_short_features.py", "src/entropy_short_fusion.py",
           "src/entropy_value_analysis.py", "scripts/freeze_entropy_fusion.py")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    freezing = sub.add_parser("freeze", help="Fit the locked old 38 valid pairs once")
    freezing.add_argument("--expected-input-sha256", required=True)
    evaluating = sub.add_parser("evaluate", help="Apply frozen models; never fit new labels")
    evaluating.add_argument("--model", type=Path, required=True)
    for child in (freezing, evaluating):
        child.add_argument("--input", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Preserve existing output: " + str(args.output))
    data = args.input.read_bytes()
    input_hash = hashlib.sha256(data).hexdigest()
    sources = {name: sha(ROOT / name) for name in SOURCES}
    document = json.loads(data)
    model_hash = None
    if args.action == "freeze":
        if input_hash != args.expected_input_sha256:
            raise ValueError("Training input does not match the expected frozen digest")
        result = freeze(document, {"input_sha256": input_hash, "source_sha256": sources})
        if result["training"]["valid_records"] != 38:
            raise ValueError("This protocol freezes exactly 38 valid old pairs")
    else:
        model_bytes = args.model.read_bytes()
        model_hash = hashlib.sha256(model_bytes).hexdigest()
        artifact = validate_artifact(json.loads(model_bytes))
        if artifact["training_identity"]["source_sha256"] != sources:
            raise ValueError("Current sources differ from the frozen model sources")
        result = evaluate(document, artifact)
        result.update(input_sha256=input_hash, model_file_sha256=model_hash, source_sha256=sources,
                      completed_utc=datetime.now(timezone.utc).isoformat())
    if sha(args.input) != input_hash or sources != {name: sha(ROOT / name) for name in SOURCES}:
        raise ValueError("Input or source changed during execution")
    if model_hash is not None and sha(args.model) != model_hash:
        raise ValueError("Frozen model file changed during evaluation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"action": args.action, "output": str(args.output), "sha256": sha(args.output),
                      "new_label_fits": 0 if args.action == "evaluate" else None}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
