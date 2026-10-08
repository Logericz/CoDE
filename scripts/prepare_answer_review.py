#!/usr/bin/env python3
"""Prepare pending, method/gold-masked review copies; never grade or adjudicate.

All executed input rows are included. The original inputs and strict reports
remain unchanged. Only reviewer/ is suitable for a reviewer: private/ and the
root manifest contain the identities needed for later unmasking. A manifest is
published last; its absence means an incomplete packet, which is never resumed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from math_grading import file_sha256, read_jsonl, validate_records

PACKET_VERSION = "pending-answer-boundary-review-v1"
SCOPE = "development-exposed"
INSTRUCTIONS = """# 待复核答案资料

本目录只供答案边界复核，所有条目尚未裁决。只向复核者提供 reviewer/；
不要同时提供根目录的 manifest.json 或 private/。

- 每个 case_id 是随机标识，顺序已随机打乱。answer_text 是原答案文本的完整副本，
  重复 think 标记、未闭合内容、空文本和尾部说明均未删除或修补。
- 不根据正确答案选择表达式，不改写 answer_text，也不从其他记录寻找答案。
  本包不会自动接受重复标记，不替代另行冻结的统一审查准则。
- review.status 初始为 pending，boundary_decision、逐字表达式、字符区间和说明均为空。
  如依据已冻结准则能够锁定表达式，逐字抄录 selected_expression_verbatim，并填写
  selected_expression_char_span=[start,end]（半开区间）。位置从零开始，按解码后的 Unicode 字符计数，
  end 不包含在区间中；须满足 answer_text[start:end] 与逐字表达式完全相同。
  不确定时不要补造答案或填写数学对错。
- 裁决应另存新文件，保留原始待审包。先封存边界裁决，之后再解盲并独立进行数学比较；
  原严格判分不被替换，也不自动合并到任何旧报告。

盲审限制：本包范围是 development-exposed。匿名标识和去除方法/gold 字段不能消除
文本本身的辨识线索或审查者已有知识。准备本包的助手可能已经见过开发样例及其结果，
不能把自己的再次判断声称为真正盲审。资料生成不代表人工审查已经完成。
"""


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def verify_inputs(inputs):
    for item in inputs:
        if file_sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Input changed during review preparation: {item['path']}")


def load_inputs(paths):
    inputs, entries = [], []
    for supplied in paths:
        path = Path(supplied).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError("Every --input must be an existing JSONL file")
        digest = file_sha256(path)
        records = read_jsonl(path)
        identity = {"path": str(path), "sha256": digest, "row_count": len(records)}
        verify_inputs([identity])
        inputs.append(identity)
        for number, record in enumerate(records, 1):
            if isinstance(record, dict) and isinstance(record.get("answer_text"), str):
                answer_digest = digest_bytes(record["answer_text"].encode("utf-8"))
                if "answer_text_sha256" in record and record["answer_text_sha256"] != answer_digest:
                    raise ValueError("Declared answer_text_sha256 does not match the complete answer text")
            entries.append({"input_path": str(path), "input_sha256": digest,
                            "input_row_number": number, "original_record": record})
    # The existing validator is dependency-free and checks IDs across all files.
    # Failed executed requests remain included; no status/grade-based selection.
    validate_records([entry["original_record"] for entry in entries], len(entries))
    if not entries:
        raise ValueError("At least one executed input row is required")
    verify_inputs(inputs)
    return inputs, entries


def packet_payloads(entries):
    shuffled = list(entries)
    random.SystemRandom().shuffle(shuffled)
    reviewer, private, seen = [], [], set()
    for entry in shuffled:
        case_id = uuid.uuid4().hex
        if case_id in seen:
            raise RuntimeError("Duplicate random review identity; no packet published")
        seen.add(case_id)
        record = entry["original_record"]
        reviewer.append({"case_id": case_id, "answer_text": record["answer_text"],
                         "review": {"status": "pending", "boundary_decision": None,
                                    "selected_expression_verbatim": None,
                                    "selected_expression_char_span": None, "review_note": None}})
        private.append({"case_id": case_id, **entry,
                        "original_record_canonical_sha256": digest_bytes(encoded(record)),
                        "answer_text_sha256": digest_bytes(record["answer_text"].encode("utf-8"))})
    return {"reviewer/cases.jsonl": b"".join(encoded(item) for item in reviewer),
            "reviewer/README.md": INSTRUCTIONS.encode("utf-8"),
            "private/mapping.jsonl": b"".join(encoded(item) for item in private)}


def write_exclusive(path, payload):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def prepare_review(input_paths, output_dir, *, scope):
    if scope != SCOPE:
        raise ValueError("Only the explicit development-exposed scope is supported")
    if not input_paths:
        raise ValueError("At least one --input is required")
    output = Path(output_dir).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("Output directory already exists; refusing to overwrite or resume")
    inputs, entries = load_inputs(input_paths)
    payloads = packet_payloads(entries)
    manifest = {"schema_version": 1, "packet_version": PACKET_VERSION, "scope": scope,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "prepared_all_pending", "case_count": len(entries),
                "selection_policy": "all executed input rows; no status, gold or grade filtering",
                "reviewer_fields": ["case_id", "answer_text", "review"],
                "order_policy": "SystemRandom shuffle; independent UUID4 case IDs",
                "inputs": inputs,
                "outputs": {name: {"sha256": digest_bytes(value), "bytes": len(value)}
                            for name, value in payloads.items()},
                "preparation_source_sha256": file_sha256(Path(__file__)),
                "original_record_hash_encoding": "UTF-8 sorted-key compact JSON, ensure_ascii=False, trailing newline",
                "all_reviews_pending": True, "adjudication_performed": False,
                "math_grading_performed": False, "strict_reports_modified": False,
                "input_hashes_verified_before_and_after_preparation": True,
                "share_only": "reviewer/; root manifest and private/ contain unmasked identities",
                "provenance_limit": "Input bytes and copied answer text verified; source_sha256 inside original rows remains an assertion unless separately audited.",
                "blinding_limitations": [
                    "These are already exposed development samples; no claim of confirmatory evaluation.",
                    "The preparing assistant may know the outcomes and cannot claim to be a blinded reviewer.",
                    "Original answer text remains intact and may identify a question or method to a knowledgeable reviewer.",
                    "Opaque IDs and shuffled order do not establish actual reviewer independence."]}
    verify_inputs(inputs)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Stage complete files, then reserve a new output directory atomically.
    # Hard links publish files without replacing any existing destination.
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-prepare-", dir=output.parent) as temporary:
        stage = Path(temporary)
        for name, payload in payloads.items():
            destination = stage / name
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            write_exclusive(destination, payload)
        write_exclusive(stage / "manifest.json", encoded(manifest))
        verify_inputs(inputs)
        output.mkdir(mode=0o700, exist_ok=False)
        for name in payloads:
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.link(stage / name, destination)
        for name, identity in manifest["outputs"].items():
            destination = output / name
            if (file_sha256(destination) != identity["sha256"]
                    or destination.stat().st_size != identity["bytes"]):
                raise ValueError(f"Published output changed during review preparation: {name}")
        verify_inputs(inputs)
        os.link(stage / "manifest.json", output / "manifest.json")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True,
                        help="Executed-answer JSONL; repeatable; every row is included")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory only; never resumed")
    parser.add_argument("--scope", choices=(SCOPE,), required=True)
    args = parser.parse_args(argv)
    try:
        manifest = prepare_review(args.input, args.output_dir, scope=args.scope)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"Review preparation failed: {error}\nNo review or grading was performed. A directory without manifest.json is incomplete; do not reuse it.\n")
    print(json.dumps({"status": manifest["status"], "case_count": manifest["case_count"],
                      "all_reviews_pending": True, "output_dir": str(args.output_dir.absolute())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
