#!/usr/bin/env python3
"""Seal two masked boundary reviews before preparing separate CPU grading input.

本脚本不运行模型、不改原判分。seal 阶段不读取 private mapping/gold；
prepare 只接受已封存且来源未变化的一致裁决。数学比较另用现有固定判分器。
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from math_grading import REASONING_MARKER, extract_final_answer, file_sha256, read_jsonl, validate_records

POLICY_ID = "development-answer-boundary-review-v1"
POLICY_SHA256 = "cb10a8cf353f15cb66a2457c3fa8f7a0b94dc2924166f62a47dc88d08f6931f3"
TAIL = re.compile(
    r"\s*(?:</think>\s*)+(?:\*\*Final Answer\*\*\s*)?"
    r"(?:The final answer is\s*)?(?:\\boxed\s*\{?\s*)?")
# 补充层拒绝所有渠道 token；不改原严格判分器的历史协议。
CONTROL_MARKER = re.compile(REASONING_MARKER.pattern + r"|<\|[^<>\r\n]*\|>", re.IGNORECASE)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def jsonl_bytes(rows):
    return b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
                    for row in rows)


def text_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_sha256(path)}


def verify_identity(item):
    if file_sha256(Path(item["path"])) != item["sha256"]:
        raise ValueError("Sealed input changed: " + item["path"])


def write_new(path, data):
    with Path(path).open("xb") as stream:
        stream.write(data)


def new_directory(path):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise ValueError("Output exists; do not overwrite a sealed analysis")
    path.mkdir(parents=True)
    return path


def bare_boundary_complete(text):
    """Reject obvious truncation; mathematical validity belongs to later grading.

    花括号必须配对；圆/方括号允许区间的混合端点，如 (0, 1]。
    这里只检验边界，不使用标准答案，也不进行符号比较。
    """
    value = text.strip()
    for opening, closing in (("$$", "$$"), ("$", "$"), (r"\[", r"\]"), (r"\(", r"\)")):
        if value.startswith(opening):
            if not value.endswith(closing) or len(value) <= len(opening) + len(closing):
                return False
            value = value[len(opening):-len(closing)].strip()
            break
    if not value or re.search(r"[+\-*/^_=,;:&\\]$", value):
        return False
    stack = []
    for match in re.finditer(r"\\[A-Za-z]+|\\.|[{}()\[\]]", value):
        token = match.group()
        if token in (r"\[", r"\]", r"\(", r"\)"):
            return False  # A second or unmatched math delimiter is ambiguous.
        if token in (r"\{", r"\}"):
            token = token[-1]
        if token in ("{", "(", "["):
            stack.append(token)
        elif token in ("}", ")", "]"):
            if not stack or (token == "}") != (stack.pop() == "{"):
                return False
    return not stack


def policy_selection(text):
    """Executable check of the frozen N1/T1 rules; returns no inferred answers."""
    marker = CONTROL_MARKER.search(text)
    prefix = text if marker is None else text[:marker.start()]
    extraction = extract_final_answer(prefix)
    if extraction["status"] != "extracted":
        return None
    if extraction["extraction"] == "last_complete_boxed":
        start, end = extraction["selected_span"]
    else:
        if not bare_boundary_complete(prefix):
            return None
        start = len(prefix) - len(prefix.lstrip())
        end = len(prefix.rstrip())
    if marker is None:
        return {"rule_id": "N1", "selected_expression_verbatim": text[start:end],
                "selected_expression_char_span": [start, end]}
    if extraction["extraction"] != "last_complete_boxed" or TAIL.fullmatch(text[end:]) is None:
        return None
    return {"rule_id": "T1", "selected_expression_verbatim": text[start:end],
            "selected_expression_char_span": [start, end]}


def validate_review(cases, rows):
    if len(rows) != len(cases) or [r.get("case_id") for r in rows] != [c["case_id"] for c in cases]:
        raise ValueError("Every anonymous case must be reviewed once in source order")
    for case, row in zip(cases, rows):
        text = case["answer_text"]
        if row.get("answer_text_sha256") != text_hash(text):
            raise ValueError("Review answer hash differs")
        if not isinstance(row.get("review_note"), str) or not row["review_note"].strip():
            raise ValueError("Boundary evidence note required")
        if row.get("boundary_decision") == "unresolved":
            if (row.get("rule_id") != "U1" or row.get("selected_expression_verbatim") is not None
                    or row.get("selected_expression_char_span") is not None):
                raise ValueError("Unresolved review must not supply an answer")
            continue
        if row.get("boundary_decision") != "accepted":
            raise ValueError("Unknown review decision")
        selected = policy_selection(text)
        if selected is None or any(row.get(k) != v for k, v in selected.items()):
            raise ValueError("Accepted expression is not the frozen policy's exact substring")
        span = row["selected_expression_char_span"]
        if (any(type(v) is not int for v in span)
                or text[span[0]:span[1]] != row["selected_expression_verbatim"]):
            raise ValueError("Selected Unicode character span differs")


def packet_cases(packet):
    """Only manifest metadata and masked cases are opened here."""
    packet = Path(packet)
    manifest_path = packet / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases_path = packet / "reviewer/cases.jsonl"
    if file_sha256(cases_path) != manifest["outputs"]["reviewer/cases.jsonl"]["sha256"]:
        raise ValueError("Masked cases differ from the original packet")
    cases = read_jsonl(cases_path)
    if len(cases) != manifest["case_count"] or len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Packet count or case identities differ")
    if not cases or any(set(c) != {"case_id", "answer_text", "review"} for c in cases):
        raise ValueError("Expected the original masked case schema")
    return manifest, cases, [identity(manifest_path), identity(cases_path)]


def seal_reviews(packet, policy, review_a, review_b, output):
    if file_sha256(policy) != POLICY_SHA256:
        raise ValueError("Review policy differs from frozen v1")
    if Path(review_a).resolve() == Path(review_b).resolve():
        raise ValueError("Two separate reviewer artifacts required")
    _, cases, inputs = packet_cases(packet)
    inputs += [identity(policy), identity(review_a), identity(review_b),
               identity(__file__), identity(ROOT / "src/math_grading.py")]
    reviews = [read_jsonl(path) for path in (review_a, review_b)]
    for rows in reviews:
        validate_review(cases, rows)
    consensus = []
    compared = ("boundary_decision", "rule_id", "selected_expression_verbatim", "selected_expression_char_span")
    for case, a, b in zip(cases, *reviews):
        agreement = all(a[key] == b[key] for key in compared)
        accepted = agreement and a["boundary_decision"] == "accepted"
        consensus.append({
            "case_id": case["case_id"], "answer_text_sha256": text_hash(case["answer_text"]),
            "reviewer_agreement": agreement, "boundary_decision": "accepted" if accepted else "unresolved",
            "rule_id": a["rule_id"] if accepted else "U1",
            "selected_expression_verbatim": a["selected_expression_verbatim"] if accepted else None,
            "selected_expression_char_span": a["selected_expression_char_span"] if accepted else None,
            "reason": "two_reviews_agree" if accepted else
                      "both_unresolved" if agreement else "reviewer_disagreement"})
    for item in inputs:
        verify_identity(item)
    output = new_directory(output)
    payload = jsonl_bytes(consensus)
    write_new(output / "consensus.jsonl", payload)
    seal = {"policy_id": POLICY_ID, "policy_sha256": POLICY_SHA256, "scope": "posthoc_development_only",
            "source_inputs": inputs, "case_count": len(cases),
            "consensus_sha256": hashlib.sha256(payload).hexdigest(),
            "counts": dict(Counter(r["boundary_decision"] for r in consensus)),
            "rule_counts": dict(Counter(r["rule_id"] for r in consensus)),
            "disagreements": sum(not r["reviewer_agreement"] for r in consensus),
            "gold_or_private_mapping_read_in_seal": False, "human_review_performed": False,
            "review_kind": "two_assistant_contexts_method_and_gold_masked",
            "strict_reports_modified": False}
    write_new(output / "seal.json", encoded(seal))  # Publish the seal last.
    return seal


def load_seal(directory):
    directory = Path(directory)
    seal = json.loads((directory / "seal.json").read_text(encoding="utf-8"))
    if seal["policy_id"] != POLICY_ID or seal["policy_sha256"] != POLICY_SHA256:
        raise ValueError("Unknown seal policy")
    for item in seal["source_inputs"]:
        verify_identity(item)
    if file_sha256(directory / "consensus.jsonl") != seal["consensus_sha256"]:
        raise ValueError("Consensus changed after sealing")
    rows = read_jsonl(directory / "consensus.jsonl")
    if len(rows) != seal["case_count"] or len({r["case_id"] for r in rows}) != len(rows):
        raise ValueError("Sealed consensus identities differ")
    return seal, rows


def prepare_grading(packet, sealed, output):
    seal, consensus = load_seal(sealed)
    manifest, cases, inputs = packet_cases(packet)
    # Reject a different packet even if someone reused anonymous IDs.
    sealed_ids = {item["path"]: item["sha256"] for item in seal["source_inputs"]}
    if any(sealed_ids.get(item["path"]) != item["sha256"] for item in inputs):
        raise ValueError("Packet was not the one used by the sealed reviews")
    # Private identities/gold are opened only after validating the published seal.
    mapping_path = Path(packet) / "private/mapping.jsonl"
    mapping_id = identity(mapping_path)
    if mapping_id["sha256"] != manifest["outputs"]["private/mapping.jsonl"]["sha256"]:
        raise ValueError("Private mapping differs from original packet")
    mapping = read_jsonl(mapping_path)
    if ([r["case_id"] for r in mapping] != [r["case_id"] for r in cases]
            or [r["case_id"] for r in consensus] != [r["case_id"] for r in cases]):
        raise ValueError("Private mapping identities differ")
    inputs += [mapping_id, identity(Path(sealed) / "seal.json"), identity(Path(sealed) / "consensus.jsonl")]
    grading, evidence = [], []
    for case, decision, row in zip(cases, consensus, mapping):
        original = row["original_record"]
        canonical = (json.dumps(original, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":")) + "\n").encode("utf-8")
        if hashlib.sha256(canonical).hexdigest() != row["original_record_canonical_sha256"]:
            raise ValueError("Original record mapping hash differs")
        if (original["answer_text"] != case["answer_text"]
                or row["answer_text_sha256"] != text_hash(case["answer_text"])):
            raise ValueError("Mapping answer is not the reviewed text")
        accepted = decision["boundary_decision"] == "accepted"
        original_boundary = original.get("answer_boundary_confirmed")
        if original_boundary is not True:
            raise ValueError("This policy only reviews runner-confirmed original answer spans")
        selected = decision["selected_expression_verbatim"] if accepted else original["answer_text"]
        grading.append({
            "id": original["id"], "gold": original["gold"], "answer_text": selected,
            "execution_status": original.get("execution_status", "completed"),
            "answer_boundary_confirmed": accepted,
            "boundary_policy": POLICY_ID + "; supplementary assistant-assisted review; sealed before math grading",
            "source_sha256": original["source_sha256"], "review_case_id": case["case_id"],
            "original_answer_text_sha256": decision["answer_text_sha256"],
            "consensus_seal_sha256": file_sha256(Path(sealed) / "seal.json")})
        evidence.append({"id": original["id"], **decision, "original_source_sha256": original["source_sha256"]})
    validate_records(grading, len(cases))
    # The packet's source inputs are hashed without rewriting their old contents.
    inputs += manifest["inputs"]
    for item in inputs:
        verify_identity(item)
    output = new_directory(output)
    payloads = {"grading-input.jsonl": jsonl_bytes(grading), "unmasked-boundaries.jsonl": jsonl_bytes(evidence)}
    for filename, payload in payloads.items():
        write_new(output / filename, payload)
    report = {"scope": "posthoc_development_only", "policy_id": POLICY_ID, "planned_count": len(cases),
              "source_inputs": inputs, "sealed_reviews": identity(Path(sealed) / "seal.json"),
              "outputs": {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()},
              "human_review_performed": False, "strict_reports_modified": False}
    write_new(output / "manifest.json", encoded(report))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    seal = sub.add_parser("seal")
    for arg in ("packet", "policy", "review-a", "review-b", "output"):
        seal.add_argument("--" + arg, type=Path, required=True)
    prepare = sub.add_parser("prepare")
    for arg in ("packet", "sealed", "output"):
        prepare.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.phase == "seal":
        report = seal_reviews(args.packet, args.policy, args.review_a, args.review_b, args.output)
    else:
        report = prepare_grading(args.packet, args.sealed, args.output)
    print(json.dumps({k: v for k, v in report.items() if k in
                      ("policy_id", "case_count", "planned_count", "counts", "rule_counts", "disagreements")},
                     ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
