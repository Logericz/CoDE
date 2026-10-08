#!/usr/bin/env python3
"""Decode a saved generation call using a local tokenizer; never run inference.

This standalone inspector does not import or modify the frozen diagnostic runner.
Reports are JSON on stdout. Invalid records produce JSON on stderr and exit 1.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

TOOL_VERSION = "1.0.0"
SCHEMA_VERSION = 1
THINK_END = "</think>"
TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "vocab.json", "vocab.txt", "merges.txt",
    "tokenizer.model", "spiece.model", "config.json", "generation_config.json",
    "chat_template.jinja",
)


class InspectionError(ValueError):
    """A saved input cannot be inspected without guessing or changing it."""


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reject_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise InspectionError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def reject_nonfinite(value):
    raise InspectionError(f"Non-finite JSON value: {value}")


def error_source(path):
    path = Path(path).resolve()
    result = {"path": str(path)}
    try:
        result["sha256"] = file_sha256(path)
    except OSError as exc:
        result["sha256"] = None
        result["read_error"] = type(exc).__name__
    return result


def read_object(path):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates,
                           parse_constant=reject_nonfinite)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise InspectionError(f"Invalid UTF-8 JSON in {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise InspectionError(f"{path.name} must contain a JSON object")
    return value, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def token_sequence(call, name, *, allow_empty=False):
    batch = call.get(name)
    if not isinstance(batch, list) or len(batch) != 1 or not isinstance(batch[0], list):
        raise InspectionError(f"{name} must contain exactly one token-ID sequence (batch=1)")
    ids = batch[0]
    if not ids and not allow_empty:
        raise InspectionError(f"{name} cannot be empty")
    if any(type(token) is not int or token < 0 for token in ids):
        raise InspectionError(f"{name} must contain nonnegative integer token IDs")
    return ids


def select_call(record):
    calls = record.get("generation_calls")
    if not isinstance(calls, list) or any(not isinstance(call, dict) for call in calls):
        raise InspectionError("generation_calls must be a list of objects")
    if any(call.get("status") not in ("ok", "failed") for call in calls):
        raise InspectionError("Every generation call must have status 'ok' or 'failed'")
    successful = [i for i, call in enumerate(calls) if call["status"] == "ok"]
    if not successful:
        raise InspectionError("No successful generation call is saved; no finalization to decode")
    index = successful[-1]
    call = calls[index]
    inputs = token_sequence(call, "input_token_ids")
    outputs = token_sequence(call, "output_token_ids")
    generated = token_sequence(call, "generated_token_ids", allow_empty=True)
    if outputs[:len(inputs)] != inputs:
        raise InspectionError("output_token_ids does not preserve the complete input_token_ids prefix")
    if outputs[len(inputs):] != generated:
        raise InspectionError("generated_token_ids does not equal the output suffix after the input prefix")
    budget = call.get("max_new_tokens")
    if budget is not None and (type(budget) is not int or budget <= 0):
        raise InspectionError("max_new_tokens must be a positive integer or null")
    if budget is not None and len(generated) > budget:
        raise InspectionError("Generated token count exceeds the recorded max_new_tokens")
    return index, call, inputs, outputs, generated


def local_tokenizer_path(environment, environment_path, override=None):
    if override is not None:
        path = Path(override).expanduser()
    else:
        model = environment.get("model")
        snapshot = model.get("snapshot") if isinstance(model, dict) else None
        if not isinstance(snapshot, str) or not snapshot.strip():
            raise InspectionError("environment.model.snapshot is missing; pass --tokenizer-path to an existing local directory")
        path = Path(snapshot).expanduser()
        if not path.is_absolute():
            path = Path(environment_path).resolve().parent / path
    if not path.is_dir():
        raise InspectionError(f"Tokenizer directory is not available locally: {path}; pass --tokenizer-path after moving records")
    return path.resolve(strict=True)


def load_local_tokenizer(path):
    # Set offline mode before importing Transformers, including its Hub helpers.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(path), local_files_only=True,
                                         trust_remote_code=False)


def thinking_boundaries(text):
    boundaries, cursor = [], 0
    while True:
        start = text.find(THINK_END, cursor)
        if start < 0:
            break
        cursor = start + len(THINK_END)
        boundaries.append({"occurrence": len(boundaries) + 1,
                           "start_character": start, "end_character": cursor})
    parts = text.split(THINK_END)
    return {"marker": THINK_END, "count": len(boundaries),
            "offset_unit": "zero-based Python Unicode character index; end exclusive",
            "boundaries": boundaries, "split_parts": parts,
            "text_before_last_marker": text.rsplit(THINK_END, 1)[0] if boundaries else None,
            "text_after_last_marker": parts[-1] if boundaries else None}


def decode_both(tokenizer, ids):
    result = {}
    for label, skip in (("special_tokens_preserved", False), ("special_tokens_removed", True)):
        text = tokenizer.decode(ids, skip_special_tokens=skip, clean_up_tokenization_spaces=False)
        if not isinstance(text, str):
            raise InspectionError("Tokenizer.decode did not return text")
        result[label] = {"text": text, "thinking_end": thinking_boundaries(text)}
    return result


def inspect_saved(stage_path, environment_path=None, tokenizer_path=None, *, tokenizer=None):
    stage_path = Path(stage_path)
    environment_path = Path(environment_path) if environment_path else stage_path.parent / "environment.json"
    record, stage_source = read_object(stage_path)
    environment, environment_source = read_object(environment_path)
    index, call, inputs, outputs, generated = select_call(record)
    method_output = record.get("method_output")
    response = method_output.get("response") if isinstance(method_output, dict) else None
    if not isinstance(response, str):
        raise InspectionError("method_output.response must be saved as text")
    path = local_tokenizer_path(environment, environment_path, tokenizer_path)
    tokenizer = tokenizer if tokenizer is not None else load_local_tokenizer(path)
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None and (type(eos) is not int or eos < 0):
        raise InspectionError("Tokenizer eos_token_id must be a nonnegative integer or null")
    try:
        transformers_version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        transformers_version = None
    budget = call.get("max_new_tokens")
    return {
        "schema_version": SCHEMA_VERSION, "tool_version": TOOL_VERSION, "status": "inspected",
        "scope": "saved_generation_evidence_only_no_inference_or_grading",
        "sources": {"stage": stage_source, "environment": environment_source,
                    "inspector": {"path": str(Path(__file__).resolve()), "sha256": file_sha256(__file__)}},
        "tokenizer": {"local_path": str(path), "path_overridden": tokenizer_path is not None,
                      "local_files_only": True, "trust_remote_code": False,
                      "implementation_class": type(tokenizer).__name__,
                      "transformers_version": transformers_version,
                      "files_sha256": {name: file_sha256(path / name) for name in TOKENIZER_FILES if (path / name).is_file()},
                      "eos_token_id": eos},
        "selected_call": {"list_index": index, "recorded_call_index": call.get("call_index"),
                          "stage": call.get("stage", record.get("stage")),
                          "selection_rule": "last successful generation call",
                          "later_failed_calls": len(record["generation_calls"]) - index - 1,
                          "do_sample": call.get("do_sample"), "max_new_tokens": budget},
        "integrity": {"output_preserves_full_input_prefix": True,
                      "generated_equals_output_suffix": True},
        "length_and_eos_facts": {
            "input_token_count": len(inputs), "output_token_count": len(outputs),
            "generated_token_count": len(generated),
            "recorded_budget_reached": len(generated) == budget if budget is not None else None,
            "generated_tokenizer_eos_positions": [i for i, token in enumerate(generated) if eos is not None and token == eos],
            "last_generated_token_is_tokenizer_eos": generated[-1] == eos if generated and eos is not None else None,
        },
        "token_ids": {"input": inputs, "output": outputs, "generated": generated},
        "decoded": {"input": decode_both(tokenizer, inputs), "output": decode_both(tokenizer, outputs),
                    "generated": decode_both(tokenizer, generated)},
        "saved_response": {"text": response, "thinking_end": thinking_boundaries(response)},
        "notes": [
            "No model weights are loaded, no generation is run, and no input record is written.",
            "Budget exhaustion and tokenizer EOS presence are observations, not correctness or truncation verdicts.",
            "Tokenizer EOS alone does not establish the effective generation stopping configuration.",
            "The last successful call can precede failed calls; it is not automatically a completed final answer.",
            "All </think> boundaries are shown; the last-marker tail can omit earlier answer text.",
            "This checks saved token consistency, not the experiment-wide manifest identity or answer grade.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage_json", type=Path, help="Saved stage record, for example items/q001/codestop.json")
    parser.add_argument("--environment", type=Path, help="Environment JSON (default: beside stage record)")
    parser.add_argument("--tokenizer-path", type=Path, help="Existing local tokenizer directory; overrides saved snapshot path")
    args = parser.parse_args(argv)
    try:
        report = inspect_saved(args.stage_json, args.environment, args.tokenizer_path)
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    except Exception as exc:
        print(json.dumps({"schema_version": SCHEMA_VERSION, "tool_version": TOOL_VERSION,
                          "status": "error", "error_type": type(exc).__name__, "message": str(exc),
                          "sources": {"stage": error_source(args.stage_json),
                                      "environment": error_source(args.environment or args.stage_json.parent / "environment.json"),
                                      "inspector": error_source(__file__)}},
                         ensure_ascii=False, allow_nan=False), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
