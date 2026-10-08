"""Independent, answer-only Math-Verify grading; no inference record is modified.

Every symbolic comparison runs in a fresh process. Internal Math-Verify timers
are disabled so their caught timeouts cannot silently become negative grades.
"""
from __future__ import annotations

import contextlib
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time


PROTOCOL_VERSION = "math-answer-only-v1"
REQUIRED_VERSIONS = {
    "math-verify": "0.9.0", "antlr4-python3-runtime": "4.13.2",
    "latex2sympy2_extended": "1.11.0", "sympy": "1.14.0", "mpmath": "1.3.0",
}
NORMALIZATION = {
    "basic_latex": True, "units": False, "malformed_operators": False,
    "nits": False, "boxed": "none", "equations": False,
}
VERIFY_PARAMETERS = {
    "float_rounding": 6, "numeric_precision": 15, "strict": True,
    "allow_set_relation_comp": False, "timeout_seconds": None,
    "raise_on_error": True,
}
# An answer-only boundary must be supplied by the caller, never inferred here.
REASONING_MARKER = re.compile(
    r"<\s*/?\s*(?:think|analysis|reasoning)\b|"
    r"<\|(?:analysis|im_start|im_end|start|end|channel|message)\|>|"
    r"\[/?(?:think|analysis|reasoning)\]", re.IGNORECASE,
)
BOX_START = re.compile(r"(?<!\\)\\boxed\s*\{")
SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
EXECUTION_STATUSES = {"completed", "failed"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _escaped(text: str, position: int) -> bool:
    backslashes = 0
    position -= 1
    while position >= 0 and text[position] == "\\":
        backslashes += 1
        position -= 1
    return backslashes % 2 == 1


def extract_final_answer(answer_text: str) -> dict:
    """Select the last complete balanced box, without scanning a thinking span.

    A complete box before an unclosed trailing box remains the selected answer,
    as required by the protocol. The malformed tail is retained as evidence.
    Without boxes, only a single bare mathematical expression is accepted.
    """
    result = {"selected_answer": None, "extraction": None,
              "complete_box_count": 0, "unclosed_box_positions": [],
              "has_unclosed_tail": False}
    if REASONING_MARKER.search(answer_text):
        return {**result, "status": "needs_review", "reason": "reasoning_boundary_not_isolated"}
    if not answer_text.strip():
        return {**result, "status": "empty_answer", "reason": "empty_final_answer_span"}
    complete = []
    nested_positions = []
    for match in BOX_START.finditer(answer_text):
        depth, end = 1, None
        for index in range(match.end(), len(answer_text)):
            char = answer_text[index]
            if char in "{}" and not _escaped(answer_text, index):
                depth += 1 if char == "{" else -1
                if depth == 0:
                    end = index
                    break
        if end is None:
            result["unclosed_box_positions"].append(match.start())
        else:
            complete.append((match.start(), end + 1, answer_text[match.end():end]))
    # A box nested inside another box is mathematical content, not another
    # candidate final answer. An unclosed enclosing box also blocks promotion
    # of an inner box to a top-level answer.
    top_level = []
    for box in complete:
        start, end, _ = box
        if (any(outer_start < start < outer_end for outer_start, outer_end, _ in complete)
                or any(outer_start < start for outer_start in result["unclosed_box_positions"])):
            nested_positions.append(start)
        else:
            top_level.append(box)
    result["complete_box_count"] = len(top_level)
    result["nested_box_positions"] = nested_positions
    if top_level:
        start, end, answer = top_level[-1]
        result.update(selected_answer=answer.strip(), extraction="last_complete_boxed",
                      selected_span=[start, end], has_unclosed_tail=any(
                          position >= end for position in result["unclosed_box_positions"]))
        if any(start < position < end for position in nested_positions):
            return {**result, "status": "needs_review", "reason": "nested_boxed_answer_not_supported"}
    elif result["unclosed_box_positions"] or re.search(r"\\boxed\b", answer_text):
        return {**result, "status": "malformed_box", "reason": "no_complete_boxed_answer"}
    else:
        answer = answer_text.strip()
        for opening, closing in (("$$", "$$"), ("$", "$"), (r"\[", r"\]"), (r"\(", r"\)")):
            if answer.startswith(opening) and answer.endswith(closing) and len(answer) >= len(opening) + len(closing):
                answer = answer[len(opening):-len(closing)].strip()
                break
        # Do not let an extraction regex select a convenient number from prose.
        # Bare multi-letter prose must be reviewed; boxed expressions may contain
        # mathematical commands, which the pinned parser will validate.
        without_commands = re.sub(r"\\[A-Za-z]+", "", answer)
        if re.search(r"[A-Za-z]{2,}", without_commands) or any(c in answer for c in ("\n", "$", "`")):
            return {**result, "status": "needs_review", "reason": "unboxed_answer_not_single_math_expression"}
        result.update(selected_answer=answer, extraction="bare_math_expression")
    if not result["selected_answer"]:
        return {**result, "status": "empty_answer", "reason": "selected_box_is_empty"}
    return {**result, "status": "extracted", "reason": None}


def validate_records(records: list[dict], planned_count: int) -> None:
    if isinstance(planned_count, bool) or not isinstance(planned_count, int) or planned_count < len(records):
        raise ValueError("planned_count must be an integer >= executed input row count")
    seen = set()
    for number, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise ValueError(f"row {number}: expected a JSON object")
        identifier = record.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"row {number}: id must be a nonempty string")
        if identifier in seen:
            raise ValueError(f"row {number}: duplicate id {identifier!r}")
        seen.add(identifier)
        if not isinstance(record.get("gold"), str) or not record["gold"].strip():
            raise ValueError(f"row {number}: gold must be a nonempty clean LaTeX answer")
        if not isinstance(record.get("answer_text"), str):
            raise ValueError(f"row {number}: answer_text must be an explicitly isolated final-answer string")
        if record.get("execution_status", "completed") not in EXECUTION_STATUSES:
            raise ValueError(f"row {number}: execution_status must be completed or failed; omit unexecuted rows")
        if "source_sha256" in record and (not isinstance(record["source_sha256"], str) or not SHA256.fullmatch(record["source_sha256"])):
            raise ValueError(f"row {number}: source_sha256 must be 64 hexadecimal characters")
        if "boundary_policy" in record and (not isinstance(record["boundary_policy"], str) or not record["boundary_policy"].strip()):
            raise ValueError(f"row {number}: boundary_policy must be a nonempty description if supplied")


def read_jsonl(path: Path) -> list[dict]:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"nonfinite JSON number {value}")

    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"line {number}: blank lines are not executed records")
            try:
                records.append(json.loads(line, object_pairs_hook=unique_object, parse_constant=reject_constant))
            except ValueError as error:
                raise ValueError(f"line {number}: invalid JSON: {error}") from error
    return records


def _check_dependencies() -> dict:
    actual = {}
    for name, expected in REQUIRED_VERSIONS.items():
        try:
            actual[name] = metadata.version(name)
        except metadata.PackageNotFoundError as error:
            raise RuntimeError(f"Missing {name}; install math-verify[antlr4_13_2]==0.9.0") from error
        if actual[name] != expected:
            raise RuntimeError(f"Expected {name}=={expected}, found {actual[name]}")
    return actual


def build_manifest(input_path: Path, timeout_seconds: float, cli_path: Path | None = None) -> dict:
    _check_dependencies()
    package_root = Path(metadata.distribution("math-verify").locate_file("math_verify"))
    sources = [Path(__file__).resolve(), *(package_root / name for name in ("parser.py", "grader.py", "utils.py"))]
    if cli_path is not None:
        sources.append(Path(cli_path).resolve())
    return {
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {"path": str(Path(input_path).resolve()), "sha256": file_sha256(input_path)},
        "python": {"version": sys.version, "executable": sys.executable},
        "required_versions": REQUIRED_VERSIONS,
        "installed_distributions": dict(sorted((dist.metadata["Name"], dist.version)
                                               for dist in metadata.distributions() if dist.metadata["Name"])),
        "source_sha256": {str(path): file_sha256(path) for path in sources},
        "extraction": {"input_contract": "caller-isolated final-answer span only",
                       "rule": "last complete balanced boxed; record unclosed tail; otherwise single bare math expression",
                       "reasoning_marker_pattern": REASONING_MARKER.pattern,
                       "latex_only": True, "fallback_mode": "no_fallback",
                       "extraction_mode": "first_match", "boxed_match_priority": -1,
                       "try_extract_without_anchor": True, "normalization": NORMALIZATION,
                       "parsing_timeout": None, "raise_on_error": True},
        "comparison": {"argument_order": "verify(gold, prediction)", **VERIFY_PARAMETERS},
        "timeout": {"seconds": timeout_seconds, "scope": "fresh child process, including import, parse and verify",
                    "action": "kill child and report timeout; never convert timeout to incorrect"},
        "summary_rule": "confirmed correct / all executed rows; pending rows are reported separately",
        "limitations": ["source_sha256 is a caller-supplied provenance assertion, not proof of isolation",
                        "Math-Verify equivalence under frozen parameters is not a mathematical proof",
                        "uncertain outcomes require method-blinded human review in a separate artifact"],
    }


def _parse_isolated(value: str, parse, config) -> tuple[list | None, str | None]:
    # Reject delimiter injection: extraction must cover this one selected value.
    if REASONING_MARKER.search(value) or "$" in value or r"\boxed" in value:
        return None, "not_clean_isolated_math"
    try:
        parsed = parse("$" + value.strip() + "$", extraction_config=[config],
                       fallback_mode="no_fallback", extraction_mode="first_match",
                       parsing_timeout=None, raise_on_error=True)
    except Exception as error:
        if "timeout" in type(error).__name__.lower():
            raise
        return None, "parse_exception:" + type(error).__name__
    if not parsed:
        return None, "no_symbolic_parse"
    from sympy import Basic, MatrixBase
    if len(parsed) != 1 or any(not isinstance(item, (Basic, MatrixBase)) for item in parsed):
        return None, "unsupported_symbolic_type_or_multiple_extractions"
    return parsed, None


def _compare_payload(payload: dict) -> dict:
    _check_dependencies()
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import LatexExtractionConfig, parse, verify
    config = LatexExtractionConfig(try_extract_without_anchor=True, boxed_match_priority=-1,
                                   normalization_config=NormalizationConfig(**NORMALIZATION))
    gold, reason = _parse_isolated(payload["gold"], parse, config)
    if reason:
        return {"status": "gold_parse_failure", "reason": reason, "grade": None}
    prediction, reason = _parse_isolated(payload["prediction"], parse, config)
    if reason:
        status = "unsupported_type" if reason.startswith("unsupported") else "prediction_parse_failure"
        return {"status": status, "reason": reason, "grade": None}
    result = verify(gold, prediction, **VERIFY_PARAMETERS)
    if not isinstance(result, bool):
        return {"status": "evaluator_error", "reason": "non_boolean_verifier_result", "grade": None}
    return {"status": "correct" if result else "incorrect", "reason": None, "grade": result,
            "parsed_gold_type": type(gold[0]).__name__, "parsed_prediction_type": type(prediction[0]).__name__}


def _run_worker(payload: dict, timeout_seconds: float) -> dict:
    try:
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker"],
                                   input=json.dumps(payload), capture_output=True, text=True,
                                   timeout=timeout_seconds, check=False)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "reason": "outer_process_deadline", "grade": None}
    except OSError as error:
        return {"status": "evaluator_error", "reason": type(error).__name__, "grade": None}
    if completed.returncode:
        return {"status": "evaluator_error", "reason": "worker_exit", "worker_returncode": completed.returncode,
                "grade": None}
    try:
        result = json.loads(completed.stdout)
        if not isinstance(result, dict) or result.get("status") not in {
            "correct", "incorrect", "gold_parse_failure", "prediction_parse_failure",
            "unsupported_type", "evaluator_error", "timeout", "dependency_error",
        } or "grade" not in result:
            raise ValueError("invalid worker schema")
        if result["grade"] is not ({"correct": True, "incorrect": False}.get(result["status"])):
            raise ValueError("inconsistent worker grade")
        return result
    except (ValueError, TypeError):
        return {"status": "evaluator_error", "reason": "invalid_worker_response", "grade": None}


def grade_record(record: dict, timeout_seconds: float = 10.0) -> dict:
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    started = time.monotonic()
    result = {"id": record["id"], "execution_status": record.get("execution_status", "completed"),
              "gold": record["gold"], "answer_text": record["answer_text"],
              "answer_text_sha256": hashlib.sha256(record["answer_text"].encode()).hexdigest(),
              "source_sha256": record.get("source_sha256"),
              "boundary_policy": record.get("boundary_policy", "unknown"),
              "provenance_status": "caller_asserted_not_independently_verified" if record.get("source_sha256") else "unknown"}
    if "stop_reason" in record:
        result["stop_reason"] = record["stop_reason"]
    if result["execution_status"] == "failed":
        result.update(status="run_failure", reason="caller_reported_execution_failure", grade=None)
    else:
        extraction = extract_final_answer(record["answer_text"])
        result.update(extraction)
        if extraction["status"] == "extracted":
            result.update(_run_worker({"gold": record["gold"], "prediction": extraction["selected_answer"]},
                                      timeout_seconds))
        else:
            result["grade"] = None
    result["review_required"] = result["grade"] is None and result["status"] != "run_failure"
    result["grading_elapsed_seconds"] = time.monotonic() - started
    return result


def summarize(results: list[dict], planned_count: int) -> dict:
    if planned_count < len(results):
        raise ValueError("planned_count is smaller than executed rows")
    executed = len(results)
    correct = sum(item["status"] == "correct" for item in results)
    unresolved = sum(item["review_required"] for item in results)
    return {"planned_count": planned_count, "executed_count": executed,
            "unexecuted_count": planned_count - executed,
            "matrix_execution_complete": planned_count == executed,
            "confirmed_correct_count": correct,
            "confirmed_incorrect_count": sum(item["status"] == "incorrect" for item in results),
            "run_failure_count": sum(item["status"] == "run_failure" for item in results),
            "unresolved_count": unresolved,
            "unclosed_tail_count": sum(item.get("has_unclosed_tail", False) for item in results),
            "unknown_boundary_policy_count": sum(item.get("boundary_policy", "unknown") == "unknown" for item in results),
            "accuracy_confirmed_over_executed": correct / executed if executed else None,
            "accuracy_upper_if_all_unresolved_correct": (correct + unresolved) / executed if executed else None,
            "status_counts": dict(sorted(Counter(item["status"] for item in results).items())),
            "denominator_note": "all executed input rows; unexecuted rows are not experimental results"}


def grade_records(records: list[dict], planned_count: int, timeout_seconds: float = 10.0) -> dict:
    validate_records(records, planned_count)
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    results = [grade_record(record, timeout_seconds) for record in records]
    return {"summary": summarize(results, planned_count), "results": results}


def _worker_main() -> None:
    try:
        payload = json.load(sys.stdin)
        # Keep stdout machine-readable even if a parser dependency prints.
        with contextlib.redirect_stdout(sys.stderr):
            result = _compare_payload(payload)
    except (ImportError, metadata.PackageNotFoundError) as error:
        result = {"status": "dependency_error", "reason": type(error).__name__, "grade": None}
    except BaseException as error:
        # Includes Math-Verify's TimeoutException (a BaseException subclass).
        status = "timeout" if "timeout" in type(error).__name__.lower() else "evaluator_error"
        result = {"status": status, "reason": type(error).__name__, "grade": None}
    json.dump(result, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("Use scripts/grade_math_answers.py for the public CLI")
    _worker_main()
