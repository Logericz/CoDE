#!/usr/bin/env python3
"""Post hoc fixed-Q A/B/C diagnostic using frozen a08/a09 saved observations.

No imports from current online algorithms, no source-gate waiver, GPU, grading,
or parameter selection. B uses unqueried history and is explicitly an oracle.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANCHORS = {
    "collection_manifest": "7936c756fd18b46935e2d7ef04f9c6e37906820ea5933e1c7da9806492dd12f8",
    "collection_summary": "76af617b43cfdd5e05ce7fef4f5dc030951d0a3c589f4ec700d6868e81a8cd45",
    "development_manifest": "102780d05e8b5adcf91c15468eb0828a898b72fb36e65e7b06f279b5b857ddf1",
    "development_summary": "570fb6bc2c6a7e6367165fd05f32ee0c9236d96124859945d23074d2dfef1bea",
    "prior_analysis": "e8849e6d8298b252ef2080a6c7f0ba386b7641c0cf86d317b8c8e29c9125856a",
}
PROTOCOL = {"rule": "codestop", "r_min": .9, "r_max": .95, "ramp_steps": 2,
            "tau": 2.0, "deer_threshold": .95}
BRANCHES, VARIANTS = ("confidence", "degeneration", "or"), ("A", "B", "C")


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def probability(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def inside(root, relative):
    root, relative = Path(root).resolve(), Path(relative)
    path = (root / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root):
        raise ValueError("Input path escapes its run root")
    return path


class Evidence:
    def __init__(self):
        self.hashes, self.checks = {}, []

    def check(self, name, condition):
        self.checks.append({"check": name, "passed": bool(condition)})
        if not condition:
            raise ValueError(name)

    def read(self, path, expected=None, parse=True):
        path = str(Path(path).resolve())
        raw = Path(path).read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if path in self.hashes:
            self.check("unchanged_on_reread:" + path, self.hashes[path] == sha)
        else:
            self.hashes[path] = sha
        if expected is not None:
            self.check("sha256:" + path, sha == expected)
        if not parse:
            return raw
        def pairs(values):
            result = {}
            for key, value in values:
                if key in result:
                    raise ValueError("Duplicate JSON key: " + key)
                result[key] = value
            return result
        def reject(value):
            raise ValueError("Nonstandard JSON number: " + value)
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)

    def finish(self):
        after = {}
        for path, before in self.hashes.items():
            try:
                after[path] = digest(path)
            except OSError:
                after[path] = None
            self.checks.append({"check": "input_unchanged:" + path, "passed": after[path] == before})
        return after


def query_allowed(j):
    return j <= 3 or (j - 3) % 4 == 0


def threshold(j):
    return .9 + (.95 - .9) * min(1, (j - 1) / 2)


def score(history):
    """Independent expression for the code-log weighted decline count."""
    if len(history) < 3:
        return 0.0
    terminal = history[-1][0]
    return sum(1 + math.log(terminal / current[0]) for previous, current in zip(history, history[1:])
               if math.log(max(previous[1], 1e-12)) > math.log(max(current[1], 1e-12)))


def observations(probes, markers):
    rows, previous_position = [], 0
    for j, saved in enumerate(probes, 1):
        raw, position = saved["raw_observation"], saved["token_position"]
        ids, probs = raw["token_ids"], raw["token_probs"]
        valid = (3 <= len(ids) <= 21 and len(ids) == len(probs)
                 and all(type(i) is int and i >= 0 for i in ids)
                 and all(probability(p) for p in probs) and probability(raw["confidence_raw"])
                 and raw.get("invalid_reason") is None and not set(ids).intersection(markers["eos_ids"]))
        if not valid:
            raise ValueError("Fixed-Q scope requires valid frozen observations; invalid input is not repaired")
        if (saved["candidate_j"] != j or type(position) is not int or position <= previous_position
                or type(raw["ended_with_think"]) is not bool
                or raw["ended_with_think"] != (ids[-1] == markers["end_think"])):
            raise ValueError("Candidate order, position or probe ending is inconsistent")
        rows.append({"candidate_j": j, "token_position": position,
                     "confidence": raw["confidence_raw"], "complete": raw["ended_with_think"]})
        previous_position = position
    return rows


def crossing(point, variant, branch):
    value = point["D_sparse"] if variant == "C" else point["D_dense"]
    reasons = []
    if branch in ("confidence", "or") and point["complete"] and point["confidence"] > point["threshold_j"]:
        reasons.append("confidence")
    if branch in ("degeneration", "or") and value > PROTOCOL["tau"]:
        reasons.append("degeneration")
    return ({"candidate_j": point["candidate_j"], "token_position": point["token_position"],
             "generated_main_tokens_including_pending_wait": point["token_position"] + 1,
             "confidence": point["confidence"], "D": value, "reasons": reasons} if reasons else None)


def compare_crossings(first, later, effect, endpoint):
    result = {"candidate_delay": None, "token_delay": None,
              "endpoint_if_later_untriggered": endpoint if later is None else None}
    if first is None:
        result["classification"] = "both_untriggered" if later is None else "later_policy_only_crossing"
    elif later is None:
        result["classification"] = effect + "_no_crossing_before_endpoint"
    else:
        result.update(candidate_delay=later["candidate_j"] - first["candidate_j"],
                      token_delay=later["token_position"] - first["token_position"])
        if result["candidate_delay"] < 0:
            raise ValueError("A/B/C crossing order contradicts the fixed valid-history diagnostic")
        result["classification"] = "same_crossing" if result["candidate_delay"] == 0 else effect + "_delayed"
    return result


def diagnose(rows, endpoint):
    dense_history, sparse_history, points = [], [], []
    first = {branch: {variant: None for variant in VARIANTS} for branch in BRANCHES}
    for row in rows:
        j, position, confidence = row["candidate_j"], row["token_position"], row["confidence"]
        dense_history.append((position, confidence))
        queried = query_allowed(j)
        if queried:
            sparse_history.append((position, confidence))
        d_dense, d_sparse = score(dense_history), score(sparse_history) if queried else None
        point = {**row, "in_Q": queried, "query_k": len(sparse_history) if queried else None,
                 "D_dense": d_dense, "D_sparse": d_sparse,
                 "D_dense_minus_sparse_same_T": d_dense - d_sparse if queried else None,
                 "threshold_j": threshold(j), "threshold_k": threshold(len(sparse_history)) if queried else None}
        if queried and (point["threshold_j"] != point["threshold_k"] or d_sparse > d_dense + 1e-12):
            raise ValueError("Ramp zero control or common-endpoint score ordering failed")
        for branch in BRANCHES:
            for variant in VARIANTS:
                if first[branch][variant] is None and (variant == "A" or queried):
                    first[branch][variant] = crossing(point, variant, branch)
        points.append(point)
    comparisons = {}
    for branch, values in first.items():
        next_q = next((p for p in points if p["in_Q"] and values["A"] is not None
                       and p["candidate_j"] >= values["A"]["candidate_j"]), None)
        next_query = None if next_q is None else {
            "candidate_j": next_q["candidate_j"], "token_position": next_q["token_position"],
            "candidate_delay": next_q["candidate_j"] - values["A"]["candidate_j"],
            "token_delay": next_q["token_position"] - values["A"]["token_position"],
            "confidence": next_q["confidence"], "complete": next_q["complete"],
            "confidence_declined_since_first_A": next_q["confidence"] < values["A"]["confidence"],
            "confidence_crossing_at_next_Q": bool(crossing(next_q, "B", "confidence")),
        }
        comparisons[branch] = {
            "A_to_B_opportunity": compare_crossings(values["A"], values["B"], "opportunity", endpoint),
            "B_to_C_history": compare_crossings(values["B"], values["C"], "history", endpoint),
            "first_A_crossing_was_skipped": values["A"] is not None and not query_allowed(values["A"]["candidate_j"]),
            "next_Q_at_or_after_first_A": next_query,
            "endpoint_if_untriggered": {v: endpoint if values[v] is None else None for v in VARIANTS},
        }
    return {"points": points, "first_crossing": first, "comparisons": comparisons,
            "fixed_Q": [p["candidate_j"] for p in points if p["in_Q"]],
            "ramp_j_k_zero_control": True, "ramp_comparison_count": len(sparse_history), "endpoint": endpoint}


def load_run(evidence, root, name, count, prior):
    root = Path(root).resolve()
    manifest = evidence.read(root / "manifest.json", ANCHORS[name + "_manifest"])
    summary = evidence.read(root / "summary.json", ANCHORS[name + "_summary"])
    evidence.check(name + ":complete_denominator", summary["status"] == "completed"
        and summary["planned_count"] == summary["executed_count"] == summary["completed_count"] == count
        and summary["failed_count"] == summary["unexecuted_count"] == 0 and summary["unexecuted_requests"] == []
        and manifest["planned_count"] == len(manifest["requests"]) == len(summary["requests"]) == count)
    before, after = [evidence.read(root / filename) for filename in ("integrity-before.json", "integrity-after.json")]
    evidence.check(name + ":historical_recorded_integrity", before == after == manifest["source_input_sha256"]
                   and summary["source_input_identity_integrity"] == "unchanged")
    records = {}
    for index, (job, row) in enumerate(zip(manifest["requests"], summary["requests"])):
        evidence.check(name + f":plan_row:{index}", row["request_index"] == job["request_index"] == index
                       and all(row.get(k) == v for k, v in job.items())
                       and row["status"] == row["source_execution_status"] == "completed")
        record = evidence.read(inside(root, row["relative_directory"]) / "request.json", row["source_sha256"])
        key = (row["development_id"], row["configuration"])
        evidence.check(name + f":identity:{index}", key not in records and record["status"] == "completed"
            and all(record[k] == row[k] for k in ("development_id", "configuration", "sample_id"))
            and record["request_configuration"] == job["request_configuration"] and record["protocol_config"] == PROTOCOL)
        records[key] = record
    # Rebind only the run files actually consumed, not current sources replacing
    # the archived sources cited by the old 434-check analysis.
    for path, sha in list(evidence.hashes.items()):
        p = Path(path)
        if p.is_relative_to(root):
            suffix = "/" + root.name + "/" + str(p.relative_to(root))
            matches = [value for key, value in prior["input_sha256"].items() if key.endswith(suffix)]
            evidence.check(name + ":prior_analysis_binding:" + str(p.relative_to(root)), matches == [sha])
    return manifest, records


def match_actual(evidence, name, predicted, actual, collection):
    stops = [p for p in actual["probes"] if p["stop_applied"]]
    if predicted is None:
        matched = not stops and actual["stop_reason"] == collection["stop_reason"] == "natural_eos"
        matched &= actual["main_samples"] == collection["main_samples"]
    else:
        matched = len(stops) == 1 and all(stops[0][k] == predicted[k] for k in ("candidate_j", "token_position"))
        matched &= actual["main_generated_tokens"] == predicted["token_position"] + 1
        matched &= actual["stop_reasons"] == predicted["reasons"]
    evidence.check(name + ":actual_OR_endpoint", matched)
    evidence.check(name + ":actual_main_prefix", actual["main_samples"] == collection["main_samples"][:len(actual["main_samples"])])
    by_j = {p["candidate_j"]: p for p in collection["probes"]}
    evidence.check(name + ":actual_probe_observations", all(
        p["token_position"] == by_j[p["candidate_j"]]["token_position"]
        and p["raw_observation"] == by_j[p["candidate_j"]]["raw_observation"] for p in actual["probes"]))


def analyze(collection_root, development_root, prior_path):
    evidence = Evidence()
    report = {"schema_version": 1, "scope": "posthoc_development_ten_question_fixed_Q_mechanism_diagnostic",
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "planned_question_count": 10,
              "protocol": PROTOCOL, "Q_definition": "1,2,3,7,11,...; no terminal probe added",
              "variants": {"A": "all candidates, full history", "B": "Q only, full-history oracle", "C": "Q only, sparse history"},
              "questions": [], "online_latency_estimate_ms": None, "quality_selection": False}
    try:
        evidence.read(__file__, parse=False)
        evidence.read(ROOT / "manuscript/coling2027/EXPERIMENT_PLAN.md", parse=False)
        prior = evidence.read(prior_path, ANCHORS["prior_analysis"])
        evidence.check("historical_434_check_gate", prior["status"] == "passed" and prior["check_count"] == len(prior["checks"]) == 434
                       and all(c["passed"] for c in prior["checks"]) and prior["failed_checks"] == []
                       and prior["collection_complete"] is True and prior["natural_endpoint_count"] == 10)
        cm, collected = load_run(evidence, collection_root, "collection", 10, prior)
        dm, previous = load_run(evidence, development_root, "development", 40, prior)
        evidence.check("same_frozen_ten_questions", cm["questions"] == dm["questions"]
            and [q["development_id"] for q in cm["questions"]] == [f"dev{i:02d}" for i in range(1, 11)]
            and cm["master_seed"] == dm["master_seed"] == 42 and cm["rollout_id"] == dm["rollout_id"] == 0)
        report["frozen_questions"] = cm["questions"]
        for question in cm["questions"]:
            dev = question["development_id"]
            record = collected[(dev, "dense-collect-no-stop")]
            vanilla, actual_dense, actual_fixed = [previous[(dev, label)] for label in ("vanilla", "codestop-dense", "codestop-fixed")]
            markers, probes = record["backend_metadata"]["markers"], record["probes"]
            evidence.check(dev + ":complete_natural_collection", record["stop_reason"] == "natural_eos"
                and record["collection_complete_to_termination_or_cap"] is True
                and record["main_samples"] == vanilla["main_samples"]
                and record["output_token_ids"][-1] in markers["eos_ids"]
                and record["finalization_tokens"] == record["injected_prompt_tokens"] == 0
                and record["n_probes"] == record["n_candidates"] == len(probes) == len(record["candidates"]))
            rows = observations(probes, markers)
            evidence.check(dev + ":candidate_positions", [(p["candidate_j"], p["token_position"]) for p in probes]
                == [(c["candidate_j"], c["token_position"]) for c in record["candidates"]]
                and all(record["main_samples"][p["token_position"]]["token_id"] == markers["wait"]
                        and record["main_samples"][p["token_position"]]["phase"] == "reason" for p in probes))
            endpoint = {"kind": "natural_eos", "main_generated_tokens": record["main_generated_tokens"],
                        "last_candidate_j": len(probes) or None, "last_candidate_token_position": rows[-1]["token_position"] if rows else None,
                        "crossing_imputed": False}
            result = diagnose(rows, endpoint)
            for point, saved in zip(result["points"], probes):
                evidence.check(dev + f":saved_dense_D:{point['candidate_j']}", math.isclose(point["D_dense"], saved["decision"]["D_observed"], rel_tol=1e-12, abs_tol=1e-12)
                    and point["threshold_j"] == saved["decision"]["threshold_r"])
                reasons = crossing(point, "A", "or")
                evidence.check(dev + f":saved_stop_predicate:{point['candidate_j']}",
                    bool(reasons) == saved["would_stop"] and saved["should_stop"] is False and saved["stop_applied"] is False)
            for variant, actual, label in (("A", actual_dense, "dense"), ("C", actual_fixed, "fixed")):
                match_actual(evidence, dev + ":" + label, result["first_crossing"]["or"][variant], actual, record)
            evidence.check(dev + ":actual_fixed_Q", actual_fixed["schedule_config"]["kind"] == "fixed"
                and actual_fixed["schedule_config"]["fixed_interval"] == 4
                and [p["candidate_j"] for p in actual_fixed["probes"]]
                == [p["candidate_j"] for p in probes if query_allowed(p["candidate_j"])
                    and (result["first_crossing"]["or"]["C"] is None or p["candidate_j"] <= result["first_crossing"]["or"]["C"]["candidate_j"])])
            result.update(development_id=dev, sample_id=question["original_id"],
                recorded_a08_cost={"dense_total_ms": actual_dense["time_total_ms"], "fixed_total_ms": actual_fixed["time_total_ms"],
                    "fixed_minus_dense_total_ms": actual_fixed["time_total_ms"] - actual_dense["time_total_ms"],
                    "dense_main_tokens": actual_dense["main_generated_tokens"], "fixed_main_tokens": actual_fixed["main_generated_tokens"],
                    "interpretation": "Previously measured separate online requests; not a causal time decomposition of A/B/C or accumulated D gaps."})
            report["questions"].append(result)
        evidence.check("complete_diagnostic_denominator", len(report["questions"]) == 10)
        report["summary"] = {"questions": 10, "probes": sum(len(q["points"]) for q in report["questions"]),
            "classification_counts": {branch: {effect: dict(Counter(q["comparisons"][branch][effect]["classification"]
                for q in report["questions"])) for effect in ("A_to_B_opportunity", "B_to_C_history")} for branch in BRANCHES}}
    except (Exception, KeyboardInterrupt) as error:
        evidence.checks.append({"check": "analysis_exception", "passed": False,
                                "error_type": type(error).__name__, "message": str(error)})
    finally:
        after = evidence.finish()
    report.update(status="passed" if all(c["passed"] for c in evidence.checks) else "failed",
        checks=evidence.checks, check_count=len(evidence.checks), input_sha256_before=evidence.hashes, input_sha256_after=after,
        failed_checks=[c for c in evidence.checks if not c["passed"]],
        limitations=["Fixed ten exposed development questions, one main seed; 47 probes are not 47 independent problem samples.",
          "B and after-stop branch trajectories are offline diagnostics, not information free to an online policy.",
          "No crossing is null with an explicit endpoint; natural EOS is never fabricated as a threshold crossing.",
          "D gaps are compared only at common queried positions, not summed into a causal total-time explanation.",
          "Confidence, degeneration and OR are separate; a branch crossing after the actual OR stop is counterfactual.",
          "No candidate answers are generated or graded; no quality ranking, formal tuning or online speed estimate is performed.",
          "The historical 434-check artifact certifies its saved inputs, not current refactored source or a new GPU run."])
    return report


def markdown(report):
    lines = ["# 十题固定 Q 停止机会与历史损失诊断", "", f"状态：{report['status']}；检查 {report['check_count']} 项。", "",
             "Q = 1, 2, 3, 7, 11, …。A：全部候选／完整历史；B：仅 Q／完整历史 oracle；C：仅 Q／稀疏历史。", "",
             "crossing 表示为 候选编号@前缀token位置；已生成主token数还包含待处置的 Wait，即位置 + 1。", ""]
    for question in report["questions"]:
        lines += ["## " + question["development_id"] + " · " + question["sample_id"], "",
            "| 分支 | A | B | C | A→B 检查机会 | B→C 历史 |", "| --- | --- | --- | --- | --- | --- |"]
        for branch in BRANCHES:
            crossings = question["first_crossing"][branch]
            rendered = [f"{crossings[v]['candidate_j']}@{crossings[v]['token_position']}" if crossings[v] else "未触发（null）" for v in VARIANTS]
            def describe(effect):
                value = question["comparisons"][branch][effect]
                return value["classification"] + (f"；+{value['token_delay']} tokens" if value["token_delay"] is not None else "")
            lines.append("| " + " | ".join([branch, *rendered, describe("A_to_B_opportunity"), describe("B_to_C_history")]) + " |")
        lines += ["", f"自然终点：{question['endpoint']['main_generated_tokens']} 主tokens；不当作 crossing。j/k ramp 比较 {question['ramp_comparison_count']} 个 Q 点，差异为零（0 点时仅为空集）。", "",
                  "| Q 点 j | 前缀tokens | D_dense | D_Q | 同 T 分数差 |", "| --- | --- | --- | --- | --- |"]
        for p in question["points"]:
            if p["in_Q"]:
                lines.append(f"| {p['candidate_j']} | {p['token_position']} | {p['D_dense']:.6f} | {p['D_sparse']:.6f} | {p['D_dense_minus_sparse_same_T']:.6f} |")
        costs = question["recorded_a08_cost"]
        lines += ["", f"a08 已测请求：dense {costs['dense_total_ms']/1000:.3f} s，fixed {costs['fixed_total_ms']/1000:.3f} s，差 {costs['fixed_minus_dense_total_ms']/1000:+.3f} s。差值单列，不按上述 D 差或分支延迟分摊。", ""]
    lines += ["## 边界", "", *["- " + value for value in report["limitations"]], ""]
    if report["failed_checks"]:
        lines += ["失败检查：", "", *["- " + value["check"] for value in report["failed_checks"]], ""]
    return "\n".join(lines)


def write_report(report, output, inputs):
    output = Path(output).resolve()
    allowed = (ROOT / "runs/development10-analysis-20261009/mechanism").resolve()
    if output.exists() or not output.is_relative_to(allowed):
        raise ValueError("Output must be new and beneath the authorized mechanism directory")
    if any(output.is_relative_to(Path(p).resolve()) or Path(p).resolve().is_relative_to(output) for p in inputs):
        raise ValueError("Output must not overlap original inputs")
    output.mkdir(parents=True, exist_ok=False)
    (output / "mechanism.json").write_bytes(encoded(report))
    (output / "mechanism.md").write_text(markdown(report), encoding="utf-8")
    (output / "OUTPUT_SHA256.json").write_bytes(encoded({name: digest(output / name) for name in ("mechanism.json", "mechanism.md")}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("collection-run", "development-run", "prior-analysis", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args(argv)
    report = analyze(args.collection_run, args.development_run, args.prior_analysis)
    write_report(report, args.output_dir, (args.collection_run, args.development_run, args.prior_analysis))
    print(json.dumps({"status": report["status"], "checks": report["check_count"], "failed": report["failed_checks"],
                      "summary": report.get("summary"), "output": str(args.output_dir)}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
