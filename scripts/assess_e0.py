"""Descriptive E0 continuation gate; this is not a hypothesis test."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from teamlearn.common import read_json, read_jsonl, write_json


METHODS = ("no_memory", "uniform", "source_evidence", "full_history")


def assess(rows, source_gate):
    by_method = defaultdict(list)
    paired = defaultdict(dict)
    for row in rows:
        by_method[row["method"]].append(row)
        key = (row["meta_id"], row["target_id"], row["seed"])
        paired[key][row["method"]] = row
    complete = {key: group for key, group in paired.items() if all(method in group for method in METHODS)}
    sensitive = []
    mechanisms = set()
    for key, group in complete.items():
        baseline = group["no_memory"]
        if any((group[method]["first_success"], group[method]["recurrence"])
               != (baseline["first_success"], baseline["recurrence"]) for method in METHODS[1:]):
            sensitive.append(key)
            mechanisms.add(baseline["mechanism"])
    summary = {}
    for method in METHODS:
        group = by_method.get(method, [])
        summary[method] = {
            "n": len(group),
            "first_success": sum(bool(row["first_success"]) for row in group) / len(group) if group else None,
            "recurrence": sum(bool(row["recurrence"]) for row in group) / len(group) if group else None,
        }
    required_sensitive = max(2, round(0.05 * len(complete)))
    continuation_passed = bool(source_gate.get("passed")) and len(complete) > 0 \
        and summary["no_memory"]["recurrence"] >= 0.20 \
        and len(sensitive) >= required_sensitive and len(mechanisms) >= 2
    return {
        "purpose": "operational E0 task/intervention sensitivity gate; not confirmatory evidence",
        "source_gate": source_gate,
        "rows": len(rows),
        "complete_paired_units": len(complete),
        "method_summary": summary,
        "intervention_sensitive_pairs": len(sensitive),
        "sensitive_mechanisms": sorted(mechanisms),
        "minimum_sensitive_pairs": required_sensitive,
        "minimum_sensitive_mechanisms": 2,
        "minimum_no_memory_recurrence": 0.20,
        "passed": continuation_passed,
        "caution": "Gate checks observable variation and baseline difficulty only; H1/H4 require frozen independent-test contrasts.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--source-gate", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = assess(read_jsonl(args.results), read_json(args.source_gate))
    write_json(Path(args.out), result)
    print(result)
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
