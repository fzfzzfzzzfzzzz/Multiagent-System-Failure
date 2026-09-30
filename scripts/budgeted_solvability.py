"""Run legal-view feasibility witnesses under the formal team scheduler."""
from __future__ import annotations

import argparse
import json
import tempfile
from collections import defaultdict
from pathlib import Path

from teamlearn.common import ROLES, write_json, write_jsonl
from teamlearn.datasets import RULES, make_task
from teamlearn.diagnostics import BudgetedLegalModel, budget_ceiling_roles
from teamlearn.runner import run_team


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/budgeted_solvability")
    parser.add_argument("--turns", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--split", choices=tuple(RULES), default="dev")
    parser.add_argument("--backends", nargs="+", default=["workflow", "sqlite"])
    parser.add_argument("--kinds", nargs="+", default=["same", "transfer", "reversal"])
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix="teamlearn-budgeted-") as directory:
        directory = Path(directory)
        for turns in args.turns:
            for backend in args.backends:
                for kind in args.kinds:
                    for rule_index in range(len(RULES[args.split])):
                        for mechanism in ("F1", "F2", "F3", "F4"):
                            task = make_task(args.split, rule_index, mechanism, 0, kind, backend)
                            path = directory / f"{turns}-{backend}-{kind}-{rule_index}-{mechanism}.json"
                            write_json(path, task)
                            models = {role: BudgetedLegalModel() for role in ROLES}
                            episode = run_team(path, models, seed=19, rounds=args.rounds, turns_per_role=turns)
                            calls = {role: sum(row["role"] == role for row in episode["transcripts"]) for role in ROLES}
                            rows.append({
                                "turns_per_role": turns, "rounds": args.rounds,
                                "backend": backend, "kind": kind,
                                "rule_family": RULES[args.split][rule_index][0],
                                "mechanism": mechanism, **episode["score"],
                                "model_calls": calls,
                                "budget_ceiling_roles": budget_ceiling_roles(episode["transcripts"], turns),
                                "tool_errors": sum("error" in event["observation"] for event in episode["events"]),
                            })
    write_jsonl(out / "runs.jsonl", rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["turns_per_role"]].append(row)
    summary = {
        "purpose": "formal-interface feasibility witness; not a model baseline or research result",
        "split": args.split, "rounds": args.rounds,
        "runs": len(rows),
        "by_turn_budget": {
            str(turns): {
                "runs": len(items),
                "first_success": sum(row["first_success"] for row in items),
                "final_success": sum(row["final_success"] for row in items),
                "recurrence": sum(row["recurrence"] for row in items),
                "abstained": sum(row["abstained"] for row in items),
                "tool_errors": sum(row["tool_errors"] for row in items),
                "budget_ceiling_runs": sum(bool(row["budget_ceiling_roles"]) for row in items),
            } for turns, items in sorted(grouped.items())
        },
    }
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["by_turn_budget"].get("3", {}).get("first_success") != summary["by_turn_budget"].get("3", {}).get("runs"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
