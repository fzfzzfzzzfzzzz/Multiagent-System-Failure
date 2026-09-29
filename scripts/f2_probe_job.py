"""Development-only probe for a call-budget setting that exposes F2 failures."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from teamlearn.common import read_json, read_jsonl, write_json
from teamlearn.pipeline import collect, source_records
from teamlearn.progress import progress


def candidates(rows):
    families = defaultdict(list)
    for row in rows:
        if row["mechanism"] == "F2" and row["backend"] == "workflow":
            families[row["family_id"]].append(row)
    # One candidate from each development rule family, fixed by meta_id.
    return [sorted(values, key=lambda value: value["meta_id"])[0]
            for _, values in sorted(families.items())]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--models", default="configs/models.yaml")
    parser.add_argument("--model", default="qwen3_8b")
    parser.add_argument("--turns", type=int, default=3)
    args = parser.parse_args()
    out = Path(args.out)
    selected = candidates(read_jsonl(Path(args.dataset) / "dev_manifest.jsonl"))
    design = {
        "stage": "development-only-F2-difficulty-probe",
        "dataset_hash": read_json(Path(args.dataset) / "manifest.json")["hash"],
        "selection": "one F2 workflow candidate per development rule family; no outcome selection",
        "meta_ids": [row["meta_id"] for row in selected],
        "source_seeds": [11, 29, 47],
        "rounds": 2,
        "turns_per_role": args.turns,
        "model": args.model,
    }
    if (out / "design.json").exists() and read_json(out / "design.json") != design:
        raise ValueError("probe design changed; use a fresh directory")
    write_json(out / "design.json", design)
    progress("f2_probe_starting", n_candidates=len(selected), turns=args.turns)
    collect(args.dataset, out / "collected", "dev", args.models, args.model,
            seeds=(11, 29, 47), rounds=2, turns=args.turns, meta_ids=set(design["meta_ids"]))
    records = source_records(out / "collected", "dev")
    result = {"qualified_F2": len(records), "candidates": len(selected), "passed": bool(records)}
    write_json(out / "completion.json", result)
    progress("f2_probe_completed", **result)


if __name__ == "__main__":
    main()
