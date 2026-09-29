"""Staged, outcome-blind core confirmation for the preregistered H1/H4 tests.

The development source failures come from the completed E0 batch.  Test
candidate IDs are frozen before any test source execution is read.  Re-running
the same stage is safe: source collection and target trials resume from their
durable records.
"""
from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

from teamlearn.analysis import summarize
from teamlearn.common import digest, read_json, read_jsonl, write_json
from teamlearn.pipeline import code_hash, collect, freeze_protocol, run_experiment, source_records
from teamlearn.progress import progress


MECHANISMS = ("F1", "F2", "F3", "F4")
TEST_SEEDS = (101, 211, 307)
DEV_SEEDS = (101,)
TURNS = 3
E1_ARMS = ("components_SOCDKPU", "components_SP")
E3_ARMS = ("matched", "cycle", "random")


def select_candidates(rows, per_mechanism=12):
    """Round-robin across rule families without consulting outcomes."""
    selected = []
    for mechanism in MECHANISMS:
        families = defaultdict(list)
        for row in rows:
            if row["mechanism"] == mechanism and row["backend"] == "workflow":
                families[row["family_id"]].append(row)
        groups = [sorted(values, key=lambda value: value["meta_id"])
                  for _, values in sorted(families.items())]
        ordered = [group[index] for index in range(max(map(len, groups), default=0))
                   for group in groups if index < len(group)]
        if len(ordered) < per_mechanism:
            raise ValueError(f"not enough {mechanism} test candidates")
        selected.extend(ordered[:per_mechanism])
    return selected


def cluster_name(record):
    return record["family_id"].split("/")[1]


def design(dataset, out, model):
    out = Path(out)
    candidates = select_candidates(read_jsonl(Path(dataset) / "test_manifest.jsonl"))
    value = {
        "stage": "minimal-confirmatory-core",
        "dataset_hash": read_json(Path(dataset) / "manifest.json")["hash"],
        "selection": "one fixed workflow candidate per F1-F4 and test rule family; no outcome selection",
        "test_meta_ids": [row["meta_id"] for row in candidates],
        "test_candidate_counts": {mechanism: sum(row["mechanism"] == mechanism for row in candidates)
                                  for mechanism in MECHANISMS},
        "test_rule_families": sorted({cluster_name(row) for row in candidates}),
        "source_seeds": [11, 29, 47],
        "development_target_seeds": list(DEV_SEEDS),
        "test_target_seeds": list(TEST_SEEDS),
        "rounds": 2,
        "turns_per_role": TURNS,
        "budget_rationale": "development-only F2 probe: 7 calls made confirmation invariant; 3 calls yielded qualified F2 failures in 4/6 rule families",
        "model": model,
        "development": {
            "sources": "first qualified E0 source per mechanism in meta_id order",
            "E1": list(E1_ARMS),
            "E2": "all 16 recipient subsets on same-condition targets",
            "E3": list(E3_ARMS),
            "decision_rule": "freeze unchanged after executable development checks; do not choose arms from effects",
        },
        "confirmatory": {
            "H1": {"experiment": "E1", "contrast": list(E1_ARMS)},
            "H4": {"experiment": "E3", "contrasts": [["matched", "cycle"], ["matched", "random"]]},
            "target_kinds": ["same", "transfer", "reversal"],
        },
        "test_source_gate": {"minimum_qualified_per_mechanism": 4, "minimum_rule_families": 10},
    }
    path = out / "design.json"
    if path.exists() and read_json(path) != value:
        raise ValueError("core design changed; use a new output directory")
    write_json(path, value)
    return value


def check_e0(e0_sources, e0_gate):
    gate = read_json(e0_gate)
    if not gate.get("passed"):
        raise RuntimeError(f"E0 gate did not pass: {gate}")
    records = source_records(e0_sources, "dev")
    counts = {mechanism: sum(row["mechanism"] == mechanism for row in records) for mechanism in MECHANISMS}
    if not all(counts.values()):
        raise RuntimeError(f"E0 source files do not cover F1-F4: {counts}")
    if any(row.get("code_hash") != code_hash() for row in records):
        raise ValueError("E0 source code hash differs from the confirmation release")
    return records, counts


def make_dev_subset(records, destination):
    destination = Path(destination)
    chosen = []
    for mechanism in MECHANISMS:
        chosen.append(min((row for row in records if row["mechanism"] == mechanism), key=lambda row: row["meta_id"]))
    for record in chosen:
        source = next(path for path in (Path(record["_source_root"]) / "sources").glob("*.json")
                      if path.stem == record["meta_id"])
        target = destination / "sources" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError(f"development subset source changed: {target}")
        if not target.exists():
            shutil.copy2(source, target)
    write_json(destination / "selection.json", {
        "rule": "first qualified E0 source per mechanism in meta_id order",
        "meta_ids": [row["meta_id"] for row in chosen],
        "source_hashes": [row["experience"]["frozen_hash"] for row in chosen],
    })
    return chosen


def development(dataset, e0_sources, e0_gate, out, models, model):
    records, counts = check_e0(e0_sources, e0_gate)
    # Keep the source path out of the copied JSON; it is only a local locator.
    for record in records:
        record["_source_root"] = str(Path(e0_sources))
    subset = Path(out) / "development_sources"
    chosen = make_dev_subset(records, subset)
    progress("core_development_starting", e0_counts=counts, n_sources=len(chosen))
    common = dict(dataset=dataset, sources=subset, split="dev", model_path=models, model_key=model,
                  seeds=DEV_SEEDS, rounds=2, turns=TURNS)
    run_experiment(out=Path(out) / "dev_E1", experiment="E1", selected_arms=E1_ARMS, **common)
    run_experiment(out=Path(out) / "dev_E2", experiment="E2", **common)
    run_experiment(out=Path(out) / "dev_E3", experiment="E3", selected_arms=E3_ARMS, **common)
    write_json(Path(out) / "development_complete.json", {
        "complete": True,
        "source_meta_ids": [row["meta_id"] for row in chosen],
        "note": "arms remain as predeclared; development outcomes were not used to choose test arms",
    })
    progress("core_development_completed")


def freeze(dataset, out, models):
    if not (Path(out) / "development_complete.json").exists():
        raise ValueError("complete development checks before freezing")
    frozen = Path(out) / "frozen.json"
    if frozen.exists():
        expected = read_json(frozen)
        if expected["code_hash"] != code_hash():
            raise ValueError("existing freeze belongs to another code version")
        return
    freeze_protocol(frozen, dataset, models, rounds=2, turns=TURNS, seeds=TEST_SEEDS)
    progress("core_protocol_frozen")


def collect_test(dataset, out, models, model, frozen_design):
    if not (Path(out) / "frozen.json").exists():
        raise ValueError("freeze protocol before test source collection")
    collected = Path(out) / "test_collected"
    progress("core_test_collection_starting", n_candidates=len(frozen_design["test_meta_ids"]))
    collect(dataset, collected, "test", models, model, seeds=(11, 29, 47), rounds=2, turns=TURNS,
            meta_ids=set(frozen_design["test_meta_ids"]))
    records = source_records(collected, "test")
    counts = {mechanism: sum(row["mechanism"] == mechanism for row in records) for mechanism in MECHANISMS}
    clusters = sorted({cluster_name(row) for row in records})
    threshold = frozen_design["test_source_gate"]
    gate = {
        "counts": counts,
        "qualified": len(records),
        "rule_families": clusters,
        "n_rule_families": len(clusters),
        **threshold,
        "passed": min(counts.values(), default=0) >= threshold["minimum_qualified_per_mechanism"]
                  and len(clusters) >= threshold["minimum_rule_families"],
    }
    write_json(Path(out) / "test_source_gate.json", gate)
    if not gate["passed"]:
        raise RuntimeError(f"predeclared test source gate failed; stop without changing candidates: {gate}")
    progress("core_test_collection_completed", qualified=len(records), rule_families=len(clusters))


def confirm(dataset, out, models, model, experiment):
    gate = read_json(Path(out) / "test_source_gate.json")
    if not gate.get("passed"):
        raise RuntimeError("test source gate has not passed")
    selected = E1_ARMS if experiment == "E1" else E3_ARMS
    progress("core_confirmation_starting", experiment=experiment)
    run_experiment(dataset, Path(out) / "test_collected", Path(out) / f"test_{experiment}", experiment,
                   "test", models, model, seeds=TEST_SEEDS, selected_arms=selected, rounds=2, turns=TURNS,
                   freeze_path=Path(out) / "frozen.json")
    progress("core_confirmation_completed", experiment=experiment)


def analyze(out):
    out = Path(out)
    for experiment in ("E1", "E3"):
        source = out / f"test_{experiment}" / "results.jsonl"
        if not source.exists():
            raise ValueError(f"missing confirmation results: {source}")
        summarize([source], out / "analysis" / experiment)
    payload = {
        "complete": True,
        "design_hash": digest(read_json(out / "design.json")),
        "E1_rows": len(read_jsonl(out / "test_E1" / "results.jsonl")),
        "E3_rows": len(read_jsonl(out / "test_E3" / "results.jsonl")),
        "interpretation_required": True,
    }
    write_json(out / "completion.json", payload)
    progress("core_job_completed", **payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--e0-sources", required=True)
    parser.add_argument("--e0-gate", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--models", default="configs/models.yaml")
    parser.add_argument("--model", default="qwen3_8b")
    parser.add_argument("--stage", choices=("development", "freeze", "collect-test", "confirm-e1", "confirm-e3", "analyze", "all"), default="all")
    args = parser.parse_args()
    frozen_design = design(args.dataset, args.out, args.model)
    stages = ("development", "freeze", "collect-test", "confirm-e1", "confirm-e3", "analyze") if args.stage == "all" else (args.stage,)
    for stage in stages:
        if stage == "development":
            development(args.dataset, args.e0_sources, args.e0_gate, args.out, args.models, args.model)
        elif stage == "freeze":
            freeze(args.dataset, args.out, args.models)
        elif stage == "collect-test":
            collect_test(args.dataset, args.out, args.models, args.model, frozen_design)
        elif stage == "confirm-e1":
            confirm(args.dataset, args.out, args.models, args.model, "E1")
        elif stage == "confirm-e3":
            confirm(args.dataset, args.out, args.models, args.model, "E3")
        else:
            analyze(args.out)


if __name__ == "__main__":
    main()
