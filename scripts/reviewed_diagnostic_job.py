"""P2 development diagnostic: report content x recipient routing.

The design is written before target execution.  Source failures and source
evidence may be inspected; target outcomes are never used for selection or
report construction.  Results are exploratory and cannot revise H1-H4.
"""
from __future__ import annotations

import argparse
import copy
import random
import time
from collections import defaultdict
from pathlib import Path

from teamlearn.common import ROLES, append_jsonl, digest, read_json, read_jsonl, write_json
from teamlearn.datasets import verify_dataset
from teamlearn.pipeline import code_hash, load_models, source_records
from teamlearn.progress import progress
from teamlearn.reports import Memory, build_deliveries
from teamlearn.reviewed import REVIEWED_RECIPIENTS, REVIEW_PROTOCOL_VERSION, review_experience
from teamlearn.runner import EnvironmentClient, run_team


MECHANISMS = ("F1", "F2", "F3", "F4")
KINDS = ("same", "transfer")
SEED = 101
ARMS = (
    {"name": "model_auto", "content": "model", "routing": "auto"},
    {"name": "reviewed_auto", "content": "reviewed", "routing": "auto"},
    {"name": "model_reviewed_route", "content": "model", "routing": "reviewed"},
    {"name": "reviewed_reviewed_route", "content": "reviewed", "routing": "reviewed"},
)


def _family(record):
    return record["family_id"].split("/")[1]


def select_sources(records, per_mechanism=6):
    """Round-robin over source rule families without reading target outcomes."""
    selected = []
    for mechanism in MECHANISMS:
        families = defaultdict(list)
        for record in records:
            if record["mechanism"] == mechanism and record["backend"] == "workflow":
                families[_family(record)].append(record)
        groups = [sorted(values, key=lambda row: row["meta_id"]) for _, values in sorted(families.items())]
        ordered = [group[index] for index in range(max(map(len, groups), default=0))
                   for group in groups if index < len(group)]
        if len(ordered) < per_mechanism:
            raise ValueError(f"not enough qualified development sources for {mechanism}")
        selected.extend(ordered[:per_mechanism])
    return selected


def freeze_design(dataset, records, out, model, models_path):
    value = {
        "stage": "P2-development-diagnostic",
        "status": "exploratory; cannot revise H1-H4",
        "dataset_hash": read_json(Path(dataset) / "manifest.json")["hash"],
        "selection": "six qualified workflow source failures per F1-F4, round-robin over source rule family; target outcomes not read",
        "sources": [[row["meta_id"], row["experience"]["frozen_hash"]] for row in records],
        "target_kinds": list(KINDS),
        "seed": SEED,
        "rounds": 2,
        "turns_per_role": 3,
        "arms": list(ARMS),
        "content": "model-generated frozen source report versus deterministic protocol-reviewed source report",
        "routing": "model-generated related_roles versus mechanism-specific predeclared semantic roles",
        "review_protocol": REVIEW_PROTOCOL_VERSION,
        "reviewed_recipients": REVIEWED_RECIPIENTS,
        "report_components": "SOCDKPU (V remains source-empty)",
        "report_budget": "512 tokens per recipient; identical fitted atom set for identical model/content",
        "role_views": False,
        "forwarding": False,
        "model": model,
        "models_hash": digest(Path(models_path).read_text(encoding="utf-8")),
        "teamlearn_code_hash": code_hash(),
        "job_code_hash": digest(Path(__file__).read_text(encoding="utf-8")),
    }
    path = Path(out) / "design.json"
    if path.exists() and read_json(path) != value:
        raise ValueError("P2 design changed; use a new output directory")
    write_json(path, value)
    return value


def _public_task(path):
    worker = EnvironmentClient(path)
    try:
        return worker.request("view", role="coordinator")["task"]
    finally:
        worker.close()


def _rates(rows):
    result = []
    for key in sorted({(r["kind"], r["mechanism"], r["method"]) for r in rows}):
        subset = [r for r in rows if (r["kind"], r["mechanism"], r["method"]) == key]
        opportunities = [r for r in subset if r["opportunity"]]
        result.append({"kind": key[0], "mechanism": key[1], "method": key[2], "n": len(subset),
                       "first_success": sum(r["first_success"] for r in subset) / len(subset),
                       "final_success": sum(r["final_success"] for r in subset) / len(subset),
                       "abstention": sum(r["abstained"] for r in subset) / len(subset),
                       "recurrence": (sum(r["recurrence"] for r in opportunities) / len(opportunities)) if opportunities else None})
    return result


def _factor_effects(rows, metric):
    cells = {name: {} for name in ("model_auto", "reviewed_auto", "model_reviewed_route", "reviewed_reviewed_route")}
    for row in rows:
        cells[row["method"]][(row["meta_id"], row["kind"], row["target_id"], row["seed"])] = float(row[metric])
    keys = set.intersection(*(set(cell) for cell in cells.values()))
    if metric == "recurrence":
        opportunity = {(row["meta_id"], row["kind"], row["target_id"], row["seed"]): row["opportunity"] for row in rows}
        keys = {key for key in keys if opportunity.get(key)}
    contrasts = []
    for key in sorted(keys):
        a = cells["model_auto"][key]; b = cells["reviewed_auto"][key]
        c = cells["model_reviewed_route"][key]; d = cells["reviewed_reviewed_route"][key]
        contrasts.append({"content": ((b - a) + (d - c)) / 2,
                          "routing": ((c - a) + (d - b)) / 2,
                          "interaction": d - b - c + a})
    return {"metric": metric, "n_paired": len(contrasts),
            **{name: (sum(row[name] for row in contrasts) / len(contrasts) if contrasts else None)
               for name in ("content", "routing", "interaction")}}


def prepare(dataset, sources, out, model, models_path):
    dataset = Path(dataset); out = Path(out)
    dataset_hash = verify_dataset(dataset)
    all_records = source_records(sources, "dev")
    if any(row.get("dataset_hash") != dataset_hash for row in all_records):
        raise ValueError("source collection used another dataset")
    if any(row.get("mock") for row in all_records):
        raise ValueError("mock sources cannot enter P2")
    records = select_sources(all_records)
    design = freeze_design(dataset, records, out, model, models_path)
    review_rows = []
    for record in records:
        original = record["experience"]
        reviewed = review_experience(original, record["mechanism"])
        review_rows.append({
            "meta_id": record["meta_id"], "mechanism": record["mechanism"], "family_id": record["family_id"],
            "source_frozen_hash": original["frozen_hash"], "reviewed_frozen_hash": reviewed["frozen_hash"],
            "auto_recipients": original.get("related_roles") or ["coordinator"],
            "reviewed_recipients": REVIEWED_RECIPIENTS[record["mechanism"]],
            "review_refs": {component: reviewed["components"][component][0]["refs"] for component in "CDKPU"},
        })
    review_manifest = {"complete": True, "rows": len(review_rows), "dataset_hash": dataset_hash,
                       "design_hash": digest(design), "review_protocol": REVIEW_PROTOCOL_VERSION,
                       "source_counts": {mechanism: sum(row["mechanism"] == mechanism for row in review_rows)
                                         for mechanism in MECHANISMS},
                       "families": sorted({_family(record) for record in records}), "records": review_rows,
                       "target_outcomes_used": False}
    path = out / "review_manifest.json"
    if path.exists() and read_json(path) != review_manifest:
        raise ValueError("P2 review manifest changed; use a new output directory")
    write_json(path, review_manifest)
    progress("p2_prepared", sources=len(records), design_hash=review_manifest["design_hash"])
    return records, design, review_manifest


def run(dataset, sources, out, models_path, model):
    dataset = Path(dataset); out = Path(out)
    records, design, _ = prepare(dataset, sources, out, model, models_path)
    config_hash = digest(design)
    done = {row["run_id"] for row in read_jsonl(out / "results.jsonl")} if (out / "results.jsonl").exists() else set()
    jobs = [(record, kind, arm) for record in records for kind in KINDS for arm in ARMS]
    random.Random(20260930).shuffle(jobs)
    actor_models, ledger, model_config = load_models(models_path, model)
    progress("p2_started", planned=len(jobs), already_complete=len(done))
    completed = 0
    try:
        for record, kind, arm in jobs:
            run_id = digest([config_hash, record["meta_id"], kind, SEED, arm["name"]])[:24]
            if run_id in done:
                continue
            task_path = dataset / "private" / f'{record["targets"][kind]}.json'
            public = _public_task(task_path)
            original = record["experience"]
            reviewed = review_experience(original, record["mechanism"])
            experience = original if arm["content"] == "model" else reviewed
            auto_recipients = original.get("related_roles") or ["coordinator"]
            recipients = auto_recipients if arm["routing"] == "auto" else REVIEWED_RECIPIENTS[record["mechanism"]]
            delivery_arm = {"method": "uniform", "components": "SOCDKPU", "budget": 512,
                            "budget_mode": "per_agent", "role_views": False, "recipients": list(recipients)}
            reports, audit = build_deliveries(experience, delivery_arm, actor_models, public, SEED,
                                              Memory([experience]))
            audit.update({"p2_arm": arm, "auto_recipients": list(auto_recipients),
                          "reviewed_recipients": list(REVIEWED_RECIPIENTS[record["mechanism"]]),
                          "content_hash": experience["frozen_hash"]})
            begin = len(ledger); started = time.monotonic()
            try:
                episode = run_team(task_path, actor_models, reports, SEED, rounds=2, turns_per_role=3,
                                   allow_forwarding=False)
            except Exception as error:
                append_jsonl(out / "errors.jsonl", {"run_id": run_id, "error": repr(error), "cost": ledger[begin:]})
                raise
            cost = copy.deepcopy(ledger[begin:])
            result = {"run_id": run_id, "config_hash": config_hash, "experiment": "P2", "method": arm["name"],
                      "arm": arm, "meta_id": record["meta_id"], "family_id": record["family_id"],
                      "mechanism": record["mechanism"], "target_id": record["targets"][kind], "kind": kind,
                      "seed": SEED, "split": "dev", "backend": record["backend"], "mock": False,
                      **episode["score"], "report_audit": audit, "cost": cost,
                      "report_construction_cost": original.get("report_cost", []) if arm["content"] == "model" else [],
                      "report_read_tokens_actual": episode["report_read_tokens_actual"],
                      "seconds": time.monotonic() - started, "source_hash": original["source_hash"],
                      "review_protocol": REVIEW_PROTOCOL_VERSION}
            write_json(out / "episodes" / f"{run_id}.json", {**episode, "reports": reports, "audit": audit, "cost": cost})
            append_jsonl(out / "results.jsonl", result)
            completed += 1
            progress("p2_run_completed", completed_total=len(done) + completed, planned=len(jobs),
                     method=arm["name"], kind=kind, mechanism=record["mechanism"])
            print(f"P2 {len(done)+completed}/{len(jobs)} {arm['name']} {kind} {record['mechanism']}: "
                  f"first={result['first_success']} recurrence={result['recurrence']}", flush=True)
    finally:
        for actor in actor_models.values():
            actor.close()
    rows = read_jsonl(out / "results.jsonl")
    if len(rows) != len(jobs):
        raise RuntimeError(f"P2 incomplete: {len(rows)}/{len(jobs)}")
    summary = {"complete": True, "rows": len(rows), "rates": _rates(rows),
               "factor_effects": [_factor_effects(rows, metric) for metric in ("first_success", "final_success", "abstained", "recurrence")],
               "interpretation": "exploratory development diagnostic; reference reports are deterministic protocol interventions, not human annotations"}
    write_json(out / "summary.json", summary)
    progress("p2_completed", rows=len(rows))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--models", default="configs/models_core_151.yaml")
    parser.add_argument("--model", default="qwen3_8b")
    parser.add_argument("--stage", choices=("prepare", "run"), default="run")
    args = parser.parse_args()
    if args.stage == "prepare":
        records, design, manifest = prepare(args.dataset, args.sources, args.out, args.model, args.models)
        print({"prepared": len(records), "design_hash": digest(design), "review_manifest": manifest["complete"]})
    else:
        print(run(args.dataset, args.sources, args.out, args.models, args.model))


if __name__ == "__main__":
    main()
