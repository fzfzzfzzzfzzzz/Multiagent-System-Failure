"""Generate executable families, split by rule, never by random trajectory rows."""
from __future__ import annotations

import random
from pathlib import Path
from .common import ROLES, digest, write_json, write_jsonl

# Different business rules are held out; domains/numbers alone do not define splits.
RULES = {
    "train": [("invoice", "sum"), ("freight", "weighted_sum"), ("warehouse", "product"), ("demand", "median"),
              ("billing", "sum_plus_fee"), ("supply", "count_above_8"), ("weights", "weight_total"), ("packages", "pair_products")],
    "dev": [("capacity", "maximum"), ("energy", "sum_squares"), ("dispatch", "first_plus_last"),
            ("coverage", "count_even"), ("penalty", "sum_absolute_deviation"), ("storage", "ceiling_mean")],
    "test": [("portfolio", "minimum"), ("quality", "spread"), ("risk", "sum_cubes"), ("inventory", "sum_weighted_squares"),
             ("sampling", "floor_mean"), ("packing", "pair_distance"), ("water", "max_plus_min"), ("funds", "range_times_count"),
             ("signals", "alternating_sum"), ("routing", "weighted_range"), ("inspection", "sum_mod_7"), ("temperature", "median_squared")],
}
FORMULAS = {
    "sum":"sum(values)","weighted_sum":"sum(value[i] * weight[i])", "maximum":"max(values)", "sum_squares":"sum(v*v for v in values)",
    "minimum":"min(values)","spread":"max(values)-min(values)","product":"product of all values","median":"middle value after sorting",
    "sum_plus_fee":"sum(values)+7","count_above_8":"number of values strictly greater than 8","weight_total":"sum(weights)",
    "pair_products":"v0*v1+v0*v2+v1*v2","first_plus_last":"v0+v2","count_even":"number of even values",
    "sum_absolute_deviation":"sum(abs(v - median(values)))","ceiling_mean":"ceil(sum(values)/3)","sum_cubes":"sum(v**3)",
    "sum_weighted_squares":"sum(weights[i]*values[i]**2)","floor_mean":"floor(sum(values)/3)","pair_distance":"abs(v0-v1)+abs(v0-v2)+abs(v1-v2)",
    "max_plus_min":"max(values)+min(values)","range_times_count":"3*(max(values)-min(values))","alternating_sum":"v0-v1+v2",
    "weighted_range":"max(values[i]*weights[i])-min(values[i]*weights[i])","sum_mod_7":"sum(values) modulo 7","median_squared":"median(values)**2",
}
MECHANISMS = {
    "F1": "incorrect_transformation",
    "F2": "unconfirmed_handoff",
    "F3": "stale_version_consumed",
    "F4": "joint_resource_conflict",
    "F5": "nonindependent_verification",
    "F6": "unassigned_required_check",
}


def compute(rule, values, weights):
    import math
    import statistics
    if rule == "sum": return sum(values)
    if rule == "weighted_sum": return sum(v * w for v, w in zip(values, weights))
    if rule == "maximum": return max(values)
    if rule == "sum_squares": return sum(v * v for v in values)
    if rule == "minimum": return min(values)
    if rule == "spread": return max(values) - min(values)
    if rule == "product": return math.prod(values)
    if rule == "median": return statistics.median(values)
    if rule == "sum_plus_fee": return sum(values)+7
    if rule == "count_above_8": return sum(v>8 for v in values)
    if rule == "weight_total": return sum(weights)
    if rule == "pair_products": return values[0]*values[1]+values[0]*values[2]+values[1]*values[2]
    if rule == "first_plus_last": return values[0]+values[-1]
    if rule == "count_even": return sum(v%2==0 for v in values)
    if rule == "sum_absolute_deviation": return sum(abs(v-statistics.median(values)) for v in values)
    if rule == "ceiling_mean": return math.ceil(sum(values)/len(values))
    if rule == "sum_cubes": return sum(v**3 for v in values)
    if rule == "sum_weighted_squares": return sum(w*v*v for w,v in zip(weights,values))
    if rule == "floor_mean": return sum(values)//len(values)
    if rule == "pair_distance": return abs(values[0]-values[1])+abs(values[0]-values[2])+abs(values[1]-values[2])
    if rule == "max_plus_min": return max(values)+min(values)
    if rule == "range_times_count": return len(values)*(max(values)-min(values))
    if rule == "alternating_sum": return values[0]-values[1]+values[2]
    if rule == "weighted_range": return max(w*v for w,v in zip(weights,values))-min(w*v for w,v in zip(weights,values))
    if rule == "sum_mod_7": return sum(values)%7
    if rule == "median_squared": return statistics.median(values)**2
    raise ValueError(rule)


def make_task(split, rule_index, mechanism, instance, kind="same", backend="workflow"):
    domain, rule = RULES[split][rule_index]
    family = f"{backend}/{domain}/{mechanism}"
    rng = random.Random(digest([family, instance, kind]))
    values = [rng.randint(3, 15) for _ in range(3)]
    weights = [rng.randint(1, 4) for _ in values]
    names = list(ROLES)
    if kind in ("transfer", "renamed"):
        rng.shuffle(names)
    agents = {role: f"worker_{names.index(role)}" for role in ROLES}
    reversal = kind == "reversal"
    clean = kind in ("clean", "unrelated")
    mode = {"F1": "precision", "F2": "async", "F3": "versioned", "F4": "joint", "F5": "independent", "F6": "responsibility"}[mechanism]
    # Every active contract is available through legal role-specific tools.
    contract = {
        "rule": rule, "formula": FORMULAS[rule], "weights": weights, "unit": rng.choice(["credits", "units", "points"]),
        "rounding": "exact integer", "confirmation": "synchronous" if reversal or clean or mechanism != "F2" else "receipt_required",
        "version_policy": "pinned" if reversal or clean or mechanism != "F3" else "latest_at_commit",
        "joint_limit": 12 if reversal or clean or mechanism != "F4" else 7,
        "joint_min": 10 if reversal and mechanism=="F4" else 0,
        "independent_check": mechanism == "F5" and not reversal and not clean,
        "signoff_required": mechanism == "F6" and not reversal and not clean,
        "publication_update": mechanism == "F3" and not clean,
        "receipt_poll_is_legal": not reversal,
    }
    return {
        "task_id": digest([family, instance, kind])[:20], "family_id": family,
        "split": split, "mechanism": mechanism, "signature": MECHANISMS[mechanism],
        "kind": kind, "instance": instance, "backend": backend, "agents": agents,
        "opportunity": not reversal and not clean, "environment_version": "workflow-v1",
        "public": {"domain": domain, "goal": "Publish the correct aggregate of current service records, coordinate two resource allocations, and commit one complete authorized transaction.",
                   "agents": agents, "topology": "parallel_join" if kind == "transfer" else "serial",
                   "service_contract_id": f"{domain}-{mode}-{'v2' if reversal else 'v1'}"},
        "contract": contract, "records": values, "updated_records": [v + 2 for v in values],
        "allocation_options": {"provider": [2, 5], "analyst": [3, 6]},
    }


def generate(root, per_family=8, include_extended=True):
    root = Path(root)
    index = []
    task_hashes={}
    for split in RULES:
        rows = []
        for rule_index in range(len(RULES[split])):
            for mechanism in list(MECHANISMS)[:6 if include_extended else 4]:
                for backend in ("workflow", "sqlite"):
                    for instance in range(per_family):
                        source = make_task(split, rule_index, mechanism, instance, "source", backend)
                        targets = [make_task(split, rule_index, mechanism, instance, kind, backend) for kind in ("same", "transfer", "reversal", "clean", "unrelated", "renamed")]
                        meta_id = digest([source["task_id"], "meta"])[:20]
                        for task in [source] + targets:
                            write_json(root / "private" / f'{task["task_id"]}.json', task)
                            task_hashes[task["task_id"]]=digest(task)
                        row = {"meta_id": meta_id, "family_id": source["family_id"], "split": split, "mechanism": mechanism,
                               "backend": backend, "source_id": source["task_id"], "targets": {t["kind"]: t["task_id"] for t in targets}}
                        rows.append(row)
        write_jsonl(root / f"{split}_manifest.jsonl", rows)
        index.extend(rows)
    manifest = {"schema": 1, "per_family": per_family, "rules": RULES, "families": sorted({r["family_id"] for r in index}),
                "rows": len(index), "hash": digest({"index":index,"tasks":task_hashes}), "provenance": "programmatic environments; source failures must be collected, not invented",
                "external_validity": "sqlite is an executable application workflow, NOT CooperBench-derived or a public benchmark"}
    write_json(root / "manifest.json", manifest)
    write_json(root / "task_hashes.json",task_hashes)
    return manifest


def verify_dataset(root):
    from .common import read_json,read_jsonl
    root=Path(root)
    hashes=read_json(root/"task_hashes.json")
    index=[]
    for split in RULES:index.extend(read_jsonl(root/f"{split}_manifest.jsonl"))
    for task_id,expected in hashes.items():
        if digest(read_json(root/"private"/f"{task_id}.json"))!=expected:raise ValueError(f"private task changed: {task_id}")
    actual=digest({"index":index,"tasks":hashes})
    if actual!=read_json(root/"manifest.json")["hash"]:raise ValueError("dataset changed: content hash mismatch")
    return actual
