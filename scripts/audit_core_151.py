"""Audit completeness and immutability properties of the terminal core run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex

from remote import command, connection


ROOT = "/data/fangc/teamlearn/runs/core8b_20260928_v2"
PYTHON = "/data/fangc/envs/vllm-0.8.5/bin/python"


def main():
    code = f'''import collections,hashlib,json,pathlib
root=pathlib.Path({(ROOT + "/core")!r})
expected={{"E1":{{"components_SOCDKPU","components_SP"}},"E3":{{"matched","cycle","random"}}}}
audit={{}}
for experiment,arms in expected.items():
 path=root/f"test_{{experiment}}/results.jsonl"
 rows=[json.loads(line) for line in path.open() if line.strip()]
 pairs=collections.defaultdict(set)
 models=set()
 for row in rows:
  pairs[(row["meta_id"],row["target_id"],row["seed"])].add(row["method"])
  for call in row["cost"]+row["report_construction_cost"]:
   if call.get("model"):models.add(call["model"])
 audit[experiment]={{
  "rows":len(rows),
  "unique_run_ids":len({{row["run_id"] for row in rows}}),
  "duplicate_run_ids":len(rows)-len({{row["run_id"] for row in rows}}),
  "mock_rows":sum(bool(row["mock"]) for row in rows),
  "splits":sorted({{row["split"] for row in rows}}),
  "methods":sorted({{row["method"] for row in rows}}),
  "kinds":sorted({{row["kind"] for row in rows}}),
  "seeds":sorted({{row["seed"] for row in rows}}),
  "source_meta_ids":len({{row["meta_id"] for row in rows}}),
  "rule_families":len({{row["family_id"].split("/")[1] for row in rows}}),
  "paired_units":len(pairs),
  "incomplete_paired_units":sum(group!=arms for group in pairs.values()),
  "rows_with_no_model_calls":sum(not row["cost"] for row in rows),
  "models":sorted(models),
  "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
 }}
frozen=json.loads((root/"frozen.json").read_text())
design=json.loads((root/"design.json").read_text())
gate=json.loads((root/"test_source_gate.json").read_text())
completion=json.loads((root/"completion.json").read_text())
audit["frozen"]={{k:frozen.get(k) for k in ("dataset_hash","models_hash","code_hash","seeds","rounds","turns","primary")}}
audit["design_dataset_matches_freeze"]=design["dataset_hash"]==frozen["dataset_hash"]
audit["test_gate"]=gate
audit["completion"]=completion
print(json.dumps(audit))'''
    with connection("151", "fangc-README.pdf") as client:
        result = command(client, PYTHON + " -c " + shlex.quote(code), timeout=120)
    if result["exit_code"]:
        raise RuntimeError(result)
    audit = json.loads(result["stdout"])
    expected_rows = {"E1": 810, "E3": 1215}
    problems = []
    for experiment, count in expected_rows.items():
        row = audit[experiment]
        if row["rows"] != count or row["unique_run_ids"] != count:
            problems.append(f"{experiment} row count/uniqueness")
        if row["mock_rows"] or row["incomplete_paired_units"] or row["rows_with_no_model_calls"]:
            problems.append(f"{experiment} mock/incomplete/non-model rows")
        if row["splits"] != ["test"] or row["kinds"] != ["reversal", "same", "transfer"]:
            problems.append(f"{experiment} population")
        if row["seeds"] != [101, 211, 307] or row["source_meta_ids"] != 45 or row["rule_families"] != 12:
            problems.append(f"{experiment} frozen sampling")
        if row["models"] != ["Qwen3-8B"]:
            problems.append(f"{experiment} model identity")
    if not audit["design_dataset_matches_freeze"] or not audit["test_gate"].get("passed") or not audit["completion"].get("complete"):
        problems.append("freeze/gate/completion")
    audit["passed"] = not problems
    audit["problems"] = problems
    out = Path("artifacts/core_confirmation/audit.json")
    out.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": audit["passed"], "problems": problems, "out": str(out)}, ensure_ascii=False, indent=2))
    if problems:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
