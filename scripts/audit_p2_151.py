"""Audit and pull the terminal P2 development diagnostic from lab 151."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex

from remote import command, connection


ROOT = "/data/fangc/teamlearn/runs/p2review8b_20260930_v2"
RELEASE = "/home/fangc/teamlearn/releases/p2review8b_20260930_v2"
UNIT = "teamlearn-p2review8b-260930-v2"
PYTHON = "/data/fangc/envs/vllm-0.8.5/bin/python"


def main():
    remote_code = f'''import collections,hashlib,json,pathlib,subprocess
root=pathlib.Path({(ROOT + "/p2")!r})
controller=pathlib.Path({(ROOT + "/controller")!r})
expected={{"model_auto","reviewed_auto","model_reviewed_route","reviewed_reviewed_route"}}
rows=[json.loads(line) for line in (root/"results.jsonl").open() if line.strip()]
pairs=collections.defaultdict(dict)
models=set();delivery_failures=[]
for row in rows:
 key=(row["meta_id"],row["kind"],row["target_id"],row["seed"])
 pairs[key][row["method"]]=row
 for call in row["cost"]:
  if call.get("model"):models.add(call["model"])
 episode=json.loads((root/"episodes"/f'{{row["run_id"]}}.json').read_text())
 expected_deliveries=sum(bool(text) for text in episode["reports"].values())
 if len(episode["deliveries"])!=expected_deliveries or any(d.get("state")!="delivered" or d.get("entered_model_request_at") is None for d in episode["deliveries"]):
  delivery_failures.append(row["run_id"])
content_mismatches=[];routing_mismatches=[]
for key,group in pairs.items():
 if set(group)!=expected:continue
 episodes={{name:json.loads((root/"episodes"/f'{{row["run_id"]}}.json').read_text()) for name,row in group.items()}}
 content=lambda episode:sorted(set(text for text in episode["reports"].values() if text))
 if content(episodes["model_auto"])!=content(episodes["model_reviewed_route"]):content_mismatches.append([key,"model"])
 if content(episodes["reviewed_auto"])!=content(episodes["reviewed_reviewed_route"]):content_mismatches.append([key,"reviewed"])
 if group["model_auto"]["report_audit"]["recipients"]!=group["reviewed_auto"]["report_audit"]["recipients"]:routing_mismatches.append([key,"auto"])
 if group["model_reviewed_route"]["report_audit"]["recipients"]!=group["reviewed_reviewed_route"]["report_audit"]["recipients"]:routing_mismatches.append([key,"reviewed"])
design=json.loads((root/"design.json").read_text());review=json.loads((root/"review_manifest.json").read_text())
summary=json.loads((root/"summary.json").read_text());status=json.loads((controller/"status.json").read_text())
audit={{
 "rows":len(rows),"unique_run_ids":len({{row["run_id"] for row in rows}}),"episodes":len(list((root/"episodes").glob("*.json"))),
 "methods":sorted({{row["method"] for row in rows}}),"kinds":sorted({{row["kind"] for row in rows}}),"seeds":sorted({{row["seed"] for row in rows}}),
 "sources":len({{row["meta_id"] for row in rows}}),"mechanism_counts":{{m:sum(row["mechanism"]==m for row in rows) for m in ("F1","F2","F3","F4")}},
 "paired_units":len(pairs),"incomplete_paired_units":sum(set(group)!=expected for group in pairs.values()),"mock_rows":sum(bool(row["mock"]) for row in rows),
 "models":sorted(models),"delivery_failures":delivery_failures,"content_mismatches":content_mismatches,"routing_mismatches":routing_mismatches,
 "design_hash_matches":review["design_hash"]==__import__("teamlearn.common",fromlist=["digest"]).digest(design),
 "target_outcomes_used_for_design":review["target_outcomes_used"],"summary_complete":summary.get("complete"),
 "supervisor_state":status.get("state"),"gpu_released":status.get("gpu_released"),"owned_gpu_pids_remaining":status.get("owned_gpu_pids_remaining"),
 "results_sha256":hashlib.sha256((root/"results.jsonl").read_bytes()).hexdigest(),
 "systemd":subprocess.run(["systemctl","--user","show",{UNIT!r},"-p","ActiveState","-p","SubState","-p","MainPID","-p","Result"],capture_output=True,text=True).stdout,
}}
print(json.dumps(audit))'''
    with connection("151", "fangc-README.pdf") as client:
        result = command(client, "cd " + shlex.quote(RELEASE) + " && PYTHONPATH=" + shlex.quote(RELEASE) + " " + PYTHON + " -c " + shlex.quote(remote_code), timeout=180)
        if result["exit_code"]:
            raise RuntimeError(result)
        audit = json.loads(result["stdout"])
        problems = []
        expected = {"model_auto", "reviewed_auto", "model_reviewed_route", "reviewed_reviewed_route"}
        if audit["rows"] != 192 or audit["unique_run_ids"] != 192 or audit["episodes"] != 192:
            problems.append("row/episode completeness")
        if set(audit["methods"]) != expected or audit["kinds"] != ["same", "transfer"] or audit["seeds"] != [101]:
            problems.append("frozen cells/population")
        if audit["sources"] != 24 or audit["mechanism_counts"] != {mechanism: 48 for mechanism in ("F1", "F2", "F3", "F4")}:
            problems.append("source/mechanism balance")
        if audit["paired_units"] != 48 or audit["incomplete_paired_units"]:
            problems.append("paired 2x2 completeness")
        if audit["mock_rows"] or audit["models"] != ["Qwen3-8B"]:
            problems.append("real model identity")
        if audit["delivery_failures"] or audit["content_mismatches"] or audit["routing_mismatches"]:
            problems.append("delivery/content/routing fidelity")
        if not audit["design_hash_matches"] or audit["target_outcomes_used_for_design"] or not audit["summary_complete"]:
            problems.append("design/review/summary integrity")
        if audit["supervisor_state"] != "completed" or not audit["gpu_released"] or audit["owned_gpu_pids_remaining"]:
            problems.append("supervisor/GPU cleanup")
        if "ActiveState=inactive" not in audit["systemd"] or "MainPID=0" not in audit["systemd"]:
            problems.append("systemd terminal state")
        audit["passed"] = not problems; audit["problems"] = problems
        destination = Path("artifacts/p2_results"); destination.mkdir(parents=True, exist_ok=True)
        with client.open_sftp() as sftp:
            for name in ("design.json", "review_manifest.json", "results.jsonl", "summary.json"):
                sftp.get(ROOT + "/p2/" + name, str(destination / name))
            sftp.get(ROOT + "/controller/status.json", str(destination / "supervisor_status.json"))
            analysis = destination / "analysis"; analysis.mkdir(parents=True, exist_ok=True)
            for name in ("run_diagnostics.jsonl", "summary.csv", "mechanism_summary.csv", "family_summary.csv",
                         "factor_effects.csv", "factor_unit_effects.csv", "failure_counts.json", "manifest.json", "REPORT.md"):
                sftp.get(ROOT + "/p2/analysis/" + name, str(analysis / name))
        (destination / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": audit["passed"], "problems": problems, "out": str(destination),
                      "results_sha256": audit["results_sha256"]}, ensure_ascii=False, indent=2))
    if problems:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
