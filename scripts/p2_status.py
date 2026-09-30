"""Inspect or explicitly stop the project-owned P2 unit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex

from remote import command, connection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--credentials-pdf", default="fangc-README.pdf")
    parser.add_argument("--launch", default="artifacts/p2_launch_config.json")
    parser.add_argument("--stop", action="store_true")
    args = parser.parse_args()
    info = json.loads(Path(args.launch).read_text(encoding="utf-8"))
    root, unit = info["root"], info["unit"]
    if not unit.startswith("teamlearn-p2review") or not root.startswith("/data/fangc/teamlearn/"):
        raise ValueError("not a project-owned P2 job")
    with connection(info["host"], args.credentials_pdf) as client:
        if args.stop:
            stopped = command(client, "systemctl --user stop " + shlex.quote(unit))
            if stopped["exit_code"]:
                raise RuntimeError(stopped)
        probe = f'''import json,pathlib,subprocess,time
root=pathlib.Path({root!r})
def read(path):
 try:return json.loads(path.read_text())
 except (OSError,ValueError):return None
def lines(path):
 try:return sum(1 for line in path.open() if line.strip())
 except OSError:return 0
def tail(path):
 try:return path.read_text(errors="replace").splitlines()[-12:]
 except OSError:return []
result={{"checked_at":time.time(),
 "unit":subprocess.run(["systemctl","--user","show",{unit!r},"-p","ActiveState","-p","SubState","-p","MainPID","-p","Result"],capture_output=True,text=True).stdout,
 "status":read(root/"controller/status.json"),"heartbeat":read(root/"controller/client_heartbeat.json"),
 "progress":{{"rows":lines(root/"p2/results.jsonl"),"episodes":len(list((root/"p2/episodes").glob("*.json"))) }},
 "design":read(root/"p2/design.json"),"summary":read(root/"p2/summary.json"),
 "experiment_tail":tail(root/"controller/experiment.log"),"model_tail":tail(root/"controller/model.log")}}
try:
 import gpustat
 result["gpus"]=[{{"gpu":g.index,"used_mb":g.memory_used,"util":g.utilization,"compute_pids":[p["pid"] for p in g.processes if p["username"]!="gdm"]}} for g in gpustat.GPUStatCollection.new_query()]
except Exception as error:result["gpu_probe_error"]=str(error)
status=result.get("status") or {{}}
if status.get("config"):status.pop("config",None)
print(json.dumps(result))'''
        result = command(client, "PYTHONPATH=/data/fangc/teamlearn/gpu_tools /data/fangc/envs/vllm-0.8.5/bin/python -c " + shlex.quote(probe))
        if result["exit_code"]:
            raise RuntimeError(result)
        value = json.loads(result["stdout"])
        Path("artifacts/p2_latest.json").write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
