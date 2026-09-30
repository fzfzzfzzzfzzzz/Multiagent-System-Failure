"""Run and export the CPU-only P0 diagnostic on the completed core episodes."""
from __future__ import annotations

import json
from pathlib import Path
import shlex
import time

from remote import command, connection


ROOT = "/data/fangc/teamlearn/runs/core8b_20260928_v2"
CORE = ROOT + "/core"
REMOTE_CODE = "/home/fangc/teamlearn/diagnostics/core_20260930_v1/analyze_core_failures.py"
REMOTE_OUT = ROOT + "/diagnostics_20260930_v1"
UNIT = "teamlearn-corediag-260930-v1"
PYTHON = "/data/fangc/envs/vllm-0.8.5/bin/python"
FILES = ("manifest.json", "REPORT.md", "summary.csv", "mechanism_summary.csv",
         "family_summary.csv", "action_rates.csv", "first_error_rates.csv", "run_diagnostics.jsonl")


def gpu_probe(client):
    result = command(client, "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits")
    if result["exit_code"]:
        raise RuntimeError(result)
    rows = []
    for line in result["stdout"].splitlines():
        index, used, util = (int(value.strip()) for value in line.split(","))
        if index in (4, 5, 6, 7):
            rows.append({"gpu": index, "used_mb": used, "util": util})
    return rows


def main():
    local_out = Path("artifacts/core_diagnostics")
    local_out.mkdir(parents=True, exist_ok=True)
    with connection("151", "fangc-README.pdf") as client:
        preflight_code = f'''import json,pathlib
root=pathlib.Path({ROOT!r})
def read(p):return json.loads(p.read_text())
completion=read(root/"core/completion.json")
recovery=read(root/"controller/analysis_recovery.json")
print(json.dumps({{"completion":completion,"recovery_complete":recovery.get("complete")}}))'''
        preflight = command(client, PYTHON + " -c " + shlex.quote(preflight_code))
        if preflight["exit_code"]:
            raise RuntimeError(preflight)
        evidence = json.loads(preflight["stdout"])
        if not evidence["completion"].get("complete") or not evidence["recovery_complete"]:
            raise RuntimeError(f"core result is not terminal: {evidence}")
        before = gpu_probe(client)
        command(client, "mkdir -p " + shlex.quote(str(Path(REMOTE_CODE).parent).replace("\\", "/")))
        command(client, "mkdir -p " + shlex.quote(REMOTE_OUT))
        with client.open_sftp() as sftp:
            sftp.put("scripts/analyze_core_failures.py", REMOTE_CODE)
        command(client, "systemctl --user reset-failed " + shlex.quote(UNIT) + " 2>/dev/null || true")
        args = ["systemd-run", "--user", "--unit=" + UNIT, "--wait", "--collect", "--quiet",
                "--property=KillMode=control-group", "--property=RuntimeMaxSec=20min",
                PYTHON, REMOTE_CODE, "--root", CORE, "--out", REMOTE_OUT]
        run = command(client, " ".join(shlex.quote(value) for value in args), timeout=1500)
        after = gpu_probe(client)
        if run["exit_code"]:
            raise RuntimeError(run)
        with client.open_sftp() as sftp:
            for name in FILES:
                sftp.get(REMOTE_OUT + "/" + name, str(local_out / name))
        manifest = json.loads((local_out / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("missing_episodes") or manifest.get("rows") != 2025:
            raise RuntimeError(f"diagnostic export incomplete: {manifest}")
    status = {"completed_at": time.time(), "unit": UNIT, "unit_exit_code": run["exit_code"],
              "preflight": evidence, "gpu_before": before, "gpu_after": after,
              "remote_out": REMOTE_OUT, "local_out": str(local_out),
              "stdout_tail": run["stdout"].splitlines()[-20:], "stderr_tail": run["stderr"].splitlines()[-20:]}
    (local_out / "execution.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
