"""Resume frozen core confirmation only after terminal E0 passes its gate."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shlex
import time

from remote import command, connection


E0_ROOT = "/data/fangc/teamlearn/runs/pilot8b_20260928_v3"
CORE_ROOT = "/data/fangc/teamlearn/runs/core8b_20260928_v2"
RELEASE = "/home/fangc/teamlearn/releases/core8b_20260928_v3"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--credentials-pdf", default="fangc-README.pdf")
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    if args.attempt < 1:
        raise ValueError("attempt must be positive")
    history = Path("artifacts/core_all_launch_config.json")
    if not history.exists():
        raise ValueError("missing preserved all-stage launch configuration")
    launch = json.loads(history.read_text(encoding="utf-8"))
    config = copy.deepcopy(launch["supervisor"])
    config["client_command"][-1] = "all"
    unit = f"teamlearn-coreconfirm8b-260928-v{args.attempt}"
    local_config = Path(f"artifacts/core_confirm_supervisor_config_v{args.attempt}.json")
    local_config.write_text(json.dumps(config, indent=2), encoding="utf-8")
    remote_config = CORE_ROOT + f"/controller/config_confirm_v{args.attempt}.json"
    with connection("151", args.credentials_pdf) as client:
        preflight_code = f'''import json,pathlib,subprocess
e0=pathlib.Path({E0_ROOT!r});core=pathlib.Path({CORE_ROOT!r})
def read(path):return json.loads(path.read_text())
status=read(e0/"controller/status.json")
gate=read(e0/"pilot/intervention_gate.json")
dev=read(core/"core/development_complete.json")
print(json.dumps({{"e0_state":status.get("state"),"e0_gpu_released":status.get("gpu_released"),"gate":gate,"development":dev}}))'''
        checked = command(client, "/data/fangc/envs/vllm-0.8.5/bin/python -c " + shlex.quote(preflight_code))
        if checked["exit_code"]:
            raise RuntimeError(checked)
        evidence = json.loads(checked["stdout"])
        if evidence["e0_state"] != "completed" or not evidence["e0_gpu_released"] or not evidence["gate"].get("passed") or not evidence["development"].get("complete"):
            raise RuntimeError(f"E0/development prerequisites not met: {evidence}")
        gpu_code = """import json,pynvml as n
n.nvmlInit()
print(json.dumps([{'gpu':i,'used_mb':n.nvmlDeviceGetMemoryInfo(n.nvmlDeviceGetHandleByIndex(i)).used//1048576,'util':n.nvmlDeviceGetUtilizationRates(n.nvmlDeviceGetHandleByIndex(i)).gpu,'pids':[p.pid for p in n.nvmlDeviceGetComputeRunningProcesses(n.nvmlDeviceGetHandleByIndex(i))]} for i in (4,5)]))"""
        probed = command(client, "PYTHONPATH=/data/fangc/teamlearn/gpu_tools /data/fangc/envs/vllm-0.8.5/bin/python -c " + shlex.quote(gpu_code))
        if probed["exit_code"]:
            raise RuntimeError(probed)
        gpus = json.loads(probed["stdout"])
        if any(row["used_mb"] > 512 or row["pids"] or row["util"] > 5 for row in gpus):
            raise RuntimeError(f"GPU 4-5 are occupied: {gpus}")
        memory = command(client, "awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo")
        if memory["exit_code"] or int(memory["stdout"].strip()) < 32768:
            raise RuntimeError(f"less than 32 GiB host memory is available: {memory}")
        active = command(client, "systemctl --user is-active " + shlex.quote(unit))
        if active["stdout"].strip() in ("active", "activating"):
            raise RuntimeError(f"{unit} is already active")
        with client.open_sftp() as sftp:
            sftp.put(str(local_config), remote_config)
        command(client, "systemctl --user reset-failed " + shlex.quote(unit) + " 2>/dev/null || true")
        start = " ".join([
            "systemd-run", "--user", "--unit=" + shlex.quote(unit), "--collect",
            "--property=KillMode=control-group", "--property=TimeoutStopSec=45s",
            shlex.quote("/usr/bin/env"), "PYTHONPATH=/data/fangc/teamlearn/gpu_tools",
            shlex.quote("/data/fangc/envs/vllm-0.8.5/bin/python"),
            shlex.quote(RELEASE + "/scripts/supervise.py"), "--config", shlex.quote(remote_config),
        ])
        started = command(client, start)
        if started["exit_code"]:
            raise RuntimeError(started)
        time.sleep(2)
        state = command(client, "systemctl --user show " + shlex.quote(unit) + " -p ActiveState -p SubState -p MainPID -p Result")
        if "ActiveState=active" not in state["stdout"]:
            raise RuntimeError(state)
    launch.update(unit=unit, supervisor=config, mode="all-after-E0-gate", started_at=time.time(), attempt=args.attempt)
    Path("artifacts/core_launch_config.json").write_text(json.dumps(launch, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"unit": state["stdout"], "gpu_before": gpus, "prerequisites": evidence}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
