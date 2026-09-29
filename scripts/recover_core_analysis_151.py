"""Recover the CPU-only core analysis after all frozen model trials completed.

The failed confirmation supervisor already released its model process group.  This
script verifies that state, deploys only the optional-plot fix, and runs the
unchanged preregistered numerical analysis in its own bounded systemd unit.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex
import time

from remote import command, connection


ROOT = "/data/fangc/teamlearn/runs/core8b_20260928_v2"
RELEASE = "/home/fangc/teamlearn/releases/core8b_20260928_v3"
UNIT = "teamlearn-coreanalysis8b-260929-v1"
PYTHON = "/data/fangc/envs/vllm-0.8.5/bin/python"


def read_remote_json(sftp, path):
    with sftp.open(path, "r") as handle:
        raw = handle.read()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def gpu_probe(client):
    code = """import json,pynvml as n
n.nvmlInit()
print(json.dumps([{'gpu':i,'used_mb':n.nvmlDeviceGetMemoryInfo(n.nvmlDeviceGetHandleByIndex(i)).used//1048576,'util':n.nvmlDeviceGetUtilizationRates(n.nvmlDeviceGetHandleByIndex(i)).gpu,'pids':[p.pid for p in n.nvmlDeviceGetComputeRunningProcesses(n.nvmlDeviceGetHandleByIndex(i))]} for i in (4,5)]))"""
    result = command(client, f"PYTHONPATH=/data/fangc/teamlearn/gpu_tools {PYTHON} -c " + shlex.quote(code))
    if result["exit_code"]:
        raise RuntimeError(result)
    return json.loads(result["stdout"])


def main():
    local_analysis = Path("teamlearn/analysis.py")
    local_bytes = local_analysis.read_bytes()
    local_sha = hashlib.sha256(local_bytes).hexdigest()
    remote_analysis = RELEASE + "/teamlearn/analysis.py"
    recovery_path = ROOT + "/controller/analysis_recovery.json"
    with connection("151", "fangc-README.pdf") as client:
        with client.open_sftp() as sftp:
            status = read_remote_json(sftp, ROOT + "/controller/status.json")
            gate = read_remote_json(sftp, ROOT + "/core/test_source_gate.json")
        if status.get("state") != "failed" or not status.get("gpu_released") or status.get("owned_gpu_pids_remaining"):
            raise RuntimeError(f"confirmation attempt is not failed-and-clean: {status}")
        tail = command(client, "tail -n 30 " + shlex.quote(ROOT + "/controller/experiment.log"))
        if "ModuleNotFoundError: No module named 'matplotlib'" not in tail["stdout"]:
            raise RuntimeError(f"unexpected terminal failure; refusing analysis-only recovery: {tail}")
        counts_code = f"""import json,pathlib
root=pathlib.Path({(ROOT + '/core')!r})
def lines(path):return sum(1 for line in path.open() if line.strip())
print(json.dumps({{'E1':lines(root/'test_E1/results.jsonl'),'E3':lines(root/'test_E3/results.jsonl')}}))"""
        counts_result = command(client, PYTHON + " -c " + shlex.quote(counts_code))
        counts = json.loads(counts_result["stdout"])
        qualified = gate["qualified"]
        planned = {"E1": qualified * 3 * 3 * 2, "E3": qualified * 3 * 3 * 3}
        if counts != planned:
            raise RuntimeError(f"frozen trial matrix is incomplete: {counts} != {planned}")
        gpu_before = gpu_probe(client)
        if any(row["pids"] or row["used_mb"] > 512 or row["util"] > 5 for row in gpu_before):
            raise RuntimeError(f"GPU 4-5 changed occupancy before recovery: {gpu_before}")
        with client.open_sftp() as sftp:
            with sftp.open(remote_analysis, "rb") as handle:
                remote_before_sha = hashlib.sha256(handle.read()).hexdigest()
            sftp.put(str(local_analysis), remote_analysis)
        command(client, "systemctl --user reset-failed " + shlex.quote(UNIT) + " 2>/dev/null || true")
        args = [
            "systemd-run", "--user", "--unit=" + UNIT, "--wait", "--collect", "--quiet",
            "--property=KillMode=control-group", "--property=RuntimeMaxSec=15min",
            "/usr/bin/env", "PYTHONPATH=/data/fangc/teamlearn/gpu_tools:" + RELEASE,
            "OMP_NUM_THREADS=4", "MKL_NUM_THREADS=4", "PYTHONUNBUFFERED=1",
            PYTHON, RELEASE + "/scripts/core_job.py",
            "--dataset", "/data/fangc/teamlearn/runs/pilot8b_20260928_v1/datasets",
            "--e0-sources", "/data/fangc/teamlearn/runs/pilot8b_20260928_v3/pilot/collected",
            "--e0-gate", "/data/fangc/teamlearn/runs/pilot8b_20260928_v3/pilot/gate.json",
            "--out", ROOT + "/core", "--models", "configs/models_core_151.yaml",
            "--model", "qwen3_8b", "--stage", "analyze",
        ]
        run = command(client, " ".join(shlex.quote(item) for item in args), timeout=900)
        gpu_after = gpu_probe(client)
        with client.open_sftp() as sftp:
            completion = read_remote_json(sftp, ROOT + "/core/completion.json")
            e1_manifest = read_remote_json(sftp, ROOT + "/core/analysis/E1/analysis_manifest.json")
            e3_manifest = read_remote_json(sftp, ROOT + "/core/analysis/E3/analysis_manifest.json")
        complete = (run["exit_code"] == 0 and completion.get("complete")
                    and completion.get("E1_rows") == planned["E1"]
                    and completion.get("E3_rows") == planned["E3"]
                    and e1_manifest.get("rows") == planned["E1"]
                    and e3_manifest.get("rows") == planned["E3"]
                    and not any(row["pids"] for row in gpu_after))
        recovery = {
            "complete": complete,
            "recovered_at": time.time(),
            "reason": "all model trials completed; original analysis stopped only because optional matplotlib was unavailable",
            "original_controller_state": status.get("state"),
            "original_gpu_released": status.get("gpu_released"),
            "unit": UNIT,
            "unit_exit_code": run["exit_code"],
            "analysis_patch": "optional visualization omission; numerical estimands and tests unchanged",
            "analysis_sha256_before": remote_before_sha,
            "analysis_sha256_after": local_sha,
            "rows": counts,
            "planned_rows": planned,
            "gpu_before": gpu_before,
            "gpu_after": gpu_after,
            "stdout_tail": run["stdout"].splitlines()[-20:],
            "stderr_tail": run["stderr"].splitlines()[-20:],
        }
        with client.open_sftp() as sftp:
            with sftp.open(recovery_path, "w") as handle:
                handle.write(json.dumps(recovery, ensure_ascii=False, indent=2))
        if not complete:
            raise RuntimeError(recovery)
    Path("artifacts/core_analysis_recovery.json").write_text(
        json.dumps(recovery, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(recovery, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
