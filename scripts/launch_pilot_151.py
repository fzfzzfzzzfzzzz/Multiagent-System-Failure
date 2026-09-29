"""Deploy and start E0 v3 on 151 after idle-GPU and memory checks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import time
import zipfile

from remote import command, connection


RELEASE = "/home/fangc/teamlearn/releases/pilot8b_20260928_v3"
ROOT = "/data/fangc/teamlearn/runs/pilot8b_20260928_v3"
DATASET = "/data/fangc/teamlearn/runs/pilot8b_20260928_v1/datasets"
UNIT = "teamlearn-pilot8b-260928-v3"
REMOTE_BUNDLE = "/home/fangc/teamlearn/uploads/pilot8b_20260928_v3.zip"


def make_bundle(path):
    root = Path(__file__).resolve().parents[1]
    files = []
    for directory in ("teamlearn", "scripts", "configs", "protocol", "tests"):
        files.extend(source for source in (root / directory).rglob("*")
                     if source.is_file() and "__pycache__" not in source.parts and source.suffix != ".pyc")
    files.extend(root / name for name in ("README.md", "pyproject.toml") if (root / name).exists())
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(files):
            archive.write(source, source.relative_to(root).as_posix())
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--credentials-pdf", default="fangc-README.pdf")
    args = parser.parse_args()
    bundle = Path("artifacts/pilot8b_20260928_v3.zip")
    bundle.parent.mkdir(parents=True, exist_ok=True)
    bundle_hash = make_bundle(bundle)
    supervisor = {
        "cwd": RELEASE,
        "state_dir": ROOT + "/controller",
        "port": 18151,
        "gpus": [6, 7],
        "server_command": ["bash", RELEASE + "/scripts/serve_model.sh", "8b", "6,7", "18151"],
        "client_command": [
            "/data/fangc/envs/vllm-0.8.5/bin/python", RELEASE + "/scripts/pilot_job.py",
            "--dataset", DATASET, "--out", ROOT + "/pilot", "--models", "configs/models.yaml",
            "--model", "qwen3_8b", "--turns", "3",
        ],
        "ready_url": "http://127.0.0.1:18151/v1/models",
        "expected_model": "Qwen3-8B",
        "startup_timeout": 600,
        "idle_timeout": 300,
        "max_runtime": 43200,
        "minimum_disk_gb": 5,
        "env": {
            "PYTHONPATH": "/data/fangc/teamlearn/gpu_tools:" + RELEASE,
            "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
            "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1",
        },
    }
    local_config = Path("artifacts/pilot8b_v3_supervisor_config.json")
    local_config.write_text(json.dumps(supervisor, indent=2), encoding="utf-8")
    with connection("151", args.credentials_pdf) as client:
        probe_code = """import json,pynvml as n
n.nvmlInit()
print(json.dumps([{'gpu':i,'used_mb':n.nvmlDeviceGetMemoryInfo(n.nvmlDeviceGetHandleByIndex(i)).used//1048576,'util':n.nvmlDeviceGetUtilizationRates(n.nvmlDeviceGetHandleByIndex(i)).gpu,'pids':[p.pid for p in n.nvmlDeviceGetComputeRunningProcesses(n.nvmlDeviceGetHandleByIndex(i))]} for i in (6,7)]))"""
        probe = command(client, "PYTHONPATH=/data/fangc/teamlearn/gpu_tools /data/fangc/envs/vllm-0.8.5/bin/python -c " + shlex.quote(probe_code))
        if probe["exit_code"]:
            raise RuntimeError(probe)
        gpu_state = json.loads(probe["stdout"])
        if any(row["used_mb"] > 512 or row["pids"] or row["util"] > 5 for row in gpu_state):
            raise RuntimeError(f"GPU 6-7 are occupied: {gpu_state}")
        memory = command(client, "awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo")
        if memory["exit_code"] or int(memory["stdout"].strip()) < 32768:
            raise RuntimeError(f"less than 32 GiB host memory is available: {memory}")
        active = command(client, "systemctl --user is-active " + shlex.quote(UNIT))
        if active["stdout"].strip() in ("active", "activating"):
            raise RuntimeError(f"{UNIT} is already active")
        made = command(client, "mkdir -p /home/fangc/teamlearn/uploads " + shlex.quote(RELEASE) + " " + shlex.quote(ROOT + "/controller"))
        if made["exit_code"]:
            raise RuntimeError(made)
        with client.open_sftp() as sftp:
            sftp.put(str(bundle), REMOTE_BUNDLE)
            sftp.put(str(local_config), ROOT + "/controller/config.json")
        unpack_code = "import zipfile; zipfile.ZipFile(%r).extractall(%r)" % (REMOTE_BUNDLE, RELEASE)
        unpacked = command(client, "/data/fangc/envs/vllm-0.8.5/bin/python -c " + shlex.quote(unpack_code))
        if unpacked["exit_code"]:
            raise RuntimeError(unpacked)
        remote_hash = command(client, "sha256sum " + shlex.quote(REMOTE_BUNDLE) + " | cut -d' ' -f1")["stdout"].strip()
        if remote_hash != bundle_hash:
            raise RuntimeError("uploaded pilot bundle hash mismatch")
        command(client, "systemctl --user reset-failed " + shlex.quote(UNIT) + " 2>/dev/null || true")
        start = " ".join([
            "systemd-run", "--user", "--unit=" + shlex.quote(UNIT), "--collect",
            "--property=KillMode=control-group", "--property=TimeoutStopSec=45s",
            shlex.quote("/usr/bin/env"), "PYTHONPATH=/data/fangc/teamlearn/gpu_tools",
            shlex.quote("/data/fangc/envs/vllm-0.8.5/bin/python"),
            shlex.quote(RELEASE + "/scripts/supervise.py"), "--config", shlex.quote(ROOT + "/controller/config.json"),
        ])
        launched = command(client, start)
        if launched["exit_code"]:
            raise RuntimeError(launched)
        time.sleep(2)
        unit_state = command(client, "systemctl --user show " + shlex.quote(UNIT) + " -p ActiveState -p SubState -p MainPID -p Result")
        if "ActiveState=active" not in unit_state["stdout"]:
            raise RuntimeError(f"pilot unit did not remain active: {unit_state}")
    launch = {"host": "151", "release": RELEASE, "root": ROOT, "bundle_sha256": bundle_hash,
              "supervisor": supervisor, "unit": UNIT, "started": True, "started_at": time.time(),
              "development_change": "turns_per_role=3 after F2 probe passed 4/6; no test task inspected"}
    Path("artifacts/pilot_launch_config.json").write_text(json.dumps(launch, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"gpu_before": gpu_state, "unit": unit_state["stdout"], "launch": launch}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
