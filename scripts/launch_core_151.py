"""Deploy and start the frozen core-confirmation job on idle 151 GPUs 4-5."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import time
import zipfile

from remote import command, connection


RELEASE = "/home/fangc/teamlearn/releases/core8b_20260928_v2"
ROOT = "/data/fangc/teamlearn/runs/core8b_20260928_v2"
E0_ROOT = "/data/fangc/teamlearn/runs/pilot8b_20260928_v3/pilot"
DATASET = "/data/fangc/teamlearn/runs/pilot8b_20260928_v1/datasets"
UNIT = "teamlearn-core8b-260928-v2"
REMOTE_BUNDLE = "/home/fangc/teamlearn/uploads/core8b_20260928_v2.zip"


def bundle(path):
    root = Path(__file__).resolve().parents[1]
    include = ("teamlearn", "scripts", "configs", "protocol", "tests")
    files = []
    for item in include:
        files.extend(p for p in (root / item).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    files.extend(root / name for name in ("README.md", "pyproject.toml") if (root / name).exists())
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(files):
            archive.write(source, source.relative_to(root).as_posix())
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--credentials-pdf", default="fangc-README.pdf")
    args = parser.parse_args()
    local_bundle = Path("artifacts/core8b_20260928_v2.zip")
    local_bundle.parent.mkdir(parents=True, exist_ok=True)
    bundle_hash = bundle(local_bundle)
    supervisor = {
        "cwd": RELEASE,
        "state_dir": ROOT + "/controller",
        "port": 18152,
        "gpus": [4, 5],
        "server_command": ["bash", RELEASE + "/scripts/serve_model.sh", "8b", "4,5", "18152"],
        "client_command": [
            "/data/fangc/envs/vllm-0.8.5/bin/python", RELEASE + "/scripts/core_job.py",
            "--dataset", DATASET,
            "--e0-sources", E0_ROOT + "/collected",
            "--e0-gate", E0_ROOT + "/gate.json",
            "--out", ROOT + "/core",
            "--models", "configs/models_core_151.yaml",
            "--model", "qwen3_8b",
            "--stage", "all",
        ],
        "ready_url": "http://127.0.0.1:18152/v1/models",
        "expected_model": "Qwen3-8B",
        "startup_timeout": 600,
        "idle_timeout": 300,
        "max_runtime": 43200,
        "minimum_disk_gb": 5,
        "env": {
            "PYTHONPATH": "/data/fangc/teamlearn/gpu_tools:" + RELEASE,
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        },
    }
    launch = {"host": "151", "release": RELEASE, "root": ROOT, "bundle_sha256": bundle_hash,
              "supervisor": supervisor, "unit": UNIT, "started": False, "e0_root": E0_ROOT}
    config_local = Path("artifacts/core_supervisor_config.json")
    config_local.write_text(json.dumps(supervisor, indent=2), encoding="utf-8")
    with connection("151", args.credentials_pdf) as client:
        gate_probe = command(client, "cat " + shlex.quote(E0_ROOT + "/gate.json"))
        if gate_probe["exit_code"]:
            raise RuntimeError("E0 gate is not ready; core job was not started")
        gate = json.loads(gate_probe["stdout"])
        if not gate.get("passed"):
            raise RuntimeError(f"E0 gate failed; core job was not started: {gate}")
        active = command(client, "systemctl --user is-active " + shlex.quote(UNIT))
        if active["stdout"].strip() in ("active", "activating"):
            raise RuntimeError(f"{UNIT} is already active")
        memory = command(client, "awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo")
        if memory["exit_code"] or int(memory["stdout"].strip()) < 32768:
            raise RuntimeError(f"less than 32 GiB host memory is available: {memory}")
        command(client, "mkdir -p /home/fangc/teamlearn/uploads " + shlex.quote(RELEASE) + " " + shlex.quote(ROOT + "/controller"))
        with client.open_sftp() as sftp:
            sftp.put(str(local_bundle), REMOTE_BUNDLE)
            sftp.put(str(config_local), ROOT + "/controller/config.json")
        unpack = (
            "/data/fangc/envs/vllm-0.8.5/bin/python -c "
            + shlex.quote("import zipfile; zipfile.ZipFile(" + repr(REMOTE_BUNDLE) + ").extractall(" + repr(RELEASE) + ")")
        )
        result = command(client, unpack)
        if result["exit_code"]:
            raise RuntimeError(result)
        remote_hash = command(client, "sha256sum " + shlex.quote(REMOTE_BUNDLE) + " | cut -d' ' -f1")
        if remote_hash["stdout"].strip() != bundle_hash:
            raise RuntimeError("uploaded core bundle hash mismatch")
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
        state = command(client, "systemctl --user show " + shlex.quote(UNIT) + " -p ActiveState -p SubState -p MainPID -p Result")
        if "ActiveState=active" not in state["stdout"]:
            raise RuntimeError(f"core unit did not remain active: {state}")
    launch["started"] = True
    launch["started_at"] = time.time()
    Path("artifacts/core_launch_config.json").write_text(json.dumps(launch, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(launch, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
