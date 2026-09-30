"""Deploy and start the supervised P2 diagnostic on idle 151 GPUs 4-5."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import time
import zipfile

from remote import command, connection


RELEASE = "/home/fangc/teamlearn/releases/p2review8b_20260930_v2"
ROOT = "/data/fangc/teamlearn/runs/p2review8b_20260930_v2"
SOURCES = "/data/fangc/teamlearn/runs/pilot8b_20260928_v3/pilot/collected"
E0_GATE = "/data/fangc/teamlearn/runs/pilot8b_20260928_v3/pilot/gate.json"
DATASET = "/data/fangc/teamlearn/runs/pilot8b_20260928_v1/datasets"
UNIT = "teamlearn-p2review8b-260930-v2"
REMOTE_BUNDLE = "/home/fangc/teamlearn/uploads/p2review8b_20260930_v2.zip"


def bundle(path):
    root = Path(__file__).resolve().parents[1]
    files = []
    for item in ("teamlearn", "scripts", "configs", "protocol", "tests"):
        files.extend(p for p in (root / item).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    files.extend(root / name for name in ("README.md", "pyproject.toml") if (root / name).exists())
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(files):
            archive.write(source, source.relative_to(root).as_posix())
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--credentials-pdf", default="fangc-README.pdf")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    local_bundle = Path("artifacts/p2review8b_20260930_v2.zip")
    local_bundle.parent.mkdir(parents=True, exist_ok=True)
    bundle_hash = bundle(local_bundle)
    supervisor = {
        "cwd": RELEASE,
        "state_dir": ROOT + "/controller",
        "port": 18153,
        "gpus": [4, 5],
        "server_command": ["bash", RELEASE + "/scripts/serve_model.sh", "8b", "4,5", "18153"],
        "client_command": [
            "/data/fangc/envs/vllm-0.8.5/bin/python", RELEASE + "/scripts/reviewed_diagnostic_job.py",
            "--dataset", DATASET, "--sources", SOURCES, "--out", ROOT + "/p2",
            "--models", "configs/models_p2_151.yaml", "--model", "qwen3_8b",
        ],
        "ready_url": "http://127.0.0.1:18153/v1/models",
        "expected_model": "Qwen3-8B",
        "startup_timeout": 600,
        "idle_timeout": 300,
        "max_runtime": 28800,
        "minimum_disk_gb": 10,
        "env": {"PYTHONPATH": "/data/fangc/teamlearn/gpu_tools:" + RELEASE, "OMP_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"},
    }
    launch = {"host": "151", "release": RELEASE, "root": ROOT, "bundle_sha256": bundle_hash,
              "supervisor": supervisor, "unit": UNIT, "started": False, "sources": SOURCES, "dataset": DATASET}
    config_local = Path("artifacts/p2_supervisor_config.json")
    config_local.write_text(json.dumps(supervisor, indent=2), encoding="utf-8")
    with connection("151", args.credentials_pdf) as client:
        gate_result = command(client, "cat " + shlex.quote(E0_GATE))
        if gate_result["exit_code"] or not json.loads(gate_result["stdout"]).get("passed"):
            raise RuntimeError("E0 source gate is not ready")
        active = command(client, "systemctl --user is-active " + shlex.quote(UNIT))
        if active["stdout"].strip() in ("active", "activating"):
            raise RuntimeError(f"{UNIT} is already active")
        gpu_probe = command(client, "nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader; "
                                    "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits")
        if gpu_probe["exit_code"]:
            raise RuntimeError(f"GPU preflight failed: {gpu_probe}")
        memory = command(client, "awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo")
        if memory["exit_code"] or int(memory["stdout"].strip()) < 32768:
            raise RuntimeError(f"less than 32 GiB host memory is available: {memory}")
        command(client, "mkdir -p /home/fangc/teamlearn/uploads " + shlex.quote(RELEASE) + " " + shlex.quote(ROOT + "/controller"))
        with client.open_sftp() as sftp:
            sftp.put(str(local_bundle), REMOTE_BUNDLE)
            sftp.put(str(config_local), ROOT + "/controller/config.json")
        unpack = "/data/fangc/envs/vllm-0.8.5/bin/python -c " + shlex.quote(
            "import zipfile; zipfile.ZipFile(" + repr(REMOTE_BUNDLE) + ").extractall(" + repr(RELEASE) + ")")
        result = command(client, unpack)
        if result["exit_code"]:
            raise RuntimeError(result)
        remote_hash = command(client, "sha256sum " + shlex.quote(REMOTE_BUNDLE) + " | cut -d' ' -f1")
        if remote_hash["stdout"].strip() != bundle_hash:
            raise RuntimeError("uploaded P2 bundle hash mismatch")
        if args.prepare_only:
            prepare = "cd " + shlex.quote(RELEASE) + " && " + " ".join([
                "timeout", "300", shlex.quote("/usr/bin/env"), "PYTHONPATH=" + shlex.quote(RELEASE),
                shlex.quote("/data/fangc/envs/vllm-0.8.5/bin/python"),
                shlex.quote(RELEASE + "/scripts/reviewed_diagnostic_job.py"),
                "--dataset", shlex.quote(DATASET), "--sources", shlex.quote(SOURCES),
                "--out", shlex.quote(ROOT + "/p2"), "--models", "configs/models_p2_151.yaml",
                "--model", "qwen3_8b", "--stage", "prepare",
            ])
            prepared = command(client, prepare, timeout=360)
            if prepared["exit_code"]:
                raise RuntimeError(f"P2 prepare failed: {prepared}")
            launch.update({"prepared": True, "prepare_stdout": prepared["stdout"], "prepared_at": time.time()})
            Path("artifacts/p2_prepare.json").write_text(json.dumps(launch, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(launch, ensure_ascii=False, indent=2))
            return
        command(client, "systemctl --user reset-failed " + shlex.quote(UNIT) + " 2>/dev/null || true")
        start = " ".join(["systemd-run", "--user", "--unit=" + shlex.quote(UNIT), "--collect",
                          "--property=KillMode=control-group", "--property=TimeoutStopSec=45s",
                          shlex.quote("/usr/bin/env"), "PYTHONPATH=/data/fangc/teamlearn/gpu_tools",
                          shlex.quote("/data/fangc/envs/vllm-0.8.5/bin/python"),
                          shlex.quote(RELEASE + "/scripts/supervise.py"), "--config", shlex.quote(ROOT + "/controller/config.json")])
        launched = command(client, start)
        if launched["exit_code"]:
            raise RuntimeError(launched)
        time.sleep(3)
        state = command(client, "systemctl --user show " + shlex.quote(UNIT) + " -p ActiveState -p SubState -p MainPID -p Result")
        if "ActiveState=active" not in state["stdout"]:
            raise RuntimeError(f"P2 unit did not remain active: {state}")
    launch["started"] = True; launch["started_at"] = time.time()
    Path("artifacts/p2_launch_config.json").write_text(json.dumps(launch, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(launch, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
