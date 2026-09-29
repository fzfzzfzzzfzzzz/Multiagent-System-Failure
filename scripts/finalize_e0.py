"""Verify terminal E0, assess intervention sensitivity, and persist the gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from assess_e0 import assess
from remote import connection


ROOT = "/data/fangc/teamlearn/runs/pilot8b_20260928_v3"


def remote_json(sftp, path):
    with sftp.open(path, "r") as handle:
        return json.loads(handle.read().decode())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--credentials-pdf", default="fangc-README.pdf")
    args = parser.parse_args()
    with connection("151", args.credentials_pdf) as client:
        with client.open_sftp() as sftp:
            status = remote_json(sftp, ROOT + "/controller/status.json")
            if status.get("state") != "completed" or not status.get("gpu_released"):
                raise RuntimeError(f"E0 is not completed with released GPUs: {status.get('state')}, {status.get('gpu_released')}")
            completion = remote_json(sftp, ROOT + "/pilot/completion.json")
            source_gate = remote_json(sftp, ROOT + "/pilot/gate.json")
            with sftp.open(ROOT + "/pilot/E0/results.jsonl", "r") as handle:
                raw = handle.read().decode()
            rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
            if len(rows) != completion.get("planned"):
                raise RuntimeError(f"E0 result count differs from the planned count: {len(rows)} != {completion}")
            result = assess(rows, source_gate)
            result["e0_completion"] = completion
            local = Path("artifacts/e0_intervention_gate.json")
            local.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            destination = ROOT + "/pilot/intervention_gate.json"
            try:
                existing = remote_json(sftp, destination)
            except FileNotFoundError:
                existing = None
            if existing is not None and existing != result:
                raise ValueError("existing remote E0 intervention gate differs")
            if existing is None:
                temporary = ROOT + "/pilot/intervention_gate.tmp"
                with sftp.open(temporary, "w") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False, indent=2).encode())
                sftp.rename(temporary, destination)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
