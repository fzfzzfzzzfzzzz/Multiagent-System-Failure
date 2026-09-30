"""Diagnostic policies that use only the same legal views and tools as agents.

These policies are deterministic feasibility witnesses, not research baselines.
They never receive the private task or hidden score and should not be mixed with
model experiment results.
"""
from __future__ import annotations

import json
from collections import defaultdict

from .common import canonical
from .datasets import compute


def _decoded_history(messages):
    history = []
    for index, message in enumerate(messages[:-1]):
        if message.get("role") != "assistant":
            continue
        try:
            action = json.loads(message["content"])
            following = json.loads(messages[index + 1]["content"])
            observation = following.get("tool_observation", {})
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
        history.append({"action": action, "observation": observation})
    return history


def _last(history, tool):
    for index in range(len(history) - 1, -1, -1):
        row = history[index]
        if row["action"].get("tool") == tool and not row["observation"].get("error"):
            return index, row["observation"]
    return -1, {}


def _constraint(inbox):
    for message in reversed(inbox):
        try:
            value = json.loads(message.get("text", ""))
        except (TypeError, json.JSONDecodeError):
            continue
        if "joint_min" in value and "joint_limit" in value:
            return int(value["joint_min"]), int(value["joint_limit"])
    return 0, 12


class BudgetedLegalModel:
    """A legal-observation policy used to prove budgeted workflow solvability."""

    name = "scripted-legal-diagnostic"

    def __init__(self):
        self.ledger = []

    def tokens(self, text):
        return len(text or "")

    def complete(self, messages, seed, phase="execution", max_tokens=None, schema=None):
        current = json.loads(messages[-1]["content"])
        role = current["role"]
        inbox = current.get("inbox", [])
        history = _decoded_history(messages)
        action = getattr(self, f"_{role}")(history, inbox)
        result = canonical(action)
        self.ledger.append({"model": self.name, "phase": phase, "seed": seed,
                            "input_tokens": self.tokens(canonical(messages)),
                            "output_tokens": self.tokens(result), "seconds": 0,
                            "diagnostic": True})
        return result

    def _coordinator(self, history, inbox):
        contract_index, contract = _last(history, "contract")
        if contract_index < 0:
            return {"tool": "contract", "args": {}}
        sent = {row["action"].get("args", {}).get("to") for row in history
                if row["action"].get("tool") == "message" and not row["observation"].get("error")}
        text = canonical({"joint_min": contract["joint_min"], "joint_limit": contract["joint_limit"]})
        for recipient in ("provider", "analyst"):
            if recipient not in sent:
                return {"tool": "message", "args": {"to": recipient, "text": text}}
        assign_index, _ = _last(history, "assign")
        if contract.get("signoff_required") and assign_index < 0:
            return {"tool": "assign", "args": {"check": "completion", "role": "executor"}}
        signoff_index, _ = _last(history, "signoff")
        if contract.get("signoff_required") and signoff_index < 0:
            return {"tool": "signoff", "args": {}}
        return {"tool": "finish", "args": {}}

    def _provider(self, history, inbox):
        contract_index, contract = _last(history, "contract")
        if contract_index < 0:
            return {"tool": "contract", "args": {}}
        read_index, records = _last(history, "read_records")
        if read_index < 0:
            return {"tool": "read_records", "args": {}}
        publish_index, published = _last(history, "publish")
        artifact = published.get("artifact") or {}
        if publish_index < 0 or artifact.get("version") != records.get("version"):
            return {"tool": "publish", "args": {key: records[key] for key in ("values", "version", "unit")}}
        if (contract.get("version_policy") == "latest_at_commit"
                and published.get("source_revision_now", records["version"]) > records["version"]):
            return {"tool": "read_records", "args": {}}
        confirm_index, _ = _last(history, "confirm")
        confirmed = artifact.get("status") == "confirmed" or confirm_index > publish_index
        if contract.get("confirmation") == "receipt_required" and not confirmed:
            return {"tool": "confirm", "args": {}}
        allocate_index, _ = _last(history, "allocate")
        if allocate_index < 0:
            joint_min, _ = _constraint(inbox)
            options = records["allocation_options"]
            return {"tool": "allocate", "args": {"amount": max(options) if joint_min else min(options)}}
        return {"tool": "finish", "args": {}}

    def _analyst(self, history, inbox):
        contract_index, contract = _last(history, "contract")
        if contract_index < 0:
            return {"tool": "contract", "args": {}}
        read_index, read = _last(history, "read_artifact")
        finish_index, _ = _last(history, "finish")
        if read_index < 0 or finish_index > read_index:
            return {"tool": "read_artifact", "args": {}}
        artifact = read.get("artifact")
        if not artifact:
            return {"tool": "finish", "args": {}}
        # The analyst is not allowed to see the provider's confirmation mode,
        # but the artifact status itself is in its legal view.
        if artifact.get("status") != "confirmed":
            return {"tool": "finish", "args": {}}
        if contract.get("version_policy") == "latest_at_commit" and artifact.get("version") != read.get("latest_version"):
            return {"tool": "finish", "args": {}}
        calculate_index, calculated = _last(history, "calculate")
        if calculate_index < read_index or calculated.get("analysis", {}).get("version") != artifact.get("version"):
            value = compute(contract["rule"], artifact["values"], contract["weights"])
            return {"tool": "calculate", "args": {"value": value, "version": artifact["version"]}}
        allocate_index, _ = _last(history, "allocate")
        if allocate_index < 0:
            joint_min, _ = _constraint(inbox)
            options = read["allocation_options"]
            return {"tool": "allocate", "args": {"amount": max(options) if joint_min else min(options)}}
        return {"tool": "finish", "args": {}}

    def _executor(self, history, inbox):
        contract_index, contract = _last(history, "contract")
        if contract_index < 0:
            return {"tool": "contract", "args": {}}
        commit_index, _ = _last(history, "commit")
        if commit_index >= 0:
            return {"tool": "finish", "args": {}}
        status_index, status = _last(history, "status")
        finish_index, _ = _last(history, "finish")
        if status_index < 0 or finish_index > status_index:
            return {"tool": "status", "args": {}}
        ready = (status.get("published") and status.get("calculated")
                 and set(status.get("allocations", {})) == {"provider", "analyst"}
                 and (not contract.get("signoff_required") or status.get("signed")))
        if ready:
            return {"tool": "commit", "args": {}}
        return {"tool": "finish", "args": {}}

    def close(self):
        pass


def budget_ceiling_roles(transcripts, turns_per_role):
    """Roles that used every slot in a round without choosing finish."""
    grouped = defaultdict(list)
    for row in transcripts:
        grouped[(row["role"], row["round"])].append(row)
    exhausted = set()
    for (role, _), rows in grouped.items():
        if len(rows) < turns_per_role:
            continue
        try:
            final = json.loads(rows[-1]["output"])
        except (TypeError, json.JSONDecodeError):
            final = {}
        if final.get("tool") != "finish":
            exhausted.add(role)
    return sorted(exhausted)
