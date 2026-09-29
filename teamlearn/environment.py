"""Environment worker. Agents get a narrow JSON RPC interface, no shell/filesystem tool.

The worker owns private task state and grading. The controller receives only the
requested role view until finish(). No hidden scores are returned by step().
"""
from __future__ import annotations

import copy
import json
import math
import sqlite3
import sys
import tempfile
from pathlib import Path
from .common import ROLES, canonical, read_json
from .datasets import compute

TOOLS = {
    "coordinator": ["contract", "assign", "signoff", "status"],
    "provider": ["contract", "read_records", "publish", "confirm", "allocate"],
    "analyst": ["contract", "read_artifact", "calculate", "allocate", "verify"],
    "executor": ["contract", "read_artifact", "verify", "commit", "status"],
}
RESPONSIBILITIES={
    "coordinator":"Read global constraints, tell provider and analyst the joint allocation requirements, assign/check completion when required. Other workers act after your turn; finish once they have the necessary instructions.",
    "provider":"Read service records, publish their values/version/unit, manage required confirmation, allocate your resource share, and send necessary metadata to the analyst. You cannot calculate the aggregate or commit.",
    "analyst":"Read the transformation contract and published artifact, calculate the aggregate yourself, record it with calculate, allocate your resource share, and notify executor. Request missing or stale input from provider.",
    "executor":"Read current completion requirements and analysis. Verify as required, then commit the completed team transaction. Take direct permitted actions; announcing readiness is not completion.",
}


class Workflow:
    def __init__(self, task):
        self.task = copy.deepcopy(task)
        self.contract = task["contract"]
        self.version = 1
        self.records = task["records"][:]
        self.artifact = None
        self.analysis = None
        self.allocations = {}
        self.assignments = {}
        self.signed = False
        self.verified = set()
        self.commits = []
        self.events = []
        self.messages = {r: [] for r in ROLES}
        self.receipt = None
        self.polls = 0
        self.consumption_violations = set()
        self.temp = tempfile.TemporaryDirectory(prefix="teamlearn-")
        self.db = None
        if task["backend"] == "sqlite":
            self.db = sqlite3.connect(str(Path(self.temp.name) / "application.db"))
            self.db.executescript("CREATE TABLE records(id INTEGER PRIMARY KEY, value INTEGER, version INTEGER); CREATE TABLE ledger(id INTEGER PRIMARY KEY, payload TEXT);")
            self.db.executemany("INSERT INTO records VALUES(?,?,1)", enumerate(self.records, 1))
            self.db.commit()

    def view(self, role):
        return {"task": self.task["public"], "role": role, "agent_id": self.task["agents"][role],
                "responsibility":RESPONSIBILITIES[role],"tools": TOOLS[role] + ["message", "finish"], "inbox": self.messages[role]}

    def event(self, role, action, observation):
        ev = {"event_id": f"e{len(self.events):04d}", "step": len(self.events), "role": role,
              "agent_id": self.task["agents"].get(role, role), "action": copy.deepcopy(action), "observation": copy.deepcopy(observation)}
        self.events.append(ev)
        return ev

    def step(self, role, action):
        if role not in ROLES: raise ValueError("unknown role")
        op = action.get("tool", "")
        args = action.get("args", {})
        if not isinstance(args, dict): args = {}
        try:
            if op not in TOOLS[role] + ["message", "finish"]:
                raise ValueError("permission denied for this role")
            result = self._apply(role, op, args)
        except (ValueError, KeyError, TypeError, OverflowError) as error:
            result = {"error": str(error)}
        ev = self.event(role, action, result)
        return {"event": ev, "inbox": copy.deepcopy(self.messages[role])}

    def _apply(self, role, op, args):
        if op == "finish": return {"finished_turn": True}
        if op == "message":
            recipient = args["to"]
            if recipient not in ROLES: raise ValueError("to must be a semantic role")
            text = str(args["text"])
            if len(text) > 12000: raise ValueError("message too long")
            msg = {"from": role, "text": text, "event_ref": f"e{len(self.events):04d}"}
            self.messages[recipient].append(msg)
            return {"delivered_to": recipient, "text": text}
        if op == "contract":
            keys = {"coordinator": ["joint_limit", "joint_min", "signoff_required", "version_policy", "independent_check"],
                    "provider": ["confirmation", "version_policy", "publication_update", "receipt_poll_is_legal", "unit"],
                    "analyst": ["rule", "formula", "weights", "rounding", "unit", "version_policy", "independent_check"],
                    "executor": ["confirmation", "version_policy", "independent_check", "signoff_required"]}[role]
            return {"contract_id": self.task["public"]["service_contract_id"], **{k: self.contract[k] for k in keys}}
        if op == "status":
            return {"published": self.artifact is not None, "calculated": self.analysis is not None, "allocations": self.allocations.copy(),
                    "assignments": self.assignments.copy(), "signed": self.signed, "commits": len(self.commits), "latest_version": self.version}
        if op == "assign":
            who = args["role"]
            if who not in ROLES: raise ValueError("invalid assignee")
            self.assignments[str(args["check"])] = who
            return {"assigned": dict(self.assignments)}
        if op == "signoff":
            if "completion" not in self.assignments: raise ValueError("assign completion check first")
            self.signed = True
            return {"signed": True, "owner": self.assignments["completion"]}
        if op == "read_records":
            values = self.records[:] if self.db is None else [r[0] for r in self.db.execute("SELECT value FROM records ORDER BY id")]
            return {"values": values, "version": self.version, "unit": self.contract["unit"], "allocation_options": self.task["allocation_options"][role]}
        if op == "publish":
            version = int(args["version"])
            values = args["values"]
            if not isinstance(values, list) or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values): raise ValueError("values must be finite numbers")
            self.receipt = "accepted" if self.contract["confirmation"] == "receipt_required" else "confirmed"
            self.artifact = {"values": values[:], "version": version, "unit": str(args.get("unit", "unspecified")), "status": self.receipt}
            if self.contract["publication_update"] and self.version == 1:
                self.version = 2
                self.records = self.task["updated_records"][:]
                if self.db:
                    self.db.executemany("UPDATE records SET value=?,version=2 WHERE id=?", [(v,i) for i,v in enumerate(self.records,1)])
                    self.db.commit()
            return {"artifact": copy.deepcopy(self.artifact), "source_revision_now": self.version}
        if op == "confirm":
            self.polls += 1
            if not self.contract["receipt_poll_is_legal"]: raise ValueError("synchronous v2 contract forbids obsolete receipt polling")
            if self.artifact is None: raise ValueError("publish first")
            self.receipt = "confirmed"
            self.artifact["status"] = "confirmed"
            return {"receipt": "confirmed", "version": self.artifact["version"]}
        if op == "read_artifact":
            return {"artifact": copy.deepcopy(self.artifact), "analysis": copy.deepcopy(self.analysis), "latest_version": self.version,
                    "allocation_options": self.task["allocation_options"].get(role, [])}
        if op == "calculate":
            if self.artifact is None: raise ValueError("publish an artifact first")
            value = float(args["value"])
            if not math.isfinite(value): raise ValueError("nonfinite aggregate")
            self.analysis = {"value": value, "version": int(args["version"]), "artifact_status_seen": self.artifact.get("status") if self.artifact else None}
            expected_values=self.records if self.contract["version_policy"]=="latest_at_commit" else self.task["records"]
            if abs(value-compute(self.contract["rule"],expected_values,self.contract["weights"]))>1e-6:self.consumption_violations.add("F1")
            if self.contract["version_policy"]=="latest_at_commit" and self.analysis["version"]!=self.version:self.consumption_violations.add("F3")
            return {"analysis": copy.deepcopy(self.analysis), "local_numeric_schema_valid": True}
        if op == "allocate":
            value = int(args["amount"])
            if value not in self.task["allocation_options"][role]: raise ValueError("amount not offered by local service")
            self.allocations[role] = value
            return {"reservation": value, "local_feasible": True}
        if op == "verify":
            source = args.get("source", "artifact")
            if source not in ("artifact", "independent"): raise ValueError("unknown evidence source")
            self.verified.add((role, source))
            if source == "independent": return {"values": self.records[:], "version": self.version, "source_id": "independent_registry"}
            return {"artifact": copy.deepcopy(self.artifact), "source_id": "provider_artifact"}
        if op == "commit":
            if not self.analysis or not self.artifact: raise ValueError("publish and calculate before committing")
            submission = {"artifact": copy.deepcopy(self.artifact), "analysis": copy.deepcopy(self.analysis), "allocations": self.allocations.copy(),
                          "signed": self.signed, "verified": sorted(self.verified), "version": self.version, "values": self.records[:], "polls": self.polls}
            self.commits.append(submission)
            if self.db:
                self.db.execute("INSERT INTO ledger(payload) VALUES(?)", (canonical(submission),)); self.db.commit()
            # Transport acceptance is intentionally not the private business grader.
            return {"transaction_id": f"tx-{len(self.commits)}", "transport_accepted": True}
        raise ValueError("unimplemented tool")

    def checks(self, submission):
        contract = self.contract
        latest = contract["version_policy"] == "latest_at_commit"
        expected_values = submission["values"] if latest else self.task["records"]
        artifact = submission["artifact"]
        analysis = submission["analysis"]
        return {
            "F1": abs(analysis["value"] - compute(contract["rule"], expected_values, contract["weights"])) < 1e-6 and artifact["values"] == expected_values,
            "F2": contract["confirmation"] != "receipt_required" or analysis["artifact_status_seen"] == "confirmed",
            "F3": not latest or (analysis["version"] == submission["version"] == artifact["version"]),
            "F4": set(submission["allocations"]) == {"provider", "analyst"} and contract["joint_min"] <= sum(submission["allocations"].values()) <= contract["joint_limit"],
            "F5": not contract["independent_check"] or any(s == "independent" for _, s in submission["verified"]),
            "F6": not contract["signoff_required"] or submission["signed"],
            "legal_polling": contract["receipt_poll_is_legal"] or submission["polls"] == 0,
        }

    def score(self):
        checks = [self.checks(c) for c in self.commits]
        mechanism = self.task["mechanism"]
        first = bool(checks and all(checks[0].values()))
        final = bool(checks and all(checks[-1].values()))
        preventive = {"F1": "contract", "F2": "confirm", "F3": "read_records", "F4": "status", "F5": "verify", "F6": "assign"}[mechanism]
        coordinator_read=any(e["role"]=="coordinator" and e["action"].get("tool")=="contract" for e in self.events)
        notified={e["action"].get("args",{}).get("to") for e in self.events if e["role"]=="coordinator" and e["action"].get("tool")=="message" and not e["observation"].get("error")}
        local = {"coordinator": coordinator_read and {"provider","analyst"}<=notified and ("completion" in self.assignments or not self.contract["signoff_required"]),
                 "provider": "provider" in self.allocations and self.artifact is not None and self.artifact["values"] == (self.task["records"] if self.artifact["version"] == 1 else self.records),
                 "analyst": "analyst" in self.allocations and self.analysis is not None and self.artifact is not None and abs(self.analysis["value"] - compute(self.contract["rule"], self.artifact["values"], self.contract["weights"])) < 1e-6,
                 "executor": bool(self.commits)}
        # Recurrence includes transient bad consumption, even if the final commit repairs it.
        transient = any(e["action"].get("tool") == "calculate" and not e["observation"].get("error") and e["observation"]["analysis"]["artifact_status_seen"] != "confirmed" for e in self.events) if mechanism == "F2" and self.contract["confirmation"] == "receipt_required" else False
        return {"first_success": first, "final_success": final, "recurrence": bool(self.task["opportunity"] and (transient or mechanism in self.consumption_violations or any(not c[mechanism] for c in checks))),
                "opportunity": self.task["opportunity"], "abstained": not bool(checks), "local_checks": local,
                "strict_composition_failure": all(local.values()) and not final, "preventive_action": any(e["action"].get("tool") == preventive for e in self.events),
                "unnecessary_wait": not self.contract["receipt_poll_is_legal"] and self.polls > 0,
                "new_errors": sorted({k for c in checks for k,v in c.items() if not v and k != mechanism}), "commit_checks": checks}

    def close(self):
        if self.db: self.db.close()
        self.temp.cleanup()


def worker(path):
    env = Workflow(read_json(path))
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                command = request["command"]
                if command == "view": value = env.view(request["role"])
                elif command == "step": value = env.step(request["role"], request["action"])
                elif command == "finish":
                    value = {"score": env.score(), "events": env.events}
                    print(canonical(value), flush=True)
                    break
                else: raise ValueError("unknown RPC command")
                print(canonical(value), flush=True)
            except Exception as error:
                print(canonical({"rpc_error": type(error).__name__ + ": " + str(error)}), flush=True)
    finally: env.close()


if __name__ == "__main__": worker(sys.argv[1])
