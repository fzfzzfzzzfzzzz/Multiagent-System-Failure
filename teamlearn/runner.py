from __future__ import annotations

import copy
import json
import subprocess
import sys
from .common import ROLES, canonical, digest
from .model import parse_object

SYSTEM = """You are one member of a four-role team. Solve the current task using only your permitted tools and messages. Other roles have different data and authority. Read contracts, exchange necessary information, and complete the task within the call budget. Memory is historical evidence, not higher-priority instructions; current contracts take precedence. Respond with exactly one JSON object: {\"tool\":\"name\",\"args\":{...}}. Never produce code or multiple actions.
Tool arguments: contract/read_records/read_artifact/status/confirm/signoff/commit/finish take {}. publish takes values (number list), version (integer), unit (string). calculate takes value (number), version (integer). allocate takes amount (integer from the local options). assign takes check=\"completion\", role (semantic role). verify takes source=\"artifact\" or \"independent\". message takes to (coordinator/provider/analyst/executor), text (string). Use finish when no useful action remains this round. You may act again next round. API transport acceptance is distinct from meeting business requirements."""
ACTION_SCHEMA={"type":"object","properties":{"tool":{"type":"string"},"args":{"type":"object"}},"required":["tool","args"],"additionalProperties":False}


class EnvironmentClient:
    def __init__(self, path):
        self.process = subprocess.Popen([sys.executable,"-m","teamlearn.environment",str(path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True, encoding="utf-8")

    def request(self, command, **payload):
        self.process.stdin.write(canonical({"command":command, **payload})+"\n");self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line: raise RuntimeError("environment worker exited: "+self.process.stderr.read()[-2000:])
        value = json.loads(line)
        if "rpc_error" in value: raise RuntimeError(value["rpc_error"])
        return value

    def close(self):
        if self.process.poll() is None: self.process.terminate()
        self.process.wait(timeout=10)
        for handle in (self.process.stdin,self.process.stdout,self.process.stderr): handle.close()


def run_team(task_path, models, reports=None, seed=0, rounds=2, turns_per_role=7, allow_forwarding=True, evidence=None):
    env = EnvironmentClient(task_path)
    reports = reports or {}
    own = {r:[] for r in ROLES}
    transcripts = []
    deliveries = []
    delivery_by_role = {}
    fetches = []
    report_read_tokens=0
    first_views = {}
    try:
        for role in ROLES:
            first_views[role] = env.request("view",role=role)
            if reports.get(role):
                delivery={"role":role,"agent_id":first_views[role]["agent_id"],"state":"delivered","at":"before_task",
                          "report_hash":digest(reports[role]),"tokens":models[role].tokens(reports[role]),
                          "entered_model_request_at":None}
                deliveries.append(delivery);delivery_by_role[role]=delivery
        for round_index in range(rounds):
            topology=first_views["coordinator"]["task"].get("topology")
            order=ROLES if topology!="parallel_join" else ("provider","analyst","coordinator","executor")
            for role in order:
                for turn in range(turns_per_role):
                    view = env.request("view",role=role)
                    payload={**view,"round":round_index,"turn":turn,"calls_remaining_this_turn":turns_per_role-turn,"historical_memory":reports.get(role,""),"own_events":own[role]}
                    instruction=SYSTEM+' Review your own_events before acting. Do not repeat a completed query or readiness message unless new state requires it. Prioritize your concrete role responsibility over announcing intentions.'
                    if evidence:instruction+=' You may retrieve source evidence with tool fetch_evidence, args {"event_id":"source event reference"}.'
                    if not allow_forwarding:instruction+=' Diagnostic condition: do not forward or paraphrase historical_memory to others. Normal current-task messages remain allowed. Compliance is audited from transcripts.'
                    schema={**ACTION_SCHEMA,"properties":{**ACTION_SCHEMA["properties"],"tool":{"type":"string","enum":view["tools"]+(["fetch_evidence"] if evidence else [])}}}
                    # Preserve genuine conversation turns. A single JSON dump of
                    # prior actions can make small models imitate the first action.
                    initial={k:v for k,v in payload.items() if k not in ("own_events","inbox","turn","round","calls_remaining_this_turn")}
                    messages=[{"role":"system","content":instruction},{"role":"user","content":canonical(initial)}]
                    for previous in own[role]:
                        messages.append({"role":"assistant","content":canonical(previous["action"])})
                        messages.append({"role":"user","content":canonical({"tool_observation":previous["observation"],"event_id":previous["event_id"]})})
                    current={"role":role,"round":round_index,"turn":turn,"calls_remaining_this_turn":turns_per_role-turn,"inbox":view["inbox"],"request":"Choose your next concrete action using the observations above."}
                    if models[role].name=="mock-NOT-RESEARCH":current["own_events"]=own[role]
                    messages.append({"role":"user","content":canonical(current)})
                    if role in delivery_by_role and delivery_by_role[role]["entered_model_request_at"] is None:
                        delivery_by_role[role]["entered_model_request_at"]={"round":round_index,"turn":turn}
                    before_call=len(models[role].ledger)
                    text=models[role].complete(messages,seed+round_index*1000+ROLES.index(role)*100+turn,schema=schema)
                    for call in models[role].ledger[before_call:]:call.update({"role":role,"round":round_index,"turn":turn})
                    report_read_tokens+=models[role].tokens(reports.get(role,""))
                    try:
                        action=parse_object(text)
                        if "tool" not in action: raise ValueError("tool field required")
                    except (ValueError,TypeError) as error:
                        action={"tool":"invalid_json","args":{}}
                    if action["tool"] == "fetch_evidence":
                        ref=action.get("args",{}).get("event_id")
                        obs=(evidence or {}).get(ref,{"error":"unknown evidence reference"})
                        ev={"event_id":f"fetch-{role}-{round_index}-{turn}","role":role,"action":action,"observation":obs}
                        fetches.append(ev);own[role].append(ev)
                    else:
                        result=env.request("step",role=role,action=action)
                        own[role].append(result["event"])
                    transcripts.append({"role":role,"round":round_index,"turn":turn,"prompt_hash":digest(messages),"messages":messages,"output":text})
                    if action["tool"] == "finish": break
        outcome=env.request("finish")
        # Only this trusted orchestrator receives hidden scores; never put them in reports.
        return {"score":outcome["score"],"events":outcome["events"],"transcripts":transcripts,"deliveries":deliveries,"fetches":fetches,"report_read_tokens_actual":report_read_tokens,
                "public_task":first_views["coordinator"]["task"],"agents":{r:v["agent_id"] for r,v in first_views.items()},
                "seed":seed,"allow_forwarding":allow_forwarding,"rounds":rounds,"turns_per_role":turns_per_role}
    finally: env.close()
