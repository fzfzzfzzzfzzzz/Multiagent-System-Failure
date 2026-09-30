from __future__ import annotations

import copy
import random
import re
from .common import COMPONENTS, ROLES, canonical, digest
from .model import parse_object

REPORT_PROTOCOL_VERSION="strict-v2"
REPORT_SYSTEM = """Create transferable historical lessons ONLY from the provided source trajectory. You have no target task. Return the requested JSON schema with components C,D,K,P,U, predicted_mechanism, culprits, and related_roles. Each component is empty or contains ONE atom with text, source event refs, and semantic roles. C is a possible cause; D is a consumed dependency; K states the exact applicability condition AND exception; P is an observable prevention action; U states uncertainty. Use at most 25 words per atom and at most three refs. Do not repeat the same idea across fields, infer unique blame without evidence, expose concrete source values as future answers, or invent verified repairs. Distinguish tool transport acceptance from task correctness. S/O/V are constructed separately."""
ROLE_LIST={"type":"array","items":{"type":"string","enum":list(ROLES)},"maxItems":4}
ATOM_SCHEMA={"type":"object","properties":{"text":{"type":"string","maxLength":240},"refs":{"type":"array","items":{"type":"string"},"minItems":1,"maxItems":3},"roles":ROLE_LIST},"required":["text","refs","roles"],"additionalProperties":False}
REPORT_SCHEMA={"type":"object","properties":{
    "components":{"type":"object","properties":{c:{"type":"array","items":ATOM_SCHEMA,"maxItems":1} for c in "CDKPU"},"required":list("CDKPU"),"additionalProperties":False},
    "predicted_mechanism":{"type":"string","enum":[f"F{i}" for i in range(1,7)]+["unknown"]},"culprits":ROLE_LIST,"related_roles":ROLE_LIST},
    "required":["components","predicted_mechanism","culprits","related_roles"],"additionalProperties":False}


def freeze_experience(episode, reporter, seed=0):
    events = copy.deepcopy(episode["events"])
    evidence = {e["event_id"]:e for e in events}
    payload = {"events":events,"agents":episode["agents"],"source_task":episode["public_task"],
               "visible_feedback":episode.get("visible_feedback",{})}
    start = len(reporter.ledger)
    reporter_attempts=[]
    generated=None
    text=""
    for attempt in range(2):
        text = reporter.complete([{"role":"system","content":REPORT_SYSTEM},{"role":"user","content":canonical(payload)}],seed+attempt*100003,phase="report",max_tokens=1400,schema=REPORT_SCHEMA)
        try:
            generated=parse_object(text)
            reporter_attempts.append({"attempt":attempt,"seed":seed+attempt*100003,"valid_json":True,"output":text})
            break
        except (ValueError, TypeError) as error:
            reporter_attempts.append({"attempt":attempt,"seed":seed+attempt*100003,"valid_json":False,"error":f"{type(error).__name__}: {error}","output":text})
    if generated is None:
        raise ValueError("reporter produced invalid JSON in both fixed attempts")
    if not isinstance(generated.get("components"),dict) or not all(c in generated["components"] for c in "CDKPU"):
        raise ValueError("reporter schema invalid: missing nested components")
    if not all(isinstance(r,str) and r in ROLES for r in generated.get("culprits",[])+generated.get("related_roles",[])):
        raise ValueError("reporter schema invalid: recipients must be semantic role strings")
    components = {c:[] for c in COMPONENTS}
    # Literal observations, not LLM statements masquerading as verified facts.
    last_by_tool={e["action"].get("tool"):e["event_id"] for e in events}
    for event in events:
        tool=event["action"].get("tool")
        if tool in ("finish","contract","status","read_artifact","read_records"):continue
        if tool=="message":continue  # D captures dependency hypotheses with exact message refs.
        atom={"text":canonical({"role":event["role"],"tool":tool,"observation":event["observation"]}),
              "refs":[event["event_id"]],"status":"observed","roles":[event["role"]]}
        if event["observation"].get("error") or (tool in ("publish","calculate","commit") and last_by_tool[tool]==event["event_id"]):components["O"].append(atom)
        if event["observation"].get("local_feasible") or event["observation"].get("local_numeric_schema_valid"):
            if last_by_tool[tool]==event["event_id"]:
                components["S"].append({**atom,"text":f"{event['role']}: {tool} passed its local schema/feasibility check; this does not certify global success."})
    statuses={"C":"unverified_hypothesis","D":"dependency_hypothesis","K":"applicability_condition","P":"proposed_prevention","U":"uncertainty_note"}
    for c,status in statuses.items():
        for atom in generated.get("components",{}).get(c,[]):
            refs=atom.get("refs",[])
            if not refs or not all(r in evidence for r in refs):raise ValueError(f"unsupported evidence reference in {c}")
            roles=atom.get("roles",[])
            if not all(r in ROLES for r in roles):raise ValueError("unknown role in report")
            components[c].append({"text":str(atom["text"]),"refs":refs,"status":status,"roles":roles})
    # V is empty unless an actual separately verified repair is supplied.
    if episode.get("verified_repair"):
        raise ValueError("verified repairs require explicit provenance import; never infer V from source success")
    for c,atoms in components.items():
        for index,atom in enumerate(atoms):atom["atom_id"]=f"{c}{index:03d}"
    related=[r for r in generated.get("related_roles",[]) if r in ROLES]
    culprits=[r for r in generated.get("culprits",[]) if r in ROLES]
    predicted=generated.get("predicted_mechanism","unknown")
    if predicted not in [f"F{i}" for i in range(1,7)]:predicted="unknown"
    record={"experience_id":digest(events)[:20],"source_hash":digest(events),"components":components,"evidence":evidence,
            "predicted_mechanism":predicted,"culprits":culprits,"related_roles":related,"source_agents":episode["agents"],
            "query":episode["public_task"],"version":1,"supersedes":[],"status":"active","reporter_output":text,
            "reporter_attempts":reporter_attempts,"report_protocol_version":REPORT_PROTOCOL_VERSION,
            "report_cost":copy.deepcopy(reporter.ledger[start:]),"provenance_validation":"references_exist; literal O/S; C/D/K/P are unverified inferences"}
    record["frozen_hash"]=digest(record)
    return record


def validate_frozen(record):
    payload={k:v for k,v in record.items() if k!="frozen_hash"}
    if digest(payload)!=record["frozen_hash"]:raise ValueError("frozen evidence was modified")
    for atoms in record["components"].values():
        for atom in atoms:
            if not all(ref in record["evidence"] for ref in atom["refs"]):raise ValueError("dangling evidence reference")


def words(text):return set(re.findall(r"[a-z0-9_]+",text.lower()))


def namespace_atom(record,atom):
    prefix=record["experience_id"]
    return {**atom,"atom_id":prefix+":"+atom["atom_id"],"refs":[prefix+"/"+r for r in atom["refs"]]}


def evidence_index(records):
    return {r["experience_id"]+"/"+ref:{**event,"event_id":r["experience_id"]+"/"+ref} for r in records for ref,event in r["evidence"].items()}


class Memory:
    """Shared candidate store with explicit query -> insight / interaction traversal.

    This is a documented G-Memory-inspired adaptation, not the original method.
    """
    def __init__(self,records=(),capacity=128):
        self.capacity=capacity
        self.records=[]
        for record in records:self.add(record)

    def add(self,record):
        validate_frozen(record)
        if any(r["experience_id"]==record["experience_id"] for r in self.records):return
        superseded=set(record.get("supersedes",[]))
        self.records=[r for r in self.records if r["experience_id"] not in superseded]
        self.records.append(copy.deepcopy(record))
        self.records=self.records[-self.capacity:]

    def retrieve(self,query,role=None,limit=3):
        q=words(canonical(query)+(role or ""))
        def score(record):
            text=canonical(record["query"])+canonical(record["components"]["K"]+record["components"]["P"])
            keys=words(text)
            return len(q&keys)/max(1,len(q|keys))
        return sorted((r for r in self.records if r.get("status")=="active"),key=lambda r:(-score(r),r["experience_id"]))[:limit]

    def hierarchy(self,query,role,limit=3):
        """Query graph expansion -> shared insights -> role-visible interactions."""
        direct=self.retrieve(query,role,limit=max(1,limit-1))
        if not direct:return []
        seen={r["experience_id"] for r in direct}
        # One query-graph neighbor selected by overlap of generalized insights.
        basis=words(canonical(direct[0]["components"]["P"]+direct[0]["components"]["K"]))
        neighbors=[r for r in self.records if r["experience_id"] not in seen]
        if neighbors:
            direct.append(max(neighbors,key=lambda r:len(basis&words(canonical(r["components"]["P"]+r["components"]["K"])))))
        selected=[]
        for record in direct[:limit]:
            relevant_refs=set()
            for c in "SPKCDU":
                for atom in record["components"][c]:
                    if not atom["roles"] or role in atom["roles"]:
                        selected.append(namespace_atom(record,atom))
                        relevant_refs.update(atom["refs"])
            for atom in record["components"]["O"]:
                if role in atom["roles"] or relevant_refs.intersection(atom["refs"]):
                    selected.append(namespace_atom(record,atom))
        return selected


def render(atoms,style="structured",pointers=False):
    if not atoms:return ""
    if style=="flat":return "\n".join(f'{a["atom_id"]}: {a["text"]} [{a["status"]}; refs={",".join(a["refs"])}]' for a in atoms)
    fields=("atom_id","text","refs","status")
    return canonical([{k:a[k] for k in fields} for a in atoms])


def fit_atoms(atoms,model,budget,style="structured"):
    chosen=[]
    for atom in atoms:
        if model.tokens(render(chosen+[atom],style))<=budget:chosen.append(atom)
    return chosen


def build_deliveries(experience,arm,models,public_task,seed=0,memory=None,policy=None,local_experiences=None):
    validate_frozen(experience)
    method=arm.get("method","uniform")
    components=arm.get("components","SOCDKPVU")
    if method=="none":components=""
    elif method=="success":components="S"
    elif method=="rule":components="SP"
    records=[experience]
    if memory is not None and arm.get("memory_mode")=="train_only":
        records=memory.retrieve(public_task,limit=arm.get("retrieval_k",3))
    # S remains the same prefix across content ablations and all non-B0 methods.
    ordered=[c for c in "SPKDCOUV" if c in components]
    atoms=[namespace_atom(r,a) for r in records for c in ordered for a in r["components"][c]]
    recipients=list(ROLES)
    if method=="culprit":recipients=experience["culprits"]
    elif method=="coordinator":recipients=["coordinator"]
    elif method in ("mechanism","learned"):recipients=experience["related_roles"] or ["coordinator"]
    elif method=="subset":recipients=arm["recipients"]
    # Diagnostic experiments may hold report content fixed while assigning a
    # pre-registered recipient set.  This override is recorded in the audit.
    if "recipients" in arm:
        recipients=list(arm["recipients"])
        if not recipients or any(role not in ROLES for role in recipients):
            raise ValueError("recipients must be a non-empty list of semantic roles")
    if method=="learned":
        if policy is None:raise ValueError("learned reporting requires a frozen trained policy")
        recipients=policy.select(experience,arm.get("budget",2048))
    if arm.get("recipient_mode")=="fixed":recipients=["provider","executor"]
    if arm.get("recipient_mode")=="related":recipients=experience["related_roles"] or ["coordinator"]
    if arm.get("bind_old_ids"):
        old={experience["source_agents"][r] for r in recipients}
        recipients=[r for r in ROLES if public_task["agents"][r] in old]
    by_role={r:[] for r in ROLES}
    for role in recipients:
        selected=atoms[:]
        if method=="gmemory_adapt":
            selected=(memory or Memory(records)).hierarchy(public_task,role,arm.get("retrieval_k",3))
            selected=[a for a in selected if a["atom_id"].split(":")[-1][0] in components]
        elif method=="role_rag":
            retrieved=(memory or Memory(records)).retrieve(public_task,role,arm.get("retrieval_k",3))
            selected=[]
            for record in retrieved:
                cs=components if method=="role_rag" else "SPKCDOU"
                for c in cs:
                    for atom in record["components"][c]:
                        if not atom["roles"] or role in atom["roles"]:
                            selected.append(namespace_atom(record,atom))
        elif method=="independent":
            if local_experiences is None:raise ValueError("independent reflection needs separately generated local experiences")
            local=local_experiences[role]
            selected=[namespace_atom(local,a) for c in ordered for a in local["components"][c]]
        elif arm.get("role_views"):
            selected=[a for a in atoms if not a["roles"] or role in a["roles"] or a["atom_id"].split(":")[-1].startswith("S")]
        if arm.get("pointers"):
            selected=[{**a,"text":"Evidence available by fetch_evidence(event_id)."} if a["atom_id"].split(":")[-1].startswith("O") else a for a in selected]
        by_role[role]=selected
    style=arm.get("style","structured")
    budget=arm.get("budget",2048)
    mode=arm.get("budget_mode","team_total")
    if method=="history":
        raw=canonical(list(experience["evidence"].values()))
        reports={r:raw for r in ROLES}
        return reports,{"high_resource_reference":True,"report_tokens":sum(models[r].tokens(raw) for r in ROLES),"atomic_ids":{}}
    if mode=="content_set":
        # Fit once with a common tokenizer and per-role buckets BEFORE permutation.
        allowance=budget//max(1,len(recipients))
        # Use the union cost of BOTH formats so format controls retain exactly
        # the same atom IDs even though serialization token lengths differ.
        def fit_equivalent(items):
            chosen=[]
            for item in items:
                candidate=chosen+[item]
                if max(models[ROLES[0]].tokens(render(candidate,s)) for s in ("flat","structured"))<=allowance:chosen=candidate
            return chosen
        by_role={r:fit_equivalent(a) for r,a in by_role.items()}
    else:
        allowance=budget if mode=="per_agent" else budget//max(1,len(recipients))
        by_role={r:fit_atoms(a,models[r],allowance,style) for r,a in by_role.items()}
    before_ids={r:[a["atom_id"] for a in items] for r,items in by_role.items()}
    reports={r:render(a,style) for r,a in by_role.items()}
    mapping=list(ROLES)
    if arm.get("permutation")=="cycle":mapping=list(ROLES[1:])+[ROLES[0]]
    elif arm.get("permutation")=="random":
        random.Random(seed).shuffle(mapping)
        if mapping==list(ROLES):mapping=list(ROLES[1:])+[ROLES[0]]
    if arm.get("permutation"):
        if mode!="content_set":raise ValueError("permutation requires content_set matching")
        reports={dst:reports[src] for src,dst in zip(ROLES,mapping)}
    delivered_ids={r:list(before_ids[r]) for r in ROLES}
    if arm.get("permutation"):
        delivered_ids={dst:list(before_ids[src]) for src,dst in zip(ROLES,mapping)}
    tokens={r:models[r].tokens(text) for r,text in reports.items()}
    if mode in ("team_total","content_set") and sum(tokens.values())>budget:raise ValueError("report token budget exceeded")
    audit={"report_tokens":sum(tokens.values()),"tokens_by_role":tokens,"budget_mode":mode,"budget":budget,
           "budget_scope":"initial delivered context; repeated prefills separately metered in API usage",
           "atomic_ids":delivered_ids,"atomic_ids_before_routing":before_ids,"atomic_ids_delivered":delivered_ids,
           "permutation":dict(zip(ROLES,mapping)),"recipients":recipients,
           "components":components,"method":method,"frozen_hash":experience["frozen_hash"],"report_multiset_hash":digest(sorted(reports.values()))}
    return reports,audit
