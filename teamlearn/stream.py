"""Chronological L0/L1/L2/L3 evaluation with non-writing held-out probes."""
from __future__ import annotations

import copy
import random
from pathlib import Path
import numpy as np
from .common import ROLES,append_jsonl,digest,write_json
from .experiments import arm
from .pipeline import load_models,source_records,visible_feedback
from .policy import Policy,features
from .reports import Memory,build_deliveries,freeze_experience
from .runner import EnvironmentClient,run_team


def run_stream(dataset,sources,out,model_path,model_key,steps=50,streams=5,probe_every=10,policy_path=None,rounds=2,turns=7):
    records=source_records(sources,"train")
    probes=source_records(sources,"dev")
    if not records or not probes:raise ValueError("stream requires independently collected train sources and dev probes")
    if model_key!="mock" and any(r["mock"] for r in records+probes):raise ValueError("mock source cannot enter real stream")
    out=Path(out)
    if (out/"results.jsonl").exists():raise ValueError("stream output exists; use a fresh directory (atomic stream resume is not enabled)")
    models,ledger,config=load_models(model_path,model_key)
    initial=Policy.load(policy_path) if policy_path else Policy(np.zeros(len(features(records[0]["experience"],[],2048))),{"origin":"zero prior; public-feedback online updates"})
    write_json(out/"manifest.json",{"experiment":"E6","streams":streams,"steps":steps,"probe_every":probe_every,"models":config,"mock":model_key=="mock",
               "online_reward":"public transport_committed - tool_error_rate; hidden correctness never used"})
    try:
        for stream_id in range(streams):
            sequence=records[:];random.Random(stream_id+591).shuffle(sequence)
            states={label:{"memory":Memory([sequence[0]["experience"]]),"policy":copy.deepcopy(initial)} for label in ("L0","L1","L2","L3")}
            for index in range(steps):
                record=sequence[index%len(sequence)]
                kind="same" if index<steps*0.6 else ("reversal" if index%2 else "transfer")
                task_path=Path(dataset)/"private"/f'{record["targets"][kind]}.json'
                worker=EnvironmentClient(task_path)
                try:public=worker.request("view",role="coordinator")["task"]
                finally:worker.close()
                external=sequence[(index+1)%len(sequence)]["experience"]
                for label,state in states.items():
                    experience=state["memory"].retrieve(public,limit=1)[0]
                    treatment=arm(label,"learned" if label in ("L2","L3") else "role_rag")
                    delivered,audit=build_deliveries(experience,treatment,models,public,index,state["memory"],state["policy"])
                    before=len(ledger)
                    result=run_team(task_path,models,delivered,stream_id*10000+index,rounds,turns)
                    # Update only AFTER acting. L1 and L2 see byte-identical external libraries.
                    feedback=visible_feedback(result["events"])
                    if label in ("L2","L3"):
                        recipients=[r for r,text in delivered.items() if text]
                        state["policy"].update_visible(experience,recipients,2048,feedback)
                    if label in ("L1","L2"):state["memory"].add(external)
                    if label=="L3":
                        result["visible_feedback"]=feedback
                        new=freeze_experience(result,models["coordinator"],index)
                        state["memory"].add(new)
                    append_jsonl(out/"results.jsonl",{"stream":stream_id,"step":index,"group":label,"kind":kind,"source_id":record["meta_id"],"mock":model_key=="mock",
                                 **result["score"],"feedback":feedback,"cost":ledger[before:],"memory_hash":digest(state["memory"].records),"memory_size":len(state["memory"].records)})
                if digest(states["L1"]["memory"].records)!=digest(states["L2"]["memory"].records):raise AssertionError("L1/L2 libraries diverged")
                if (index+1)%probe_every==0:
                    for label,state in states.items():
                        before_hash=digest([state["memory"].records,state["policy"].weights.tolist()])
                        write_json(out/"snapshots"/f"s{stream_id}-t{index}-{label}.json",{"memory":state["memory"].records,"policy":state["policy"].weights.tolist()})
                        # Probe removal and no-K interventions operate on frozen copies.
                        probe=probes[(index//probe_every+stream_id)%len(probes)]
                        for intervention in ("normal","remove_report","no_K"):
                            path=Path(dataset)/"private"/f'{probe["targets"]["same"]}.json'
                            worker=EnvironmentClient(path)
                            try:public=worker.request("view",role="coordinator")["task"]
                            finally:worker.close()
                            exp=state["memory"].retrieve(public,limit=1)[0]
                            treatment=arm(label,"none" if intervention=="remove_report" else ("learned" if label in ("L2","L3") else "role_rag"),components="SOCDPVU" if intervention=="no_K" else "SOCDKPVU")
                            reports,audit=build_deliveries(exp,treatment,models,public,index,state["memory"],state["policy"])
                            result=run_team(path,models,reports,stream_id*10000+index,rounds,turns)
                            append_jsonl(out/"probes.jsonl",{"stream":stream_id,"step":index,"group":label,"intervention":intervention,"mock":model_key=="mock",**result["score"]})
                        if digest([state["memory"].records,state["policy"].weights.tolist()])!=before_hash:raise AssertionError("probe mutated learning state")
                print(f"stream {stream_id+1}/{streams}, step {index+1}/{steps}",flush=True)
    finally:
        for model in models.values():model.close()
    return {"streams":streams,"steps":steps,"out":str(out)}
