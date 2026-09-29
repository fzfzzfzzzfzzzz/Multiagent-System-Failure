from __future__ import annotations

import copy
import random
import time
from pathlib import Path
import yaml
from .common import ROLES,append_jsonl,digest,read_json,read_jsonl,write_json
from .experiments import arms,target_kinds
from .model import make_model
from .policy import Policy
from .reports import Memory,build_deliveries,freeze_experience,evidence_index
from .datasets import verify_dataset
from .runner import run_team


def load_models(path,key,heterogeneous=False):
    config=yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    ledger=[]
    keys={r:key for r in ROLES}
    if heterogeneous:
        keys={"coordinator":"qwen38_27b","provider":"qwen3_8b","analyst":"qwen3_14b","executor":"qwen3_8b"}
    models={r:make_model(config["models"][k],ledger) for r,k in keys.items()}
    return models,ledger,{r:config["models"][k] for r,k in keys.items()}


def visible_feedback(events):
    return {"transport_committed":any(e["action"].get("tool")=="commit" and e["observation"].get("transport_accepted") for e in events),
            "tool_error_rate":sum(bool(e["observation"].get("error")) for e in events)/max(1,len(events))}


def collect(dataset,out,split,model_path,model_key,limit=None,seeds=(11,29,47),rounds=2,turns=7,meta_ids=None):
    dataset,out=Path(dataset),Path(out)
    verify_dataset(dataset)
    rows=read_jsonl(dataset/f"{split}_manifest.jsonl")
    if meta_ids is not None:
        rows=[r for r in rows if r['meta_id'] in meta_ids]
        if {r['meta_id'] for r in rows}!=set(meta_ids):raise ValueError('unknown or duplicate candidate IDs')
    if limit is not None:rows=rows[:limit]
    models,ledger,config=load_models(model_path,model_key)
    mock=model_key=="mock"
    count=0
    try:
        for row in rows:
            destination=out/"sources"/f'{row["meta_id"]}.json'
            if destination.exists():
                saved=read_json(destination)
                if saved["model_config"]!=config or saved.get("code_hash")!=code_hash():raise ValueError("source cache model/code mismatch; choose another output directory")
                if saved.get("selected") and not saved.get("experience"):
                    saved["experience"]=freeze_experience(saved["selected"],models["coordinator"],seed=seeds[0])
                    saved["experience"]["collection_split"]=split
                    saved["experience"]["frozen_hash"]=digest({k:v for k,v in saved["experience"].items() if k!="frozen_hash"})
                    write_json(destination,saved)
                continue
            attempts=[]
            selected=None
            for seed in seeds:
                begin=len(ledger)
                episode=run_team(dataset/"private"/f'{row["source_id"]}.json',models,seed=seed,rounds=rounds,turns_per_role=turns)
                episode["visible_feedback"]=visible_feedback(episode["events"])
                write_json(out/"source_attempts"/f'{row["meta_id"]}-{seed}.json',episode)
                attempt={"seed":seed,"first_success":episode["score"]["first_success"],"recurrence":episode["score"]["recurrence"],"abstained":episode["score"]["abstained"],"cost":copy.deepcopy(ledger[begin:])}
                attempts.append(attempt)
                if episode["score"]["recurrence"]:
                    selected=episode;break
            record={**row,"mock":mock,"model_config":config,"attempts":attempts,"selected":selected,
                    "status":"failure" if selected else "no_on_mechanism_failure_in_fixed_attempts","source_selection":"first observed intended-mechanism failure in preregistered seeds; off-mechanism failure and abstention are not relabeled",
                    "code_hash":code_hash(),
                    "dataset_hash":read_json(dataset/"manifest.json")["hash"]}
            if selected:
                write_json(destination,record)  # source is durable before report generation
                record["experience"]=freeze_experience(selected,models["coordinator"],seed=seeds[0])
                record["experience"]["collection_split"]=split
                # Include added provenance in the immutable hash.
                record["experience"]["frozen_hash"]=digest({k:v for k,v in record["experience"].items() if k!="frozen_hash"})
            write_json(destination,record)
            append_jsonl(out/"source_collection.jsonl",{k:v for k,v in record.items() if k not in ("selected","experience")})
            count+=1
            print(f"source {row['meta_id']}: {record['status']}",flush=True)
    finally:
        for model in models.values():model.close()
    return {"collected":count,"mock":mock}


def source_records(root,split=None):
    records=[read_json(p) for p in sorted((Path(root)/"sources").glob("*.json"))]
    return [r for r in records if r["status"]=="failure" and r.get("experience") and (split is None or r["split"]==split)]


def run_experiment(dataset,sources,out,experiment,split,model_path,model_key,limit=None,seeds=(101,211,307),policy_path=None,
                   selected_arms=None,rounds=2,turns=7,diagnostic=False,heterogeneous=False,freeze_path=None):
    dataset,sources,out=Path(dataset),Path(sources),Path(out)
    dataset_hash=verify_dataset(dataset)
    records=source_records(sources,split)
    if experiment=="E7":records=[r for r in records if r["backend"]=="sqlite"]
    elif experiment!="BASELINES":records=[r for r in records if r["backend"]=="workflow"]
    if limit is not None:records=records[:limit]
    if not records:raise ValueError("no eligible source failures; collect sources first")
    if any(r["dataset_hash"]!=dataset_hash for r in records):raise ValueError("source collection used a different dataset")
    if any(r["mock"] for r in records) and model_key!="mock":raise ValueError("mock sources cannot enter real experiments")
    chosen=arms(experiment)
    if selected_arms:chosen=[a for a in chosen if a["name"] in selected_arms]
    if not chosen:raise ValueError("no matching arms")
    if any(a["method"]=="learned" for a in chosen) and not policy_path:raise ValueError("fit a policy from TRAIN outcomes before learned arms")
    policy=Policy.load(policy_path) if policy_path else None
    memory=Memory([r["experience"] for r in source_records(sources,"train")])
    if any(a.get("memory_mode")=="train_only" for a in chosen) and not memory.records:raise ValueError("train-only arm requires train experience library")
    models,ledger,model_config=load_models(model_path,model_key,heterogeneous)
    config={"experiment":experiment,"split":split,"arms":chosen,"seeds":list(seeds),"rounds":rounds,"turns":turns,
            "code_hash":code_hash(),
            "diagnostic":diagnostic,"models":model_config,"dataset_hash":read_json(dataset/"manifest.json")["hash"],
            "sources":[[r["meta_id"],r["experience"]["frozen_hash"]] for r in records],"policy":policy.metadata if policy else None,"mock":model_key=="mock"}
    if split=="test" and model_key!="mock":
        if not freeze_path:raise ValueError("test execution requires --freeze (created from development protocol)")
        verify_freeze(freeze_path,dataset,model_path)
        frozen=read_json(freeze_path)
        if list(seeds)!=frozen["seeds"] or rounds!=frozen["rounds"] or turns!=frozen["turns"]:raise ValueError("test inference budget/seeds differ from freeze")
        if policy and frozen.get("policy_hash")!=digest({"weights":policy.weights.tolist(),"metadata":policy.metadata}):raise ValueError("policy changed or was not frozen")
    config_hash=digest(config)
    manifest_path=out/"manifest.json"
    if manifest_path.exists() and read_json(manifest_path)["config_hash"]!=config_hash:raise ValueError("resume configuration differs; choose a fresh output directory")
    write_json(manifest_path,{**config,"config_hash":config_hash})
    done={r["run_id"] for r in read_jsonl(out/"results.jsonl")} if (out/"results.jsonl").exists() else set()
    jobs=[(record,kind,seed,arm) for record in records for kind in target_kinds(experiment) for seed in seeds for arm in chosen]
    random.Random(9467).shuffle(jobs)
    completed=0
    try:
        for record,kind,seed,arm in jobs:
            run_id=digest([config_hash,record["meta_id"],kind,seed,arm["name"]])[:24]
            if run_id in done:continue
            # Only public next-task information is available to the report selector.
            from .runner import EnvironmentClient
            task_path=dataset/"private"/f'{record["targets"][kind]}.json'
            worker=EnvironmentClient(task_path)
            try:public=worker.request("view",role="coordinator")["task"]
            finally:worker.close()
            experience=record["experience"]
            if arm.get("unrelated"):
                options=[r for r in records if r["mechanism"]!=record["mechanism"]]
                if not options:raise ValueError("unrelated control needs another mechanism")
                experience=options[0]["experience"]
            local_experiences=None
            local_cost=[]
            if arm["method"]=="independent":
                local_experiences={}
                for role in ROLES:
                    cache=out/"local_reflections"/f'{record["meta_id"]}-{role}.json'
                    if cache.exists():local=read_json(cache)
                    else:
                        own=[e for e in record["selected"]["events"] if e["role"]==role or (e["action"].get("tool")=="message" and e["action"].get("args",{}).get("to")==role)]
                        local=freeze_experience({"events":own,"agents":record["selected"]["agents"],"public_task":record["selected"]["public_task"]},models[role],seed=17)
                        write_json(cache,local)
                    local_experiences[role]=local;local_cost+=local["report_cost"]
            routing_start=time.monotonic()
            reports,audit=build_deliveries(experience,arm,models,public,seed,memory if arm.get("memory_mode")=="train_only" else Memory([experience]),policy,local_experiences)
            audit["routing_seconds"]=time.monotonic()-routing_start
            begin=len(ledger);start=time.monotonic()
            try:
                # Ablations cannot restore deleted content through evidence fetch.
                available=evidence_index(memory.records if arm.get("memory_mode")=="train_only" else [experience]) if arm.get("pointers") else {}
                episode=run_team(task_path,models,reports,seed,rounds,turns,not diagnostic,available)
            except Exception as error:
                append_jsonl(out/"errors.jsonl",{"run_id":run_id,"error":repr(error),"cost":ledger[begin:]})
                raise
            cost=copy.deepcopy(ledger[begin:])
            result={"run_id":run_id,"config_hash":config_hash,"experiment":experiment,"method":arm["name"],"arm":arm,
                    "meta_id":record["meta_id"],"family_id":record["family_id"],"mechanism":record["mechanism"],"target_id":record["targets"][kind],
                    "kind":kind,"seed":seed,"split":split,"backend":record["backend"],"mock":model_key=="mock",
                    **episode["score"],"report_audit":audit,"cost":cost,"source_cost":record["attempts"],"report_construction_cost":local_cost if local_experiences else ([] if arm["method"] in ("none","history","success") else experience["report_cost"]),
                    "V_available":bool(experience["components"]["V"]),
                    "report_read_tokens_actual":episode["report_read_tokens_actual"],
                    "seconds":time.monotonic()-start,"source_hash":experience["source_hash"],"policy_version":digest(policy.metadata) if policy else "fixed-v1"}
            write_json(out/"episodes"/f"{run_id}.json",{**episode,"reports":reports,"audit":audit,"cost":cost})
            append_jsonl(out/"results.jsonl",result)
            if experiment=="TRAIN":
                append_jsonl(out/"policy_training.jsonl",{"split":split,"mock":model_key=="mock","family_id":record["family_id"],
                             "experience":experience,"recipients":arm["recipients"],"budget":arm["budget"],
                             "report_tokens":audit["report_tokens"],"first_success":result["first_success"],"recurrence":result["recurrence"],
                             "training_tokens":sum(c.get("input_tokens",0)+c.get("output_tokens",0) for c in cost)})
            completed+=1
            print(f"{experiment} {completed}/{len(jobs)-len(done)} {arm['name']} {kind}: first={result['first_success']} recurrence={result['recurrence']}",flush=True)
    finally:
        for model in models.values():model.close()
    return {"completed":completed,"planned":len(jobs),"mock":model_key=="mock","out":str(out)}


def freeze_protocol(path,dataset,models,policy_path=None,rounds=2,turns=7,seeds=(101,211,307)):
    verify_dataset(dataset)
    payload={"dataset_hash":read_json(Path(dataset)/"manifest.json")["hash"],"models_hash":digest(Path(models).read_text(encoding="utf-8")),
             "arms":{e:arms(e) for e in ("E0","E1","E2","E3","E4","E5","E7","TRAIN","BASELINES")},
             "code_hash":code_hash(),"seeds":list(seeds),"rounds":rounds,"turns":turns,"primary":["H1","H4"],"timestamp":time.time(),
             "policy_hash":digest(read_json(policy_path)) if policy_path else None}
    write_json(path,payload);return payload


def code_hash():return digest({str(p):p.read_text(encoding="utf-8") for p in sorted(Path("teamlearn").glob("*.py"))})


def verify_freeze(path,dataset,models):
    value=read_json(path)
    verify_dataset(dataset)
    if value["dataset_hash"]!=read_json(Path(dataset)/"manifest.json")["hash"]:raise ValueError("dataset changed since freeze")
    if value["models_hash"]!=digest(Path(models).read_text(encoding="utf-8")):raise ValueError("models changed since freeze")
    if value["code_hash"]!=code_hash():raise ValueError("code changed since freeze")
