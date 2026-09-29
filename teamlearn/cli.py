from __future__ import annotations

import argparse
import json
from pathlib import Path
from .common import read_jsonl,write_json


def main(argv=None):
    parser=argparse.ArgumentParser(prog="teamlearn",description="Controlled cross-task failure-report experiments")
    sub=parser.add_subparsers(dest="command",required=True)
    generate=sub.add_parser("generate");generate.add_argument("--out",default="datasets");generate.add_argument("--per-family",type=int,default=8)
    collect=sub.add_parser("collect")
    run=sub.add_parser("run")
    stream=sub.add_parser("stream")
    for p in (collect,run,stream):
        p.add_argument("--dataset",default="datasets");p.add_argument("--sources",default="runs/sources")
        p.add_argument("--models",default="configs/models.yaml");p.add_argument("--model",default="qwen3_8b")
        p.add_argument("--out",required=True);p.add_argument("--rounds",type=int,default=2);p.add_argument("--turns",type=int,default=7)
    for p in (collect,run):
        p.add_argument("--split",choices=["train","dev","test"],default="dev");p.add_argument("--limit",type=int)
        p.add_argument("--seeds",default="11,29,47" if p is collect else "101,211,307")
    run.add_argument("--experiment",choices=["E0","E1","E2","E3","E4","E5","E7","TRAIN","BASELINES"],required=True)
    run.add_argument("--policy");run.add_argument("--arms");run.add_argument("--diagnostic",action="store_true")
    run.add_argument("--heterogeneous",action="store_true");run.add_argument("--freeze")
    stream.add_argument("--steps",type=int,default=50);stream.add_argument("--streams",type=int,default=5)
    stream.add_argument("--probe-every",type=int,default=10);stream.add_argument("--policy")
    fit=sub.add_parser("fit-policy");fit.add_argument("--input",required=True);fit.add_argument("--out",required=True)
    freeze=sub.add_parser("freeze");freeze.add_argument("--dataset",default="datasets");freeze.add_argument("--models",default="configs/models.yaml");freeze.add_argument("--out",default="protocol/frozen.json")
    freeze.add_argument("--policy")
    analyze=sub.add_parser("analyze");analyze.add_argument("--input",nargs="+",required=True);analyze.add_argument("--out",required=True);analyze.add_argument("--allow-mock",action="store_true")
    plan=sub.add_parser("plan");plan.add_argument("--out",default="artifacts/experiment_plan.json")
    doctor=sub.add_parser("doctor");doctor.add_argument("--models",default="configs/models.yaml");doctor.add_argument("--model",default="mock")
    args=parser.parse_args(argv)
    if args.command=="generate":
        from .datasets import generate
        result=generate(args.out,args.per_family)
    elif args.command=="collect":
        from .pipeline import collect
        result=collect(args.dataset,args.out,args.split,args.models,args.model,args.limit,tuple(map(int,args.seeds.split(","))),args.rounds,args.turns)
    elif args.command=="run":
        from .pipeline import run_experiment
        if args.experiment=="TRAIN" and args.split!="train":parser.error("TRAIN must use --split train")
        result=run_experiment(args.dataset,args.sources,args.out,args.experiment,args.split,args.models,args.model,args.limit,tuple(map(int,args.seeds.split(","))),
                              args.policy,args.arms.split(",") if args.arms else None,args.rounds,args.turns,args.diagnostic,args.heterogeneous,args.freeze)
    elif args.command=="stream":
        from .stream import run_stream
        result=run_stream(args.dataset,args.sources,args.out,args.models,args.model,args.steps,args.streams,args.probe_every,args.policy,args.rounds,args.turns)
    elif args.command=="fit-policy":
        from .policy import Policy
        policy=Policy.fit(read_jsonl(args.input));policy.save(args.out);result=policy.metadata
    elif args.command=="freeze":
        from .pipeline import freeze_protocol
        result=freeze_protocol(args.out,args.dataset,args.models,args.policy)
    elif args.command=="analyze":
        from .analysis import summarize
        result=summarize(args.input,args.out,args.allow_mock)
    elif args.command=="plan":
        from .experiments import arms,target_kinds
        result={e:{"arms":arms(e),"target_kinds":target_kinds(e),"runs_per_source_at_3_seeds":len(arms(e))*len(target_kinds(e))*3} for e in ("E0","E1","E2","E3","E4","E5","E7","TRAIN","BASELINES")}
        result["E6"]={"groups":["L0","L1","L2","L3"],"default_streams":5,"default_steps":50,"read_only_probes":True}
        write_json(args.out,result)
    elif args.command=="doctor":
        from .pipeline import load_models
        models,ledger,config=load_models(args.models,args.model)
        try:
            model=models["coordinator"]
            result={"model":model.name,"tokenizer_test":model.tokens("Check the current contract before using an old report."),"reply":model.complete([{"role":"user","content":"Reply with exactly {\"ready\":true}"}],1,phase="doctor",max_tokens=32),"usage":ledger}
        finally:
            for model in models.values():model.close()
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
