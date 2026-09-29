"""Bounded live smoke: two team runs, no research score or claim.

This explicitly exercises source evidence -> frozen report -> fresh task.
It does NOT require a conveniently failing source or measure treatment effects.
"""
import argparse
import json
from pathlib import Path
from teamlearn.common import write_json,ROLES
from teamlearn.datasets import make_task
from teamlearn.pipeline import load_models,visible_feedback
from teamlearn.reports import freeze_experience,build_deliveries,evidence_index
from teamlearn.runner import run_team


def main():
    p=argparse.ArgumentParser();p.add_argument("--model",default="mock");p.add_argument("--out",default="runs/smoke");args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    models,ledger,config=load_models("configs/models.yaml",args.model)
    try:
        source=make_task("dev",0,"F3",739,"source","sqlite")
        target=make_task("dev",0,"F3",739,"same","workflow")
        write_json(out/"source_task.json",source);write_json(out/"target_task.json",target)
        first=run_team(out/"source_task.json",models,seed=739,rounds=1,turns_per_role=7)
        first["visible_feedback"]=visible_feedback(first["events"])
        write_json(out/"source_episode.json",first)
        exp=freeze_experience(first,models["coordinator"],739)
        write_json(out/"experience.json",exp)
        reports,audit=build_deliveries(exp,{"method":"uniform","budget":2048},models,target["public"],739)
        second=run_team(out/"target_task.json",models,reports,seed=743,rounds=1,turns_per_role=7)
        write_json(out/"target_episode.json",second)
        result={"purpose":"software_smoke_only_not_research","model_config":config,"source_backend":"sqlite","target_backend":"workflow",
                "source_score":first["score"],"target_score":second["score"],"report_atoms":{c:len(v) for c,v in exp["components"].items()},
                "delivery_audit":audit,"activated_deliveries":len(second["deliveries"]),"api_calls":len(ledger),
                "total_input_tokens":sum(c.get("input_tokens",0) for c in ledger),"total_output_tokens":sum(c.get("output_tokens",0) for c in ledger),
                "usage":ledger,"completed":True}
        write_json(out/"summary.json",result)
        print(json.dumps({k:v for k,v in result.items() if k not in ("usage","delivery_audit","model_config")},indent=2))
    finally:
        for model in models.values():model.close()


if __name__=="__main__":main()
