"""Run staged required experiments. Merely invoking --plan starts no models or trials."""
import argparse
import subprocess
import sys
from pathlib import Path
import yaml
from teamlearn.common import read_jsonl,write_json
from teamlearn.datasets import generate
from teamlearn.pipeline import collect,run_experiment,source_records,freeze_protocol
from teamlearn.policy import Policy
from teamlearn.analysis import summarize
from teamlearn.stream import run_stream

STAGES=["data","collect-dev","pilot","collect-train","train-policy","development","freeze","collect-test","confirm","heterogeneous","sequential","analyze"]


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",default="configs/study.yaml")
    parser.add_argument("--stage",choices=STAGES+["all"],default="all");parser.add_argument("--plan",action="store_true")
    args=parser.parse_args();config=yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.plan:
        print("\n".join(STAGES));return
    root=Path(config["root"]);sources=root/"collected";policy=root/"policy.json";frozen=root/"frozen.json"
    model=config["model"];models=config["models_file"];dataset=config["dataset"]
    common={"rounds":config["rounds"],"turns":config["turns_per_role"]}
    for stage in STAGES if args.stage=="all" else [args.stage]:
        print("STAGE",stage,flush=True)
        if stage=="data":generate(dataset,config["per_family"])
        elif stage.startswith("collect-"):
            split=stage.split("-")[1]
            if split=="test" and not frozen.exists():raise ValueError("freeze development protocol before collecting test sources")
            collect(dataset,sources,split,models,model,seeds=tuple(config["source_seeds"]),**common)
        elif stage=="pilot":
            records=source_records(sources,"dev")
            counts={m:sum(r["mechanism"]==m and r["backend"]=="workflow" for r in records) for m in config["mechanisms_primary"]}
            write_json(root/"pilot_gate.json",{"counts":counts,"minimum":config["minimum_dev_failures_per_primary_mechanism"]})
            if min(counts.values(),default=0)<config["minimum_dev_failures_per_primary_mechanism"]:
                raise RuntimeError("E0 source coverage gate failed; see pilot_gate.json. No artificial failures or test-set tuning will be used.")
            run_experiment(dataset,sources,root/"dev_E0","E0","dev",models,model,seeds=tuple(config["target_seeds"]),**common)
        elif stage=="train-policy":
            run_experiment(dataset,sources,root/"train_actions","TRAIN","train",models,model,seeds=tuple(config["target_seeds"]),**common)
            fitted=Policy.fit(read_jsonl(root/"train_actions"/"policy_training.jsonl"));fitted.save(policy)
        elif stage in ("development","confirm"):
            split="dev" if stage=="development" else "test"
            for exp in config["core_experiments"]:
                run_experiment(dataset,sources,root/f"{split}_{exp}",exp,split,models,model,seeds=tuple(config["target_seeds"]),policy_path=policy,
                               freeze_path=frozen if split=="test" else None,**common)
            run_experiment(dataset,sources,root/f"{split}_baselines","BASELINES",split,models,model,seeds=tuple(config["target_seeds"]),policy_path=policy,freeze_path=frozen if split=="test" else None,**common)
        elif stage=="freeze":
            if not (root/"dev_E3"/"results.jsonl").exists():raise ValueError("run development interventions before freezing")
            freeze_protocol(frozen,dataset,models,policy,config["rounds"],config["turns_per_role"],config["target_seeds"])
        elif stage=="sequential":
            run_stream(dataset,sources,root/"E6",models,model,config["stream_steps"],config["stream_count"],config["probe_every"],policy_path=policy,**common)
        elif stage=="heterogeneous":
            for label,key,mixed in (("heterogeneous",model,True),("qwen38_27b","qwen38_27b",False)):
                run_experiment(dataset,sources,root/f"test_E7_{label}","E7","test",models,key,seeds=tuple(config["target_seeds"]),policy_path=policy,freeze_path=frozen,heterogeneous=mixed,**common)
        elif stage=="analyze":
            for directory in sorted(root.glob("test_*")):
                if (directory/"results.jsonl").exists():summarize([directory/"results.jsonl"],root/"analysis"/directory.name)
            if (root/"E6"/"results.jsonl").exists():
                subprocess.run([sys.executable,"scripts/analyze_stream.py","--input",str(root/"E6"),"--out",str(root/"analysis"/"E6")],check=True)


if __name__=="__main__":main()
