"""Stream-level bootstrap; time points and probes do not inflate sample size."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from teamlearn.common import read_jsonl,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--out",required=True);p.add_argument("--allow-mock",action="store_true");a=p.parse_args()
    root=Path(a.input);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    rows=read_jsonl(root/"results.jsonl");probes=read_jsonl(root/"probes.jsonl")
    if any(r["mock"] for r in rows) and not a.allow_mock:raise ValueError("mock data are not research")
    by_group=defaultdict(list)
    for row in rows:by_group[row["group"]].append(row)
    summary={};rng=np.random.default_rng(19)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5))
    for label,items in sorted(by_group.items()):
        streams=sorted({r["stream"] for r in items});steps=sorted({r["step"] for r in items})
        matrix=np.array([[next(float(r["first_success"]) for r in items if r["stream"]==s and r["step"]==t) for t in steps] for s in streams])
        means=matrix.mean(axis=1);draws=rng.choice(means,(5000,len(means))).mean(axis=1)
        summary[label]={"independent_streams":len(streams),"mean_curve_success":float(means.mean()),"ci95":np.quantile(draws,[.025,.975]).tolist(),
                        "probe_success":float(np.mean([r["first_success"] for r in probes if r["group"]==label and r["intervention"]=="normal"]))}
        ax.plot(steps,matrix.mean(axis=0),label=label)
    ax.set(xlabel="Chronological task",ylabel="First-attempt success",ylim=(-.05,1.05));ax.legend();fig.tight_layout();fig.savefig(out/"learning_curve.png",dpi=160);plt.close(fig)
    write_json(out/"stream_summary.json",summary)
    print(summary)


if __name__=="__main__":main()
