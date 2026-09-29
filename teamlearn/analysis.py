"""Paired cluster inference. Seeds and agents are never independent samples."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
from .common import read_jsonl,write_json


def cluster_id(row):
    # Shared aggregation rules across mechanisms/backends remain correlated.
    return row["family_id"].split("/")[1]


def pair_key(row):return row["meta_id"],row["target_id"],row["seed"]


def cost_tokens(row):
    execution=sum(c.get("input_tokens",0)+c.get("output_tokens",0) for c in row["cost"])
    construction=sum(c.get("input_tokens",0)+c.get("output_tokens",0) for c in row["report_construction_cost"])
    # Reading and forwarding are already in actual execution API input usage.
    return execution+construction


def paired_difference(rows,a,b,metric="first_success",samples=5000,seed=19):
    left={pair_key(r):r for r in rows if r["method"]==a}
    right={pair_key(r):r for r in rows if r["method"]==b}
    common=sorted(left.keys()&right.keys())
    clusters=defaultdict(list)
    for key in common:
        if metric=="recurrence" and not left[key]["opportunity"]:continue
        value=lambda r:cost_tokens(r) if metric=="total_tokens" else float(r[metric])
        clusters[cluster_id(left[key])].append(value(left[key])-value(right[key]))
    means=np.array([np.mean(v) for v in clusters.values()])
    if not len(means):return {"a":a,"b":b,"metric":metric,"n_pairs":0,"estimable":False}
    effect=float(means.mean())
    rng=np.random.default_rng(seed)
    draws=np.mean(rng.choice(means,(samples,len(means)),replace=True),axis=1)
    if len(means)<=16:
        signs=np.array([[1 if mask&(1<<i) else -1 for i in range(len(means))] for mask in range(2**len(means))])
        null=np.mean(signs*means,axis=1)
        p=float(np.mean(np.abs(null)>=abs(effect)-1e-12))
    else:
        null=np.mean(rng.choice([-1,1],(samples,len(means)))*means,axis=1)
        p=float((1+np.sum(np.abs(null)>=abs(effect)-1e-12))/(samples+1))
    return {"a":a,"b":b,"metric":metric,"n_pairs":len(common),"n_clusters":len(means),"estimable":True,
            "difference":effect,"ci95":np.quantile(draws,[.025,.975]).tolist(),"p_block_permutation":p,
            "small_cluster_warning":len(means)<10,"estimand":"equally weighted rule-family mean of paired differences"}


def interaction(rows):
    groups={name:{pair_key(r):r for r in rows if r["method"]==name} for name in ("factor_00","factor_01","factor_10","factor_11")}
    keys=set.intersection(*(set(g) for g in groups.values()))
    pseudo=[]
    for key in keys:
        r=groups["factor_11"][key]
        for method,value in (("interaction",float(r["first_success"])-float(groups["factor_10"][key]["first_success"])-float(groups["factor_01"][key]["first_success"])+float(groups["factor_00"][key]["first_success"])),("zero",0)):
            pseudo.append({**r,"method":method,"first_success":value})
    return paired_difference(pseudo,"interaction","zero")


def holm(comparisons):
    ordered=sorted((r for r in comparisons if r.get("estimable")),key=lambda r:r["p_block_permutation"])
    running=0.
    for i,row in enumerate(ordered):
        running=max(running,min(1.,row["p_block_permutation"]*(len(ordered)-i)))
        row["p_holm"]=running


def summarize(paths,out,allow_mock=False):
    rows=[]
    for path in paths:rows+=read_jsonl(path)
    if any(r["mock"] for r in rows) and not allow_mock:raise ValueError("refusing to analyze mock data as research results; use --allow-mock for software QA")
    if len({r["run_id"] for r in rows})!=len(rows):raise ValueError("duplicate runs; do not pool repeated files")
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    grouped=defaultdict(list)
    for row in rows:grouped[(row["experiment"],row["method"],row["kind"],row["backend"])].append(row)
    summary=[]
    for (exp,method,kind,backend),items in sorted(grouped.items()):
        opportunities=[r for r in items if r["opportunity"]]
        units=defaultdict(list)
        for r in items:units[cluster_id(r)].append(float(r["first_success"]))
        cluster_means=np.array([np.mean(v) for v in units.values()])
        rng=np.random.default_rng(19)
        interval=np.quantile(np.mean(rng.choice(cluster_means,(5000,len(cluster_means)),replace=True),axis=1),[.025,.975])
        summary.append({"experiment":exp,"method":method,"kind":kind,"backend":backend,"n_runs":len(items),
                        "n_sources":len({r["meta_id"] for r in items}),"n_clusters":len({cluster_id(r) for r in items}),
                        "first_success":np.mean([r["first_success"] for r in items]),"final_success":np.mean([r["final_success"] for r in items]),
                        "cluster_weighted_success":float(cluster_means.mean()),"success_ci_low":float(interval[0]),"success_ci_high":float(interval[1]),
                        "recurrence":np.mean([r["recurrence"] for r in opportunities]) if opportunities else None,
                        "opportunities":len(opportunities),"abstention":np.mean([r["abstained"] for r in items]),
                        "unnecessary_wait":np.mean([r["unnecessary_wait"] for r in items]),"mean_total_tokens":np.mean([cost_tokens(r) for r in items]),
                        "mean_report_tokens":np.mean([r["report_audit"]["report_tokens"] for r in items]),"unknown_usage_calls":sum(c.get("usage_unknown",False) for r in items for c in r["cost"])})
    with (out/"summary.csv").open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(summary[0]) if summary else []);writer.writeheader();writer.writerows(summary)
    comparisons=[]
    registered=[("E1","components_SOCDKPU","components_SP"),("E3","matched","cycle"),("E3","matched","random"),
                ("E4","full","no_K"),("E4","full","no_memory")]
    registered += [("E5",f"learned_b{budget}",f"{baseline}_b{budget}") for budget in (512,1024,2048,4096) for baseline in ("uniform","culprit","role_rag","gmemory_adapt","mechanism")]
    registered += [("E7","learned",baseline) for baseline in ("none","uniform","role_rag","gmemory_adapt","mechanism")]
    for exp,a,b in registered:
        for kind in sorted({r["kind"] for r in rows if r["experiment"]==exp}):
            subset=[r for r in rows if r["experiment"]==exp and r["kind"]==kind]
            if exp=="E4" and kind=="reversal":subset=[r for r in subset if r["mechanism"]!="F1"]
            for metric in ("first_success","recurrence","total_tokens"):
                comparisons.append({"experiment":exp,"kind":kind,**paired_difference(subset,a,b,metric)})
    holm(comparisons)
    write_json(out/"paired_comparisons.json",comparisons)
    write_json(out/"interaction.json",interaction([r for r in rows if r["experiment"]=="E3" and r["kind"]=="same"]))
    write_json(out/"analysis_manifest.json",{"mock":any(r["mock"] for r in rows),"rows":len(rows),"cluster_definition":"aggregation-rule family across mechanisms, backends and seeds",
               "cost_note":"construction plus actual API usage; report reads are included in execution, not double counted; source costs are separately in run records",
               "caution":"small cluster intervals are descriptive; no automatic hypothesis acceptance"})
    plot_status={"created":True,"path":str(out/"cost_success.png")}
    try:
        plot(summary,out)
    except ModuleNotFoundError as error:
        # Figures are a presentation artifact.  Keep the preregistered numerical
        # analysis usable on the inference environment when matplotlib is absent.
        if not (error.name or "").startswith("matplotlib"):
            raise
        plot_status={"created":False,"reason":"matplotlib is not installed in the analysis environment"}
    write_json(out/"plot_status.json",plot_status)
    return {"rows":len(rows),"out":str(out),"mock":any(r["mock"] for r in rows),"plot":plot_status}


def plot(summary,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5))
    for row in summary:
        if row["kind"]!="same":continue
        center=row["cluster_weighted_success"]
        ax.errorbar(row["mean_total_tokens"],center,yerr=[[max(0,center-row["success_ci_low"])],[max(0,row["success_ci_high"]-center)]],fmt="o",label=row["method"])
    ax.set(xlabel="Total API tokens, including report construction",ylabel="First-attempt team success",ylim=(-.05,1.05))
    if len(summary)<=20:ax.legend(fontsize=7,loc="best")
    fig.tight_layout();fig.savefig(out/"cost_success.png",dpi=160);plt.close(fig)
