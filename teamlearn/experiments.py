"""Predeclared interventions; no test-set selection of recipients or budgets."""
from __future__ import annotations

from .policy import ACTIONS


def arm(name,method="uniform",**kwargs):
    return {"name":name,"method":method,"components":"SOCDKPVU","budget":2048,"budget_mode":"team_total",**kwargs}


def arms(experiment):
    if experiment=="E0":
        return [arm("no_memory","none"),arm("uniform"),arm("source_evidence",components="SODU"),arm("full_history","history")]
    if experiment=="E1":
        return [arm("components_"+c,components=c,budget=16384,budget_mode="content_set") for c in ("S","SO","SOC","SOCD","SOCDK","SP","SOCDKP","SOCDKPU","SV","SOCDKPVU")]+[arm("unrelated_budget_matched",unrelated=True,budget=16384,budget_mode="content_set")]
    if experiment=="E2":
        return [arm("subset_"+str(i),"subset",recipients=recipients,budget_mode="per_agent",budget=512) for i,recipients in enumerate(ACTIONS)]
    if experiment=="TRAIN":
        return [arm(f"subset_{i}_b{budget}","subset",recipients=recipients,budget_mode="team_total",budget=budget) for budget in (512,1024,2048,4096) for i,recipients in enumerate(ACTIONS)]
    if experiment=="E3":
        return [arm("matched",role_views=True,budget_mode="content_set"),
                arm("cycle",role_views=True,budget_mode="content_set",permutation="cycle"),
                arm("random",role_views=True,budget_mode="content_set",permutation="random"),
                *[arm(f"factor_{c}{r}",components="SOCDKPU" if c else "SP",recipient_mode="related" if r else "fixed") for c in (0,1) for r in (0,1)],
                arm("structured",budget_mode="content_set"),arm("flat",style="flat",budget_mode="content_set"),
                arm("pointers",pointers=True),arm("evidence_full")]
    if experiment=="E4":
        return [arm("no_memory","none"),arm("full","mechanism"),arm("no_K","mechanism",components="SOCDPVU"),
                arm("rule_only","rule"),arm("old_id","mechanism",bind_old_ids=True),arm("train_library","role_rag",memory_mode="train_only")]
    if experiment=="E5":
        return [arm(f"{method}_b{budget}",method,budget=budget) for budget in (512,1024,2048,4096) for method in ("uniform","culprit","role_rag","gmemory_adapt","mechanism","learned")]+[arm("full_history","history")]
    if experiment=="E7":
        return [arm(name,name) for name in ("none","uniform","role_rag","gmemory_adapt","mechanism","learned")]
    if experiment=="BASELINES":
        return [arm(name,name) for name in ("none","success","rule","history","uniform","independent","culprit","coordinator","role_rag","gmemory_adapt","mechanism","learned")]
    raise ValueError("unknown experiment: "+experiment)


def target_kinds(experiment):
    if experiment=="E4":return ["same","transfer","reversal","clean","unrelated","renamed"]
    if experiment in ("E1","E3","E5","E7","BASELINES"):return ["same","transfer","reversal"]
    return ["same"]
