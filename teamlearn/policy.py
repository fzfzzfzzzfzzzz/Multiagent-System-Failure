"""Small joint-recipient ridge ranker, trained on actual training outcomes."""
from __future__ import annotations

import itertools
import numpy as np
from .common import ROLES,digest,read_json,write_json

ACTIONS=[list(s) for n in range(5) for s in itertools.combinations(ROLES,n)]


def features(experience,recipients,budget):
    bits=[float(r in recipients) for r in ROLES]
    mech=[float(experience["predicted_mechanism"]==f"F{i}") for i in range(1,7)]
    related=[float(r in experience.get("related_roles",[])) for r in ROLES]
    pairs=[bits[i]*bits[j] for i in range(4) for j in range(i+1,4)]
    return np.array([1.,budget/4096,*bits,*pairs,*mech,*related,
                     *[m*b for m in mech for b in bits],*[a*b for a,b in zip(bits,related)]],dtype=float)


class Policy:
    def __init__(self,weights,metadata=None):self.weights=np.asarray(weights);self.metadata=metadata or {}

    def select(self,experience,budget):
        return max(ACTIONS,key=lambda action:float(features(experience,action,budget)@self.weights))

    def save(self,path):write_json(path,{"weights":self.weights.tolist(),"metadata":self.metadata})

    @classmethod
    def load(cls,path):
        value=read_json(path);return cls(value["weights"],value["metadata"])

    @classmethod
    def fit(cls,rows,ridge=1.0):
        if not rows:raise ValueError("no policy training outcomes")
        if any(r["split"]!="train" or r.get("mock") for r in rows):raise ValueError("policy training accepts real train outcomes only")
        x=np.stack([features(r["experience"],r["recipients"],r["budget"]) for r in rows])
        y=np.array([float(r["first_success"])-float(r["recurrence"])-0.1*r["report_tokens"]/4096 for r in rows])
        weights=np.linalg.solve(x.T@x+ridge*np.eye(x.shape[1]),x.T@y)
        return cls(weights,{"training_hash":digest(rows),"n":len(rows),"families":sorted({r["family_id"] for r in rows}),"ridge":ridge,
                            "training_execution_tokens":sum(r.get("training_tokens",0) for r in rows),"objective":"first_success - recurrence - 0.1 * report_tokens/4096"})

    def update_visible(self,experience,recipients,budget,feedback,rate=0.01):
        # Online feedback contains only public transport errors and completion signals.
        # This proxy is deliberately distinct from the hidden research grader.
        allowed={"transport_committed","tool_error_rate"}
        if set(feedback)-allowed:raise ValueError("hidden or unknown online feedback field")
        target=float(feedback.get("transport_committed",False))-feedback.get("tool_error_rate",0)
        x=features(experience,recipients,budget)
        self.weights+=rate*(target-float(x@self.weights))*x/(1+float(x@x))
