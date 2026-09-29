import copy
import pytest
from teamlearn.common import ROLES,digest
from teamlearn.model import MockModel
from teamlearn.reports import build_deliveries,validate_frozen,Memory
from teamlearn.policy import Policy
from teamlearn.analysis import paired_difference


def experience():
    record={"experience_id":"x","source_hash":"x","components":{c:[{"atom_id":c+str(i),"text":c+" fact "+r,"refs":["e0"],"status":"observed" if c=="O" else "hypothesis","roles":[r]} for i,r in enumerate(ROLES)] for c in "SOCDKPVU"},
            "evidence":{"e0":{"event_id":"e0"}},"related_roles":["provider","analyst"],"culprits":["provider"],"source_agents":{r:r for r in ROLES},"status":"active","predicted_mechanism":"F2","query":{},"report_cost":[]}
    record["frozen_hash"]=digest(record);return record


def test_immutable_and_ablation():
    exp=experience();models={r:MockModel() for r in ROLES}
    reports,audit=build_deliveries(exp,{"components":"SO","budget":10000},models,{"agents":exp["source_agents"]})
    assert all(":C" not in text and ":K" not in text for text in reports.values())
    exp["components"]["O"][0]["text"]="edited"
    with pytest.raises(ValueError):validate_frozen(exp)


def test_permutation_has_same_multiset_and_budget():
    exp=experience();models={r:MockModel() for r in ROLES}
    base={"budget":2048,"budget_mode":"content_set","role_views":True}
    matched,a=build_deliveries(exp,base,models,{})
    permuted,b=build_deliveries(exp,{**base,"permutation":"cycle"},models,{})
    assert sorted(matched.values())==sorted(permuted.values())
    assert a["report_multiset_hash"]==b["report_multiset_hash"]
    assert matched!=permuted
    assert a["report_tokens"]<=2048


def test_train_only_policy_and_no_hidden_update():
    with pytest.raises(ValueError):Policy.fit([{"split":"test"}])
    import numpy as np
    from teamlearn.policy import features
    policy=Policy(np.zeros(len(features(experience(),[],2048))))
    with pytest.raises(ValueError):policy.update_visible(experience(),[],2048,{"hidden_score":True})


def test_cluster_unit_not_seeds():
    rows=[]
    for domain in ("a","b","c"):
        for seed in range(10):
            for method,value in (("a",True),("b",False)):
                rows.append({"family_id":f"workflow/{domain}/F1","meta_id":domain,"target_id":domain,"seed":seed,"method":method,"first_success":value})
    result=paired_difference(rows,"a","b",samples=100)
    assert result["n_pairs"]==30 and result["n_clusters"]==3
    assert result["difference"]==1
