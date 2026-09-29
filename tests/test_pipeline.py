import json
from pathlib import Path
import pytest
from teamlearn.datasets import generate,make_task,RULES,FORMULAS
from teamlearn.common import read_json,read_jsonl,ROLES,write_json
from teamlearn.pipeline import collect,run_experiment,load_models,freeze_protocol,verify_freeze
from teamlearn.reports import freeze_experience,build_deliveries
from teamlearn.runner import run_team
from teamlearn.model import MockModel


def test_family_partition_and_public_contracts():
    groups=[{rule for _,rule in entries} for entries in RULES.values()]
    assert not (groups[0]&groups[1] or groups[1]&groups[2] or groups[0]&groups[2])
    assert set.union(*groups)==set(FORMULAS)


def test_end_to_end_resume_and_provenance(tmp_path):
    dataset=tmp_path/"data";sources=tmp_path/"source";out=tmp_path/"E3"
    generate(dataset,1)
    collect(dataset,sources,"dev","configs/models.yaml","mock",limit=6,seeds=(11,),rounds=1)
    records=[read_json(p) for p in (sources/"sources").glob("*.json")]
    assert any(r["status"]=="failure" for r in records)
    failure=next(r for r in records if r["status"]=="failure")
    assert failure["selected"]["score"]["recurrence"]
    report_payload=failure["experience"]["reporter_output"]
    assert "commit_checks" not in report_payload
    options=dict(dataset=dataset,sources=sources,out=out,experiment="E3",split="dev",model_path="configs/models.yaml",model_key="mock",
                 limit=1,seeds=(101,),selected_arms=["matched","cycle","flat","structured"],rounds=1)
    result=run_experiment(**options)
    assert result["completed"]==12
    assert run_experiment(**options)["completed"]==0
    rows=read_jsonl(out/"results.jsonl")
    for kind in ("same","transfer","reversal"):
        group={r["method"]:r for r in rows if r["kind"]==kind}
        assert group["matched"]["report_audit"]["report_multiset_hash"]==group["cycle"]["report_audit"]["report_multiset_hash"]
        assert group["flat"]["report_audit"]["atomic_ids"]==group["structured"]["report_audit"]["atomic_ids"]
    with pytest.raises(ValueError,match="configuration differs"):
        run_experiment(**{**options,"seeds":(102,)})


def test_report_schema_does_not_silently_drop_flat_components(tmp_path):
    class BadReporter(MockModel):
        def complete(self,*args,**kwargs):return '{"C":[],"D":[]}'
    with pytest.raises(ValueError,match="schema"):
        freeze_experience({"events":[],"agents":{},"public_task":{}},BadReporter())


def test_reporter_uses_one_fixed_retry_after_invalid_json():
    from teamlearn.common import canonical
    class RetryReporter(MockModel):
        def __init__(self):super().__init__();self.calls=[]
        def complete(self,messages,seed,**kwargs):
            self.calls.append(seed)
            if len(self.calls)==1:return '{'
            self.ledger.append({'input_tokens':1,'output_tokens':1,'phase':'report'})
            atom={'text':'Check the current contract.','refs':['e0'],'roles':['analyst']}
            return canonical({'components':{'C':[],'D':[],'K':[atom],'P':[atom],'U':[]},'predicted_mechanism':'F1','culprits':[],'related_roles':['analyst']})
    reporter=RetryReporter()
    episode={'events':[{'event_id':'e0','role':'analyst','action':{'tool':'calculate'},'observation':{'error':'bad'}}],
             'agents':{r:r for r in ROLES},'public_task':{}}
    result=freeze_experience(episode,reporter,seed=11)
    assert reporter.calls==[11,100014]
    assert [a['valid_json'] for a in result['reporter_attempts']]==[False,True]
    assert result['report_protocol_version']=='strict-v2'


def test_frozen_dataset_tampering_is_rejected(tmp_path):
    dataset=tmp_path/"data";generate(dataset,1)
    frozen=tmp_path/"frozen.json";freeze_protocol(frozen,dataset,"configs/models.yaml")
    verify_freeze(frozen,dataset,"configs/models.yaml")
    value=read_json(dataset/"manifest.json");value["hash"]="changed";write_json(dataset/"manifest.json",value)
    with pytest.raises(ValueError,match="dataset changed"):verify_freeze(frozen,dataset,"configs/models.yaml")


def test_no_hidden_target_fields_in_real_actor_prompts(tmp_path):
    task=make_task("dev",0,"F3",1)
    path=tmp_path/"task.json";write_json(path,task)
    models={r:MockModel() for r in ROLES}
    episode=run_team(path,models,rounds=1)
    prompts=json.dumps([t["messages"] for t in episode["transcripts"]])
    for forbidden in ("commit_checks","consumption_violations","opportunity","signature","mechanism","updated_records"):
        assert forbidden not in prompts
