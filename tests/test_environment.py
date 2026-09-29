import copy
import pytest
from teamlearn.datasets import make_task,compute
from teamlearn.environment import Workflow


def solve(env,skip=()):
    t=env.task;c=t["contract"]
    def act(r,tool,**args):return env.step(r,{"tool":tool,"args":args})["event"]["observation"]
    act("coordinator","contract")
    for role in ("provider","analyst"):act("coordinator","message",to=role,text=f'Joint allocation range: {c["joint_min"]} to {c["joint_limit"]}.')
    act("coordinator","assign",check="completion",role="executor")
    if "F6" not in skip:act("coordinator","signoff")
    data=act("provider","read_records");act("provider","publish",**{k:data[k] for k in ("values","version","unit")})
    if c["publication_update"] and c["version_policy"]=="latest_at_commit" and "F3" not in skip:
        data=act("provider","read_records");act("provider","publish",**{k:data[k] for k in ("values","version","unit")})
    if c["confirmation"]=="receipt_required" and "F2" not in skip:act("provider","confirm")
    act("provider","allocate",amount=5 if "F4" in skip or c["joint_min"] else 2)
    act("analyst","allocate",amount=6 if "F4" in skip or c["joint_min"] else 3)
    value=compute(c["rule"],data["values"],c["weights"])
    act("analyst","calculate",value=value+1 if "F1" in skip else value,version=data["version"])
    if "F5" not in skip:act("executor","verify",source="independent")
    act("executor","commit")


@pytest.mark.parametrize("mechanism",[f"F{i}" for i in range(1,7)])
@pytest.mark.parametrize("backend",["workflow","sqlite"])
@pytest.mark.parametrize("kind",["same","transfer","reversal","clean"])
def test_task_solvable(mechanism,backend,kind):
    env=Workflow(make_task("dev",0,mechanism,0,kind,backend))
    try:
        solve(env)
        assert env.score()["first_success"]
        assert not env.score()["recurrence"]
    finally:env.close()


@pytest.mark.parametrize("mechanism",[f"F{i}" for i in range(1,7)])
def test_detector_known_bad_trace(mechanism):
    env=Workflow(make_task("dev",0,mechanism,0))
    try:
        solve(env,skip=(mechanism,))
        assert not env.score()["first_success"]
        assert env.score()["recurrence"]
        if mechanism=="F4":assert env.score()["strict_composition_failure"]
    finally:env.close()


def test_wrong_role_cannot_read_records_or_commit():
    env=Workflow(make_task("dev",0,"F1",0))
    try:
        for role,tool in (("analyst","read_records"),("provider","commit")):
            assert "permission denied" in env.step(role,{"tool":tool})["event"]["observation"]["error"]
        assert "score" not in env.view("executor")
        assert "records" not in env.view("analyst")
    finally:env.close()


def test_repaired_error_still_recurrence():
    env=Workflow(make_task("dev",0,"F2",0))
    try:
        solve(env,skip=("F2",))
        solve(env)
        score=env.score()
        assert score["recurrence"] and score["final_success"] and not score["first_success"]
    finally:env.close()


def test_abstention_does_not_remove_opportunity():
    env=Workflow(make_task("dev",0,"F4",0))
    try:
        score=env.score();assert score["opportunity"] and score["abstained"] and not score["first_success"]
    finally:env.close()
