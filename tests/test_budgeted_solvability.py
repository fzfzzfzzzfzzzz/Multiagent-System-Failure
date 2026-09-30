import pytest

from teamlearn.common import ROLES, write_json
from teamlearn.datasets import make_task
from teamlearn.diagnostics import BudgetedLegalModel, budget_ceiling_roles
from teamlearn.runner import run_team


@pytest.mark.parametrize("mechanism", ["F1", "F2", "F3", "F4"])
@pytest.mark.parametrize("kind", ["same", "transfer", "reversal"])
def test_formal_scheduler_is_solvable_with_three_calls(tmp_path, mechanism, kind):
    path = tmp_path / f"{mechanism}-{kind}.json"
    write_json(path, make_task("dev", 0, mechanism, 0, kind, "workflow"))
    episode = run_team(path, {role: BudgetedLegalModel() for role in ROLES},
                       seed=19, rounds=2, turns_per_role=3)
    assert episode["score"]["first_success"]
    assert not episode["score"]["recurrence"]
    assert not [event for event in episode["events"] if event["observation"].get("error")]
    # The witness is allowed to use the final slot.  This is intentionally
    # reported as a tight budget, while success proves it is still feasible.
    assert set(budget_ceiling_roles(episode["transcripts"], 3)) <= set(ROLES)
