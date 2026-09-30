from scripts.analyze_core_failures import budget_ceiling_roles, derive_run, outcome_category


def test_diagnostic_separates_avoided_target_from_task_success():
    row = {
        "run_id": "r", "experiment": "E1", "family_id": "workflow/funds/F2",
        "mechanism": "F2", "kind": "same", "method": "full", "seed": 101,
        "first_success": False, "final_success": False, "abstained": False,
        "opportunity": True, "recurrence": False, "strict_composition_failure": False,
        "preventive_action": True, "unnecessary_wait": False, "new_errors": ["F4"],
        "commit_checks": [{"F1": True, "F2": True, "F3": True, "F4": False}], "cost": [],
    }
    episode = {"events": [], "transcripts": [], "turns_per_role": 3}
    result = derive_run(row, episode)
    assert outcome_category(row) == "other_failure_without_target_recurrence"
    assert result["target_avoided_but_failed"]
    assert result["first_failed_checks"] == ["F4"]


def test_budget_ceiling_requires_full_round_without_finish():
    rows = [{"role": "analyst", "round": 0, "output": '{"tool":"read_artifact","args":{}}'},
            {"role": "analyst", "round": 0, "output": '{"tool":"calculate","args":{}}'},
            {"role": "analyst", "round": 0, "output": '{"tool":"allocate","args":{}}'}]
    assert budget_ceiling_roles(rows, 3) == ["analyst"]
    rows[-1]["output"] = '{"tool":"finish","args":{}}'
    assert budget_ceiling_roles(rows, 3) == []
