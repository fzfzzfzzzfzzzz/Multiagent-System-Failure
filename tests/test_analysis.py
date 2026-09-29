import json

import teamlearn.analysis as analysis


def test_summarize_keeps_numerical_outputs_when_matplotlib_is_absent(tmp_path, monkeypatch):
    row = {
        "run_id": "r1", "mock": False, "family_id": "workflow/funds/F1",
        "meta_id": "m1", "target_id": "t1", "seed": 101,
        "experiment": "E1", "method": "components_SOCDKPU", "kind": "same",
        "backend": "workflow", "first_success": False, "final_success": False,
        "opportunity": True, "recurrence": True, "abstained": False,
        "unnecessary_wait": False, "cost": [], "report_construction_cost": [],
        "report_audit": {"report_tokens": 10},
    }
    source = tmp_path / "results.jsonl"
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")

    def missing_matplotlib(*_args):
        error = ModuleNotFoundError("No module named 'matplotlib'")
        error.name = "matplotlib"
        raise error

    monkeypatch.setattr(analysis, "plot", missing_matplotlib)
    result = analysis.summarize([source], tmp_path / "out")

    assert result["rows"] == 1
    assert result["plot"]["created"] is False
    assert (tmp_path / "out" / "summary.csv").exists()
    assert (tmp_path / "out" / "paired_comparisons.json").exists()
    assert json.loads((tmp_path / "out" / "plot_status.json").read_text())["created"] is False
