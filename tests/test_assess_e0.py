from scripts.assess_e0 import METHODS, assess


def row(pair, method, recurrence, mechanism):
    return {"meta_id": f"m{pair}", "target_id": f"t{pair}", "seed": 101,
            "method": method, "first_success": False, "recurrence": recurrence,
            "mechanism": mechanism}


def test_e0_gate_requires_paired_sensitivity_across_mechanisms():
    rows = []
    for pair, mechanism in enumerate(("F1", "F2", "F3", "F4")):
        for method in METHODS:
            rows.append(row(pair, method, method == "no_memory", mechanism))
    result = assess(rows, {"passed": True})
    assert result["passed"]
    assert result["intervention_sensitive_pairs"] == 4
    assert result["sensitive_mechanisms"] == ["F1", "F2", "F3", "F4"]


def test_e0_gate_rejects_constant_outcomes():
    rows = [row(pair, method, True, "F1") for pair in range(4) for method in METHODS]
    result = assess(rows, {"passed": True})
    assert not result["passed"]
