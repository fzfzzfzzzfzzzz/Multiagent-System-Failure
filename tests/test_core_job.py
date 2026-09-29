from scripts.core_job import MECHANISMS, select_candidates


def test_core_candidate_selection_is_balanced_outcome_blind_and_stable():
    rows = []
    for mechanism in MECHANISMS:
        for family in range(12):
            for replica in range(2):
                rows.append({
                    "meta_id": f"{mechanism}-{family:02}-{replica}",
                    "mechanism": mechanism,
                    "family_id": f"workflow/rule-{family:02}/{mechanism}",
                    "backend": "workflow",
                })
    selected = select_candidates(rows)
    assert len(selected) == 48
    assert selected == select_candidates(list(reversed(rows)))
    for mechanism in MECHANISMS:
        group = [row for row in selected if row["mechanism"] == mechanism]
        assert len(group) == 12
        assert len({row["family_id"] for row in group}) == 12
