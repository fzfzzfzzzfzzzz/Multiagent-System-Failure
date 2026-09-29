from scripts.f2_probe_job import candidates


def test_probe_uses_one_f2_candidate_per_family():
    rows = []
    for family in ("a", "b", "c"):
        for replica in range(2):
            rows.append({"meta_id": f"{family}-{replica}", "mechanism": "F2",
                         "backend": "workflow", "family_id": family})
        rows.append({"meta_id": f"ignored-{family}", "mechanism": "F1",
                     "backend": "workflow", "family_id": family})
    selected = candidates(list(reversed(rows)))
    assert [row["meta_id"] for row in selected] == ["a-0", "b-0", "c-0"]
