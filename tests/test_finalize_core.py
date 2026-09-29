from scripts.finalize_core import (classify, conditional_transfer_results, holm,
                                   hypothesis_summary, same_condition_rates, select_primary)


def row(experiment, a, b, metric, difference, interval, p):
    return {"experiment": experiment, "kind": "same", "a": a, "b": b,
            "metric": metric, "difference": difference, "ci95": interval,
            "p_block_permutation": p, "estimable": True, "n_clusters": 12}


def test_primary_family_is_exactly_six_and_uses_one_holm_family():
    e1 = [row("E1", "components_SOCDKPU", "components_SP", "first_success", .2, [.1, .3], .01),
          row("E1", "components_SOCDKPU", "components_SP", "recurrence", -.2, [-.3, -.1], .02)]
    e3 = []
    for b in ("cycle", "random"):
        e3 += [row("E3", "matched", b, "first_success", .1, [.01, .2], .03),
               row("E3", "matched", b, "recurrence", -.1, [-.2, -.01], .04)]
    selected = select_primary(e1, e3)
    assert len(selected) == 6
    assert [item["p_holm_primary_family"] for item in selected] == [.06, .1, .12, .12, .12, .12]
    assert all(item["evidence"] == "inconclusive" for item in selected)


def test_directional_classification_and_summary():
    rows = [
        {"hypothesis": "H1", "metric": "first_success", "difference": .2,
         "ci95": [.1, .3], "p_holm_primary_family": .04, "estimable": True},
        {"hypothesis": "H1", "metric": "recurrence", "difference": -.2,
         "ci95": [-.3, -.1], "p_holm_primary_family": .04, "estimable": True},
    ]
    for item in rows:
        item["evidence"] = classify(item)
    assert [item["evidence"] for item in rows] == ["supports_expected_direction"] * 2
    assert hypothesis_summary(rows)["H1"]["classification"] == "supported_on_both_primary_outcomes"


def test_holm_is_monotone_in_sorted_p_values():
    rows = [{"p_block_permutation": value} for value in (.04, .001, .02)]
    holm(rows)
    adjusted = [row["p_holm_primary_family"] for row in sorted(rows, key=lambda item: item["p_block_permutation"])]
    assert adjusted == sorted(adjusted)


def test_same_condition_rates_excludes_nonprimary_populations_and_methods():
    fields = {"n_runs": "135", "first_success": "0", "recurrence": "0.7", "mean_total_tokens": "42"}
    e1 = [dict(fields, experiment="E1", method="components_SOCDKPU", kind="same"),
          dict(fields, experiment="E1", method="components_SOCDKPU", kind="transfer"),
          dict(fields, experiment="E1", method="components_S", kind="same")]
    e3 = [dict(fields, experiment="E3", method="matched", kind="same")]
    rows = same_condition_rates(e1, e3)
    assert [(row["experiment"], row["method"]) for row in rows] == [
        ("E1", "components_SOCDKPU"), ("E3", "matched")]
    assert rows[0]["n_runs"] == 135 and rows[0]["recurrence"] == .7


def test_conditional_transfer_selects_only_declared_recurrence_rows():
    e1 = [dict(row("E1", "components_SOCDKPU", "components_SP", "recurrence", -.2, [-.3, -.1], .01),
               kind="transfer", p_holm=.03)]
    e3 = [dict(row("E3", "matched", baseline, "recurrence", -.1, [-.2, -.01], .04),
               kind="transfer", p_holm=.2) for baseline in ("cycle", "random")]
    selected = conditional_transfer_results(e1, e3)
    assert [(item["experiment"], item["b"]) for item in selected] == [
        ("E1", "components_SP"), ("E3", "cycle"), ("E3", "random")]
