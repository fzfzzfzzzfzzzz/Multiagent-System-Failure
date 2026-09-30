from teamlearn.common import ROLES, digest
from teamlearn.model import MockModel
from teamlearn.reports import build_deliveries, validate_frozen
from teamlearn.reviewed import REVIEWED_RECIPIENTS, REVIEW_PROTOCOL_VERSION, review_experience
from scripts.reviewed_diagnostic_job import select_sources


def experience():
    evidence = {"e0": {"event_id": "e0", "role": "analyst", "action": {"tool": "calculate"},
                       "observation": {"error": "incorrect transformation"}}}
    components = {component: [] for component in "SOCDKPVU"}
    record = {"experience_id": "source", "source_hash": "source", "components": components, "evidence": evidence,
              "predicted_mechanism": "F1", "culprits": ["analyst"], "related_roles": ["coordinator"],
              "source_agents": {role: role for role in ROLES}, "query": {}, "version": 1, "supersedes": [],
              "status": "active", "reporter_output": "", "reporter_attempts": [], "report_cost": [],
              "report_protocol_version": "test", "provenance_validation": "test"}
    record["frozen_hash"] = digest(record)
    return record


def test_reviewed_report_is_frozen_and_source_grounded():
    reviewed = review_experience(experience(), "F1")
    validate_frozen(reviewed)
    assert reviewed["report_protocol_version"] == REVIEW_PROTOCOL_VERSION
    assert reviewed["related_roles"] == REVIEWED_RECIPIENTS["F1"]
    assert all(atom["refs"] == ["e0"] for component in "CDKPU" for atom in reviewed["components"][component])
    assert reviewed["components"]["V"] == []


def test_explicit_routing_keeps_report_content_fixed():
    reviewed = review_experience(experience(), "F1")
    models = {role: MockModel() for role in ROLES}
    common = {"method": "uniform", "components": "SOCDKPU", "budget": 512,
              "budget_mode": "per_agent", "role_views": False}
    auto, audit_auto = build_deliveries(reviewed, {**common, "recipients": ["coordinator"]}, models, {})
    routed, audit_routed = build_deliveries(reviewed, {**common, "recipients": REVIEWED_RECIPIENTS["F1"]}, models, {})
    assert {text for text in auto.values() if text} == {text for text in routed.values() if text}
    assert audit_auto["recipients"] == ["coordinator"]
    assert audit_routed["recipients"] == REVIEWED_RECIPIENTS["F1"]


def test_source_selection_is_balanced_and_round_robins_families():
    rows=[]
    for mechanism in ("F1", "F2", "F3", "F4"):
        for family in ("a", "b", "c"):
            for index in range(2):
                rows.append({"mechanism": mechanism, "backend": "workflow", "family_id": f"workflow/{family}/{mechanism}",
                             "meta_id": f"{mechanism}-{family}-{index}"})
    selected=select_sources(rows,per_mechanism=6)
    assert len(selected)==24
    assert all(sum(row["mechanism"]==mechanism for row in selected)==6 for mechanism in ("F1","F2","F3","F4"))
    assert all(len({row["family_id"] for row in selected if row["mechanism"]==mechanism})==3 for mechanism in ("F1","F2","F3","F4"))
