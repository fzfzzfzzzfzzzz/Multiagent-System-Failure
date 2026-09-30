"""Development-only, protocol-reviewed failure reports for P2 diagnostics.

These reports are deterministic reference interventions.  They are generated
from source-episode evidence and a mechanism-specific review protocol; they
are not presented as independent human annotations or confirmation data.
"""
from __future__ import annotations

import copy

from .common import COMPONENTS, ROLES, digest
from .reports import validate_frozen


REVIEW_PROTOCOL_VERSION = "p2-reference-v1"

REVIEWED_RECIPIENTS = {
    "F1": ["analyst", "executor"],
    "F2": ["provider", "analyst"],
    "F3": ["provider", "analyst", "executor"],
    "F4": list(ROLES),
}

TEMPLATES = {
    "F1": {
        "C": "A locally valid calculation can still use the wrong transformation required by the current contract.",
        "D": "The analyst and executor must combine the current artifact version with the contract's exact aggregation rule.",
        "K": "Apply when the contract requires an exact integer aggregate; except only when it explicitly authorizes another transformation.",
        "P": "Analyst recomputes from the current artifact; executor checks version and exact contract transformation before commit.",
        "U": "The source trace establishes a calculation mismatch, but does not establish the model's intent or universal generalization.",
    },
    "F2": {
        "C": "A calculation can begin after publication but before the provider has confirmed the required dependency.",
        "D": "The analyst depends on an explicit provider confirmation for the current artifact version, not publication alone.",
        "K": "Apply when the contract requires confirmation before calculation; except when the current contract explicitly removes that dependency.",
        "P": "Provider confirms the current version, then analyst verifies that confirmation before calculating.",
        "U": "The source trace shows ordering evidence, but does not prove why the required confirmation was missed.",
    },
    "F3": {
        "C": "A later revision can make an earlier read or calculation stale even when each local action was accepted.",
        "D": "Provider, analyst, and executor must agree on the latest published artifact version before calculation and commit.",
        "K": "Apply when the artifact revision changes; except when the contract proves the earlier version remains authoritative.",
        "P": "After a revision, reread or republish as required, recalculate from the latest version, and commit that version only.",
        "U": "The source trace identifies a version mismatch, but cannot establish that every revision requires identical recovery steps.",
    },
    "F4": {
        "C": "Individually feasible allocations can conflict because local choices do not guarantee the team's joint constraint.",
        "D": "All roles depend on a shared allocation plan whose combined amounts satisfy the current global contract.",
        "K": "Apply when the contract constrains the joint allocation; except when it explicitly declares choices independent.",
        "P": "Coordinator gathers options, assigns a compatible joint plan, and each role allocates only its assigned amount.",
        "U": "The source trace establishes a joint conflict, but does not prove which role originated the coordination failure.",
    },
}


def _reference(events, preferred_tools):
    for tool in preferred_tools:
        for event in events:
            if event.get("action", {}).get("tool") == tool:
                return event["event_id"]
    if not events:
        raise ValueError("a reviewed report requires source events")
    return events[0]["event_id"]


def review_experience(experience, mechanism):
    """Replace model-inferred C/D/K/P/U with a frozen reference intervention."""
    validate_frozen(experience)
    if mechanism not in TEMPLATES:
        raise ValueError("P2 review supports F1-F4 only")
    events = list(experience["evidence"].values())
    refs = {
        "C": _reference(events, ("calculate", "commit", "allocate", "confirm", "publish")),
        "D": _reference(events, ("contract", "read_artifact", "read_records", "status", "message")),
        "K": _reference(events, ("contract", "calculate", "commit", "allocate")),
        "P": _reference(events, ("calculate", "confirm", "publish", "commit", "allocate")),
        "U": _reference(events, ("commit", "calculate", "allocate", "confirm", "publish")),
    }
    reviewed = copy.deepcopy(experience)
    reviewed["experience_id"] = "reviewed-" + digest([experience["experience_id"], mechanism, REVIEW_PROTOCOL_VERSION])[:20]
    reviewed["components"] = {component: copy.deepcopy(experience["components"].get(component, []))
                              if component in ("S", "O", "V") else [] for component in COMPONENTS}
    roles = REVIEWED_RECIPIENTS[mechanism]
    status = {"C": "unverified_hypothesis", "D": "dependency_hypothesis", "K": "applicability_condition",
              "P": "proposed_prevention", "U": "uncertainty_note"}
    for component in "CDKPU":
        reviewed["components"][component] = [{
            "atom_id": component + "000",
            "text": TEMPLATES[mechanism][component],
            "refs": [refs[component]],
            "status": status[component],
            "roles": list(roles),
        }]
    reviewed.update({
        "predicted_mechanism": mechanism,
        "culprits": [],
        "related_roles": list(roles),
        "reporter_output": "",
        "reporter_attempts": [],
        "report_cost": [],
        "report_protocol_version": REVIEW_PROTOCOL_VERSION,
        "provenance_validation": "deterministic development-only protocol review; source refs exist; no target outcome used",
        "reviewed_from_frozen_hash": experience["frozen_hash"],
        "reviewed_recipients": list(roles),
    })
    reviewed["frozen_hash"] = digest({k: v for k, v in reviewed.items() if k != "frozen_hash"})
    validate_frozen(reviewed)
    return reviewed
