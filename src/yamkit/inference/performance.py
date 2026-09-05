"""Offline deployment gate, independent of checkpoint mapping and forward passes."""

PHYSICAL_MODAL_ROLLOUT_REASON = (
    "Physical Modal rollout BLOCKED: integrated real-service queue performance is unvalidated. "
    "Recorded Molmo warm RPC (~1.48 s) exceeds its 1 s action horizon; "
    "saved-observation inference and fake-robot diagnostics remain available."
)


def physical_modal_status() -> dict:
    # No CLI/env override: a reviewed deployment qualification must change this
    # gate deliberately. Credentials, readiness and mapping are not qualification.
    return {"physical_modal_rollout_allowed": False,
            "physical_modal_rollout_reason": PHYSICAL_MODAL_ROLLOUT_REASON}


def require_physical_modal_rollout() -> None:
    raise ValueError(PHYSICAL_MODAL_ROLLOUT_REASON)
