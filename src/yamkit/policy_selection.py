"""Public configured-policy names, kept outside frozen legacy model profiles."""

from __future__ import annotations

ALIASES = {
    "molmoact2": "molmoact2",
    "lerobot/MolmoAct2-BimanualYAM-LeRobot": "molmoact2",
    "pi05": "pi05-yam",  # Preserve the previously documented configured CLI alias.
    "pi05-yam": "pi05-yam",
    "pi05_yam": "pi05-yam",
    "pi05-base": "pi05-base",
    "pi05_base": "pi05-base",
}

OPENPI_YAM_BLOCKER = (
    "Official OpenPI pi05_base is not enabled in this UI workflow. Use the dedicated terminal "
    "workflow: yamkit rollout --backend lambda --policy pi05_base --task TASK --duration 60. "
    "That path uses genuine frozen official weights with a documented experimental YAM adapter, "
    "automatic software qualification and fresh on-site confirmation before motion. "
    "See docs/OPENPI_BASE_YAM.md. This UI does not substitute pi05_yam or authorize motion. "
    "No hardware was opened."
)


def canonical_policy(policy: str) -> str:
    """Normalize only explicit reviewed aliases; never infer checkpoint compatibility."""
    return ALIASES.get(policy, policy)
