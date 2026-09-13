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
    "Official OpenPI pi05_base is not physically qualified for YAM: its published assets do not "
    "establish YAM normalization, joint/gripper coordinates or an embodiment action mapping. "
    "No other checkpoint, borrowed normalization or first-14-output slicing is substituted. "
    "See docs/OPENPI_REFERENCE.md for the official-runtime offline evidence and missing contract. "
    "No hardware was opened."
)


def canonical_policy(policy: str) -> str:
    """Normalize only explicit reviewed aliases; never infer checkpoint compatibility."""
    return ALIASES.get(policy, policy)
