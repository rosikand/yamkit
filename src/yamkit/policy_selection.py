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
    "Official OpenPI pi05_base is not physically qualified for YAM: the offline experimental "
    "normalization/decoder produces out-of-range grippers and unqualified initial joint transitions "
    "on saved real inputs. A safe timebase/commitment and complete Stop/fault workflow are not qualified. "
    "No checkpoint substitution, output clipping or physical bypass is applied. "
    "See docs/OPENPI_YAM_EXPERIMENT_2026-09-13.md for native parity, command-bound evidence and next steps. "
    "No hardware was opened."
)


def canonical_policy(policy: str) -> str:
    """Normalize only explicit reviewed aliases; never infer checkpoint compatibility."""
    return ALIASES.get(policy, policy)
