"""The explicit MolmoAct2/YAM name map; vector order is never inferred by sorting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

YAM_NAMES = tuple(
    name
    for side in ("left", "right")
    for name in (*(f"{side}_joint_{i}.pos" for i in range(1, 7)), f"{side}_gripper.pos")
)
MOLMO_NAMES = tuple(
    name
    for side in ("left", "right")
    for name in (*(f"{side}_joint_{i}.pos" for i in range(6)), f"{side}_gripper.pos")
)
YAM_TO_DATASET = dict(zip(YAM_NAMES, MOLMO_NAMES, strict=True))
DATASET_TO_YAM = dict(zip(MOLMO_NAMES, YAM_NAMES, strict=True))
CAMERA_RENAME_MAP = {
    "observation.images.top": "observation.images.top",
    "observation.images.left_wrist": "observation.images.left",
    "observation.images.right_wrist": "observation.images.right",
}


def validate_order(names: Sequence[str], expected: Sequence[str]) -> None:
    if tuple(names) != tuple(expected):
        raise ValueError(f"Ordered feature names must be exactly {tuple(expected)!r}; got {tuple(names)!r}")


def _rename(values: Mapping[str, float], source: tuple[str, ...], target: tuple[str, ...]) -> dict[str, float]:
    if set(values) != set(source):
        raise ValueError("Missing or extra joints in mapping; never pad/truncate a physical vector")
    vector = np.asarray([values[name] for name in source], dtype=np.float64)
    if not np.isfinite(vector).all():
        raise ValueError("Joint vector contains nonfinite values")
    return dict(zip(target, vector.tolist(), strict=True))


def yamkit_to_dataset(values: Mapping[str, float]) -> dict[str, float]:
    return _rename(values, YAM_NAMES, MOLMO_NAMES)


def dataset_to_yamkit(values: Mapping[str, float]) -> dict[str, float]:
    return _rename(values, MOLMO_NAMES, YAM_NAMES)


def center_crop_rgb(image: np.ndarray, crop: str = "none") -> tuple[np.ndarray, dict]:
    """An explicit boundary experiment; never changes recording or claims matching extrinsics."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Images must be HWC uint8 RGB")
    h, w, _ = image.shape
    if crop == "none":
        return image, {"crop": "none", "input_hw": [h, w], "output_hw": [h, w]}
    if crop != "center_16_9":
        raise ValueError("crop must be none or center_16_9")
    crop_h, crop_w = min(h, w * 9 // 16), min(w, h * 16 // 9)
    y, x = (h - crop_h) // 2, (w - crop_w) // 2
    return image[y:y + crop_h, x:x + crop_w].copy(), {
        "crop": crop, "input_hw": [h, w], "output_hw": [crop_h, crop_w], "offset_yx": [y, x],
    }
