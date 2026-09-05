"""Shared, hardware-free CLI/UI inference option validation.

Importing this module never imports Modal, torch, a robot, or a camera.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class InferenceOptions:
    policy: str
    task: str = "pick up the object"
    backend: str = "local"
    device: str = "cpu"
    gpu: str = "L40S"
    rtc: bool = False
    async_chunks: bool = True
    center_crop: bool = False
    modal_app: str | None = None
    duration: float = 60.0
    fps: float = 30.0
    arms: tuple[str, ...] = ()

    def validate(self, *, motion: bool = False) -> InferenceOptions:
        if not self.policy.strip() or len(self.policy) > 512 or self.policy.startswith("-"):
            raise ValueError("provide a checkpoint path or a supported model preset")
        if not self.task.strip() or len(self.task) > 2048:
            raise ValueError("task must contain 1–2048 characters")
        if self.backend not in ("local", "modal"):
            raise ValueError("backend must be local or modal")
        if self.device not in ("cpu", "cuda", "mps"):
            raise ValueError("device must be cpu, cuda or mps")
        if not math.isfinite(self.duration) or not 0 < self.duration <= 3600:
            raise ValueError("duration must be finite and between 0 and 3600 seconds")
        if not math.isfinite(self.fps) or not 0 < self.fps <= 100:
            raise ValueError("fps must be finite and between 0 and 100")
        if len(self.arms) > 2 or len(set(self.arms)) != len(self.arms):
            raise ValueError("select one or two distinct follower arms")
        if self.modal_app is not None and not re.fullmatch(r"yamkit-vla-[a-z0-9-]{1,80}", self.modal_app):
            raise ValueError("Modal app must be a dedicated yamkit-vla-… app name")
        if self.backend == "modal":
            from .inference.profiles import get_profile

            profile = get_profile(self.policy)
            if self.gpu != "L40S":
                raise ValueError("this release supports one L40S per model pool")
            if self.rtc:
                raise ValueError("remote RTC guidance is unverified; use unguided async")
            if not self.async_chunks:
                raise ValueError("Modal rollout requires unguided async; synchronous RPC execution is disabled")
            if motion:
                if not profile.mapping_verified:
                    raise ValueError("physical YAM mapping is not validated: " + profile.mapping_note)
                if self.fps != profile.fps:
                    raise ValueError(f"this profile requires {profile.fps:g} Hz actions")
                if self.arms and len(self.arms) != 2:
                    raise ValueError("this profile requires both follower arms")
        else:
            if self.center_crop:
                raise ValueError("center crop is only available through the profiled Modal policy boundary")
            if motion:
                from .inference.profiles import get_profile

                try:
                    profile = get_profile(self.policy)
                except ValueError:
                    profile = None  # existing compatible local checkpoints keep their LeRobot path
                if profile is not None:
                    profile.require_robot_mapping()
                    if profile.id == "molmoact2":
                        raise ValueError("local MolmoAct2 rollout is blocked by LeRobot 0.6.1 relative-action/RTC "
                                         "gates; use Modal unguided async, or a compatible local checkpoint")
                    if self.fps != profile.fps or (self.arms and len(self.arms) != 2):
                        raise ValueError("the reviewed YAM profile requires both followers at 30 Hz")
        return self

    @property
    def operation_key(self) -> str:
        """Bind asynchronous completion to the exact options, never just the model name."""
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:20]

    def cli_args(self) -> list[str]:
        args = ["--policy", self.policy, "--task", self.task, "--backend", self.backend,
                "--device", self.device]
        if self.backend == "modal":
            args += ["--gpu", self.gpu]
            if self.modal_app:
                args += ["--modal-app", self.modal_app]
        if self.center_crop:
            args.append("--center-crop")
        return args
