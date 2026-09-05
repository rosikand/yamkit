"""Weightless LeRobot policy. Prediction always requests a fresh server chunk."""

import time
from collections import deque

import numpy as np
import torch
from lerobot.policies.pretrained import PreTrainedPolicy

from yamkit.inference.client import InvalidatedRequest, ModalTransport, RemoteFault, RemoteSession

from .configuration_yamkit_remote import YamkitRemoteConfig


def make_transport(config):
    """Dependency seam for hardware/cloud-free factory and context tests."""
    return ModalTransport(config.modal_app, config.profile,
                          shutdown_event=getattr(config, "_session_shutdown_event", None))


class YamkitRemotePolicy(PreTrainedPolicy):
    config_class = YamkitRemoteConfig
    name = "yamkit_remote"

    def __init__(self, config: YamkitRemoteConfig, **kwargs):
        from yamkit.inference.performance import require_physical_modal_rollout
        from yamkit.inference.profiles import get_profile

        require_physical_modal_rollout()  # also guards direct upstream proxy construction
        super().__init__(config)
        config.validate_features()
        self.profile = get_profile(config.profile)
        self.profile.require_robot_mapping()
        self.transport = make_transport(config)
        self.on_fault = None
        self.on_prediction_start = None
        self.on_prediction_end = None
        self.session = RemoteSession(self.transport, self.profile, timeout_s=config.request_timeout_s,
                                     max_observation_age_s=config.max_observation_age_s)
        self._actions = deque(maxlen=self.profile.chunk_size)
        self._actions_expire_at = None
        self._observation_time = None
        self.metadata = self.transport.ready(config.readiness_timeout_s)
        self.validate_readiness()
        self.session.instance_id = self.metadata["instance_id"]

    def validate_readiness(self):
        stop = getattr(self.config, "_session_shutdown_event", None)
        if stop is not None and stop.is_set():
            self.close()
            raise RemoteFault("Local Stop invalidated remote readiness before hardware activation")
        expected = {"profile": self.profile.id, "model_revision": self.profile.revision,
                    "action_names": list(self.profile.action_names),
                    "state_names": list(self.profile.state_names), "image_keys": list(self.profile.image_keys),
                    "action_units": "robot", "fps": self.profile.fps,
                    "max_chunk_steps": self.profile.chunk_size, "lerobot_version": "0.6.1",
                    "ready": True, "fresh_chunk": True, "saved_processors": True,
                    "supports_rtc": False, "image_encoding": "rgb8"}
        for key, value in expected.items():
            actual = self.metadata.get(key)
            if isinstance(value, list) and isinstance(actual, (list, tuple)):
                actual = list(actual)
            if actual != value:
                raise RemoteFault(f"Remote readiness mismatch for {key}")
        if not self.profile.mapping_verified or self.metadata.get("mapping_verified") is not True:
            raise RemoteFault("Physical YAM mapping is unverified; only hardware-free native fixtures are supported")
        if not isinstance(self.metadata.get("instance_id"), str) or not self.metadata["instance_id"]:
            raise RemoteFault("Remote readiness requires a container instance identity")

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path=None, *, config=None, **kwargs):
        # The upstream context always uses from_pretrained, including this weightless policy.
        if config is None or pretrained_name_or_path is not None:
            raise ValueError("Remote policy requires an explicit YamkitRemoteConfig and no weight path")
        return cls(config)

    def reset(self):
        self._actions.clear()
        self._actions_expire_at = None
        self._observation_time = None
        self.session.reset()

    def close(self):
        self._actions.clear()
        self.session.close()

    def supports_rtc(self):
        # Unguided background inference is not denoising guidance.
        return False

    def predict_action_chunk(self, batch, *, inference_delay=0, prev_chunk_left_over=None, **kwargs):
        event = self.on_prediction_start() if self.on_prediction_start is not None else None
        try:
            result = self._predict_chunk(batch, **kwargs)
            if self.on_prediction_end is not None:
                self.on_prediction_end(event, None)
            return result
        except InvalidatedRequest:
            if self.on_prediction_end is not None:
                self.on_prediction_end(event, "invalidated")
            raise
        except Exception:
            if self.on_prediction_end is not None:
                self.on_prediction_end(event, "prediction_failed")
            if self.on_fault is not None:
                self.on_fault()
            raise

    def _predict_chunk(self, batch, **kwargs):
        from yamkit.inference.protocol import encode_image

        # The upstream unguided worker still supplies its queue tail and estimated
        # delay. They are deliberately not forwarded into an unverified denoiser.
        if kwargs:
            raise ValueError("Unsupported remote chunk prediction arguments")
        started = time.monotonic()
        state = batch["observation.state"].detach().cpu()
        if state.shape != (1, len(self.profile.state_names)) or not torch.isfinite(state).all():
            raise RemoteFault("Remote state must have exactly one finite ordered observation")
        images = {}
        transform_s = 0.0
        serialization_s = 0.0
        for name in self.profile.image_keys:
            transform_started = time.monotonic()
            value = batch[f"observation.images.{name}"].detach().cpu()
            if value.ndim != 4 or value.shape[0] != 1 or value.shape[1] != 3:
                raise RemoteFault("Remote images must have shape [1, 3, height, width]")
            if not torch.isfinite(value).all() or value.min() < 0 or value.max() > 1:
                raise RemoteFault("Remote images must be finite RGB in [0, 1]")
            rgb = (value[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            encoded_at = time.monotonic()
            transform_s += encoded_at - transform_started
            images[name] = encode_image(rgb)
            serialization_s += time.monotonic() - encoded_at
        task = batch.get("task", [""])
        if not isinstance(task, (list, tuple)) or len(task) != 1 or not isinstance(task[0], str):
            raise RemoteFault("Remote inference requires exactly one task string")
        observation_time = self._observation_time if self._observation_time is not None else started
        encoding_s = time.monotonic() - started
        result = self.session.predict(state=state[0].tolist(), images=images, task=task[0],
                                      observation_time=observation_time,
                                      crop="center_16_9" if self.config.center_crop else "none")
        self.session.samples[-1]["encoding_s"] = encoding_s
        self.session.samples[-1]["image_tensor_transform_s"] = transform_s
        self.session.samples[-1]["image_serialization_s"] = serialization_s
        self._actions_expire_at = observation_time + self.config.max_observation_age_s
        decoded_at = time.monotonic()
        actions = torch.tensor(result["chunk"], dtype=torch.float32).unsqueeze(0)
        self.session.samples[-1]["response_tensor_decode_s"] = time.monotonic() - decoded_at
        return actions

    def select_action(self, batch, **kwargs):
        if self._actions and self._actions_expire_at is not None and time.monotonic() >= self._actions_expire_at:
            self.close()
            raise RemoteFault("Cached remote action chunk expired")
        if not self._actions:
            self._actions.extend(self.predict_action_chunk(batch, **kwargs).transpose(0, 1))
        return self._actions.popleft()

    def get_optim_params(self):
        raise NotImplementedError("Remote inference only")

    def forward(self, batch):
        raise NotImplementedError("Remote inference only")
