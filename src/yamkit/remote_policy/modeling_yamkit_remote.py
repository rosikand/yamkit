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
                          shutdown_event=getattr(config, "_session_shutdown_event", None),
                          call_mode=config.call_mode)


class YamkitRemotePolicy(PreTrainedPolicy):
    config_class = YamkitRemoteConfig
    name = "yamkit_remote"

    def __init__(self, config: YamkitRemoteConfig, **kwargs):
        from yamkit.inference.performance import require_physical_modal_rollout
        from yamkit.inference.profiles import get_profile
        from yamkit.inference.qualification import require_runner_context, settings_from_policy

        require_physical_modal_rollout(lambda: settings_from_policy(config),
                                       supervised_confirmed=config.supervised_confirmed,
                                       mapping_accepted=config.mapping_accepted)
        require_runner_context()
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
        self._last_requested_observation_time = None
        self._last_prediction_timing = {}
        readiness_started = time.monotonic()
        self.metadata = self.transport.ready(config.readiness_timeout_s)
        self.validate_readiness()
        self.warmup_s = 0.0
        if self.metadata.get("prediction_count") == 0:
            # First-forward initialization must complete before hardware connects.
            # Its checkpoint-native fixture is never inserted into an action queue.
            from yamkit.inference.protocol import encode_image, native_fixture_request, validate_response

            request = native_fixture_request(self.profile, encoding=config.image_encoding,
                                             quality=config.jpeg_quality,
                                             crop="center_16_9" if config.center_crop else "none")
            for robot_name, native_name in zip(self.profile.image_keys, self.profile.native_image_keys, strict=True):
                height, width = config.input_features[f"observation.images.{robot_name}"].shape[-2:]
                request["images"][native_name] = encode_image(np.zeros((height, width, 3), dtype=np.uint8),
                                                              encoding=config.image_encoding, quality=config.jpeg_quality)
            remaining = config.readiness_timeout_s - (time.monotonic() - readiness_started)
            if remaining <= 0:
                self.close()
                raise RemoteFault("Remote readiness deadline expired before model warmup")
            request["timeout_s"] = remaining
            warmed_at = time.monotonic()
            try:
                response = self.transport.predict_chunk(request, remaining)
                validate_response(response, request, self.profile)
                if response.get("instance_id") != self.metadata["instance_id"]:
                    raise RemoteFault("Remote container changed during model warmup")
                self.validate_readiness()  # Recheck Stop after the blocking warmup.
            except Exception:
                self.close()
                raise
            self.warmup_s = time.monotonic() - warmed_at
        require_physical_modal_rollout(lambda: settings_from_policy(config, self.metadata),
                                       supervised_confirmed=config.supervised_confirmed,
                                       mapping_accepted=config.mapping_accepted)
        self.readiness_s = time.monotonic() - readiness_started
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
                    "supports_rtc": False}
        for key, value in expected.items():
            actual = self.metadata.get(key)
            if isinstance(value, list) and isinstance(actual, (list, tuple)):
                actual = list(actual)
            if actual != value:
                raise RemoteFault(f"Remote readiness mismatch for {key}")
        encodings = self.metadata.get("supported_image_encodings", [self.metadata.get("image_encoding")])
        if self.config.image_encoding not in encodings:
            raise RemoteFault("Remote readiness does not support the selected image encoding")
        if not self.profile.mapping_verified or self.metadata.get("mapping_verified") is not True:
            raise RemoteFault("Physical YAM mapping is unverified; only hardware-free native fixtures are supported")
        if not isinstance(self.metadata.get("instance_id"), str) or not self.metadata["instance_id"]:
            raise RemoteFault("Remote readiness requires a container instance identity")
        if type(self.metadata.get("prediction_count")) is not int or self.metadata["prediction_count"] < 0:
            raise RemoteFault("Remote readiness requires a valid completed prediction count")

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
        self._last_requested_observation_time = None
        self.session.reset()

    def close(self):
        self._actions.clear()
        self.session.close()

    def supports_rtc(self):
        # Unguided background inference is not denoising guidance.
        return False

    def predict_action_chunk(self, batch, *, inference_delay=0, prev_chunk_left_over=None, **kwargs):
        self._last_prediction_timing = {}
        self._last_requested_observation_time = self._observation_time
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
        camera_timings = {}
        for name in self.profile.image_keys:
            transform_started = time.monotonic()
            value = batch[f"observation.images.{name}"].detach().cpu()
            if value.ndim != 4 or value.shape[0] != 1 or value.shape[1] != 3:
                raise RemoteFault("Remote images must have shape [1, 3, height, width]")
            if tuple(value.shape[1:]) != tuple(self.config.input_features[f"observation.images.{name}"].shape):
                raise RemoteFault("Remote image dimensions differ from the qualified policy boundary")
            # This proxy is CPU-only. NumPy shares the tensor's storage and avoids
            # three threaded torch reductions before the existing NumPy encoding.
            pixels = value[0].permute(1, 2, 0).numpy()
            if not np.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1:
                raise RemoteFault("Remote images must be finite RGB in [0, 1]")
            rgb = (pixels * 255).round().astype(np.uint8)
            encoded_at = time.monotonic()
            images[name] = encode_image(rgb, encoding=self.config.image_encoding, quality=self.config.jpeg_quality)
            encode_s = time.monotonic() - encoded_at
            camera_timings[name] = {"tensor_transform_s": encoded_at - transform_started,
                                    "image_encode_s": encode_s,
                                    "jpeg_encode_s": encode_s if self.config.image_encoding == "jpeg" else 0.0,
                                    "payload_bytes": len(images[name]["data"])}
            transform_s += camera_timings[name]["tensor_transform_s"]
            serialization_s += camera_timings[name]["image_encode_s"]
        task = batch.get("task", [""])
        if not isinstance(task, (list, tuple)) or len(task) != 1 or not isinstance(task[0], str):
            raise RemoteFault("Remote inference requires exactly one task string")
        observation_time = self._observation_time if self._observation_time is not None else started
        self._last_prediction_timing = {"encoding_s": time.monotonic() - started,
                                       "image_tensor_transform_s": transform_s,
                                       "image_serialization_s": serialization_s,
                                       "jpeg_encoding_s": sum(t["jpeg_encode_s"] for t in camera_timings.values()),
                                       "per_camera_timing": camera_timings}
        result = self.session.predict(state=state[0].tolist(), images=images, task=task[0],
                                      observation_time=observation_time,
                                      crop="center_16_9" if self.config.center_crop else "none")
        self.session.samples[-1].update(self._last_prediction_timing)
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
