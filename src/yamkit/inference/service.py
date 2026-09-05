"""Serialized, fresh-chunk LeRobot model runtime shared by CPU checks and Modal.

All heavyweight imports and downloads are explicit in ``load``. Saved processors are
loaded from the very same immutable checkpoint snapshot as the model. No synthetic
normalization statistics or changed physical feature dimensions are used here.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any

from .mapping import CAMERA_RENAME_MAP, center_crop_rgb
from .profiles import ModelProfile, get_profile
from .protocol import decode_image, validate_request, validate_response


class ModelRuntime:
    def __init__(self, profile: ModelProfile, policy: Any, pre: Any, post: Any, *, device: str,
                 load_s: float = 0.0, native_pre: Any = None):
        self.profile, self.policy = profile, policy
        self.pre, self.native_pre, self.post = pre, native_pre or pre, post
        self.device, self.load_s = device, load_s
        self.instance_id = str(uuid.uuid4())
        self._lock = threading.Lock()
        self._sequences: dict[str, int] = {}
        self._closed: set[str] = set()
        self.unclipped_post = post
        if profile.id == "molmoact2" and hasattr(post, "steps"):
            from lerobot.policies.molmoact2.processor_molmoact2 import MolmoAct2ClampActionProcessorStep
            from lerobot.processor import PolicyProcessorPipeline

            self.unclipped_post = PolicyProcessorPipeline(
                steps=[step for step in post.steps if not isinstance(step, MolmoAct2ClampActionProcessorStep)],
                name="diagnostic_unclipped_postprocessor", to_transition=post.to_transition,
                to_output=post.to_output,
            )

    @classmethod
    def load(cls, profile: str | ModelProfile, *, device: str = "cpu") -> ModelRuntime:
        from huggingface_hub import snapshot_download
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        profile = get_profile(profile)
        started = time.monotonic()
        snapshot = snapshot_download(profile.repo_id, revision=profile.revision,
                                     allow_patterns=["*.json", "*.safetensors"])
        cfg = PreTrainedConfig.from_pretrained(snapshot)
        cfg.device = device
        cfg.pretrained_path = snapshot
        cfg.pretrained_revision = profile.revision
        if cfg.type != profile.policy_type:
            raise ValueError("Checkpoint policy type does not match pinned profile")
        if tuple(cfg.input_features["observation.state"].shape) != (len(profile.state_names),):
            raise ValueError("Checkpoint state schema changed")
        if tuple(cfg.output_features["action"].shape) != (len(profile.action_names),):
            raise ValueError("Checkpoint action schema changed")
        dependency_patterns = ["*.json", "*.txt", "*.model", "*.jinja", "tokenizer*"]
        # The pinned LeRobot Molmo constructor first restores its nested HF backbone.
        # Preserve that upstream path, including its strict weight-key verification.
        if profile.id == "molmoact2":
            dependency_patterns.append("*.safetensors")
        dependency = snapshot_download(profile.dependency_repo, revision=profile.dependency_revision,
                                       allow_patterns=dependency_patterns)
        overrides: dict[str, dict] = {"device_processor": {"device": device}}
        if profile.id == "smolvla":
            cfg.vlm_model_name = dependency
            cfg.load_vlm_weights = False  # full pinned LeRobot weights are restored below
            overrides["tokenizer_processor"] = {"tokenizer_name": dependency}
        elif profile.id == "molmoact2":
            cfg.checkpoint_path = dependency
            cfg.checkpoint_revision = profile.dependency_revision
            cfg.enable_inference_cuda_graph = False
            cfg.model_dtype = "bfloat16" if device.startswith("cuda") else "float32"
            overrides["molmoact2_pack_inputs"] = {
                "checkpoint_path": dependency, "checkpoint_revision": profile.dependency_revision,
                "allow_image_key_fallback": False,
            }
        else:
            cfg.dtype = "bfloat16" if device.startswith("cuda") else "float32"
            cfg.compile_model = False
            overrides["tokenizer_processor"] = {"tokenizer_name": dependency}
        # Never enable RTC merely because the underlying policy implements its denoiser:
        # robot-unit continuation to normalized/relative coordinates must first be qualified.
        if hasattr(cfg, "rtc_config"):
            cfg.rtc_config = None
        policy = get_policy_class(cfg.type).from_pretrained(snapshot, config=cfg, strict=True)
        policy.eval()
        native_pre, post = make_pre_post_processors(
            cfg, pretrained_path=snapshot, pretrained_revision=profile.revision,
            preprocessor_overrides=overrides,
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        pre = native_pre
        if profile.id == "molmoact2":
            pre, _ = make_pre_post_processors(
                cfg, pretrained_path=snapshot, pretrained_revision=profile.revision,
                preprocessor_overrides={**overrides, "rename_observations_processor": {
                    "rename_map": CAMERA_RENAME_MAP,
                }}, postprocessor_overrides={"device_processor": {"device": "cpu"}},
            )
        return cls(profile, policy, pre, post, device=device, load_s=time.monotonic() - started,
                   native_pre=native_pre)

    def ready(self) -> dict:
        return {**self.profile.metadata(), "ready": True, "instance_id": self.instance_id,
                "device": self.device, "load_s": self.load_s, "fresh_chunk": True,
                "supports_rtc": False, "continuation": "unsupported", "image_encoding": "rgb8",
                "saved_processors": True, "session_state": "reset for every fresh chunk"}

    def reset(self, session_id: str) -> None:
        """Retire a session. The client must create a new ID and reject any late response."""
        if not isinstance(session_id, str) or not 1 <= len(session_id) <= 128:
            raise ValueError("Invalid session id")
        with self._lock:
            if len(self._closed) >= 1024:
                raise RuntimeError("Session retirement bound reached; restart this pool")
            self._closed.add(session_id)
            self.policy.reset()

    def predict_chunk(self, request: dict) -> dict:
        import torch

        validate_request(request, self.profile)
        started = time.monotonic()
        timeout = request["timeout_s"]
        if not self._lock.acquire(timeout=timeout):
            raise TimeoutError("Inference queue deadline exceeded")
        try:
            acquired = time.monotonic()
            session, sequence = request["session_id"], request["sequence_id"]
            if session in self._closed or sequence <= self._sequences.get(session, -1):
                raise ValueError("Retired session or duplicate/out-of-order sequence")
            if session not in self._sequences and len(self._sequences) >= 1024:
                raise RuntimeError("Session bound reached; restart this pool")
            self._sequences[session] = sequence
            if time.monotonic() - started >= timeout:
                raise TimeoutError("Inference request expired before execution")
            self.policy.reset()
            for processor in (self.pre, self.native_pre, self.post):
                reset = getattr(processor, "reset", None)
                if reset:
                    reset()
            frame = {"observation.state": torch.tensor(request["state"], dtype=torch.float32),
                     "task": request["task"]}
            transforms = {}
            image_started = time.monotonic()
            for name, encoded in request["images"].items():
                pixels, transform = center_crop_rgb(decode_image(encoded), request.get("crop", "none"))
                transforms[name] = transform
                frame[f"observation.images.{name}"] = torch.from_numpy(pixels.copy()).permute(2, 0, 1).float() / 255
            image_end = time.monotonic()
            pre = self.native_pre if request.get("mode", "robot") == "native_fixture" else self.pre
            prepared = pre(frame)
            pre_end = time.monotonic()
            # Do not wrap in inference_mode: upstream guidance, when qualified in a
            # future version, needs torch.enable_grad inside its denoising loop.
            with torch.no_grad():
                raw = self.policy.predict_action_chunk(prepared)
                if self.device.startswith("cuda"):
                    torch.cuda.synchronize()
                infer_end = time.monotonic()
                unclipped = self.unclipped_post(raw.clone()) if request.get("mode") in (
                    "saved_probe", "live_probe",
                ) else None
                processed = self.post(raw)
            post_end = time.monotonic()
            if processed.ndim != 3 or processed.shape[0] != 1:
                raise ValueError("Policy must produce exactly one B×T×D action chunk")
            decode_started = time.monotonic()
            chunk = processed[0].detach().float().cpu().tolist()
            decode_end = time.monotonic()
            response = {key: request[key] for key in (
                "protocol_version", "profile", "model_revision", "session_id", "sequence_id", "observation_time",
            )}
            response.update(
                action_units="checkpoint_native" if request.get("mode", "robot") == "native_fixture" else "robot",
                action_names=list(self.profile.action_names), chunk=chunk,
                timing={"queue_wait_s": acquired - started,
                        "modal_queue_s": None,  # dispatch before entry is not visible to this runtime
                        "image_decode_transform_s": image_end - image_started,
                        "state_reset_s": image_started - acquired,
                        "preprocess_s": pre_end - image_end, "inference_s": infer_end - pre_end,
                        "postprocess_s": post_end - infer_end,
                        "response_conversion_s": decode_end - decode_started,
                        "total_s": decode_end - started},
                transforms=transforms, instance_id=self.instance_id,
                payload_bytes=sum(len(image["data"]) for image in request["images"].values()),
                saved_postprocessor_clamp=self.profile.id == "molmoact2",
            )
            if request.get("mode") in ("saved_probe", "live_probe"):
                assert unclipped is not None
                response["unclipped_chunk"] = unclipped[0].detach().float().cpu().tolist()
                response["unclipped_action_units"] = "robot"
                response["unclipped_note"] = "Diagnostic only: saved numerical/frame transforms, " \
                    "excluding the saved Molmo normalized-action clamp; never execute this diagnostic chunk."
            validate_response(response, request, self.profile)
            if time.monotonic() - started >= timeout:
                raise TimeoutError("Inference deadline exceeded; result discarded")
            return response
        finally:
            self._lock.release()
