"""Native PI05 processors/model behind the existing inert HTTP chunk protocol.

No Molmo processor, CUDA graph manager or interpolation is used. Heavy imports,
downloads and GPU allocation occur only in the explicit load method.
"""

from __future__ import annotations

import time
from pathlib import Path

from yamkit.inference.service import ModelRuntime

from .contract import CAMERA_MAP, CONTRACT, PROFILE, build_id, validate_snapshot


def pinned_tokenizer_snapshot(snapshot_download) -> str:
    """Never substitute an unreviewed tokenizer or bypass publisher access."""
    from huggingface_hub.errors import GatedRepoError

    try:
        return snapshot_download(PROFILE.dependency_repo, revision=PROFILE.dependency_revision,
                                 allow_patterns=["*.json", "*.txt", "*.model", "*.jinja", "tokenizer*"])
    except GatedRepoError:
        raise ValueError(
            "π0.5 requires access to the pinned google/paligemma-3b-pt-224 tokenizer. "
            "Accept its publisher terms at https://huggingface.co/google/paligemma-3b-pt-224 "
            "with the runtime's Hugging Face account, then use yamkit hub login on the GPU checkout. "
            "No substitute tokenizer was loaded; π0.5 is not qualified."
        ) from None


def restore_native_weights(policy, state_dict) -> None:
    """Native key compatibility, strict restore, and NO upstream silent fallback.

    LeRobot 0.6.1 PI05Policy.from_pretrained catches even strict loading errors and
    returns the model. The native constructor/key mapper are retained, but a
    failed checkpoint restore must never become an apparently ready policy.
    """
    fixed = policy._fix_pytorch_state_dict_keys(state_dict, policy.config)
    remapped = {}
    for key, value in fixed.items():
        name = key if key.startswith("model.") else "model." + key
        if name in remapped:
            raise ValueError("Ambiguous duplicate π0.5 checkpoint weight after native key mapping")
        remapped[name] = value
    outcome = policy.load_state_dict(remapped, strict=True)
    if outcome.missing_keys or outcome.unexpected_keys:
        raise ValueError("π0.5 checkpoint was not loaded completely")


class Pi05Runtime(ModelRuntime):
    """Faithful independent eager PI05 runtime, with native saved processors."""

    @classmethod
    def load(cls, *, device: str = "cuda:0") -> Pi05Runtime:
        from huggingface_hub import snapshot_download
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from safetensors.torch import load_file

        if not device.startswith("cuda"):
            raise ValueError("The reviewed YAM π0.5 runtime requires CUDA and native bfloat16")
        started = time.monotonic()
        snapshot = Path(snapshot_download(PROFILE.repo_id, revision=PROFILE.revision,
                                         allow_patterns=["*.json", "*.safetensors"]))
        validate_snapshot(snapshot)
        dependency = pinned_tokenizer_snapshot(snapshot_download)
        config = PreTrainedConfig.from_pretrained(snapshot)
        config.device = device
        config.pretrained_path = snapshot
        config.pretrained_revision = PROFILE.revision
        # Eager is an explicit execution variant; no denoising/dtype/row changes.
        # Compilation is a native optional performance setting, qualified separately.
        config.compile_model = False
        policy = PI05Policy(config)
        restore_native_weights(policy, load_file(str(snapshot / "model.safetensors")))
        policy.eval()
        overrides = {"device_processor": {"device": device},
                     "tokenizer_processor": {"tokenizer_name": dependency}}
        native_pre, post = make_pre_post_processors(
            config, pretrained_path=snapshot, pretrained_revision=PROFILE.revision,
            preprocessor_overrides=overrides,
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        pre, _ = make_pre_post_processors(
            config, pretrained_path=snapshot, pretrained_revision=PROFILE.revision,
            preprocessor_overrides={**overrides, "rename_observations_processor": {"rename_map": CAMERA_MAP}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        result = cls(PROFILE, policy, pre, post, device=device, native_pre=native_pre,
                     execution_mode="eager", load_s=time.monotonic() - started)
        result._model_metadata["configured_model_dtype"] = config.dtype
        result._model_metadata["strict_weights_restored"] = True
        return result

    def _execution_identity(self) -> dict:
        return {**super()._execution_identity(), "controller_contract": CONTRACT["id"],
                "native_rtc_enabled": False, "strict_weights_restored": True,
                "compile_model": False}

    def ready(self) -> dict:
        return {**super().ready(), "controller_contract": dict(CONTRACT), "pi05_build_id": build_id(),
                "physical_ready": False, "qualification_hardware_tested": False}

    def predict_chunk(self, request: dict) -> dict:
        if request.get("crop", "none") != "none" or request.get("continuation") is not None:
            raise ValueError("π0.5 reference uses native image padding, no boundary crop or RTC continuation")
        response = super().predict_chunk(request)
        if len(response["chunk"]) != PROFILE.chunk_size:
            raise ValueError("π0.5 must return all 30 native action rows")
        response["controller_contract"] = CONTRACT["id"]
        response["pi05_build_id"] = build_id()
        return response
