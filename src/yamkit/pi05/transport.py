"""PI-specific identity binding over the unchanged bounded HTTP transport."""

from __future__ import annotations

from yamkit.inference.http_transport import HttpTransport
from yamkit.inference.protocol import validate_request, validate_response

from .contract import ACTION_TRANSFORM, CONTRACT, CONTRACT_ID, PROFILE, build_id


def validate_readiness(metadata: dict) -> None:
    """Never confuse the old generic pi05 base fixture with the YAM checkpoint."""
    if (not isinstance(metadata, dict) or not isinstance(metadata.get("execution_identity"), dict)
            or not isinstance(metadata.get("controller_contract"), dict)):
        raise ValueError("π0.5 readiness must contain a native execution identity and controller contract")  # noqa: TRY004 — uniform wire rejection
    identity = metadata.get("execution_identity", {})
    if (metadata.get("profile") != PROFILE.id or metadata.get("model_revision") != PROFILE.revision
            or metadata.get("model") != PROFILE.repo_id or metadata.get("execution_mode") != "eager"
            or metadata.get("pi05_build_id") != build_id()
            or metadata.get("controller_contract", {}).get("id") != CONTRACT_ID
            or metadata["controller_contract"].get("version") != CONTRACT["version"]
            or metadata["controller_contract"].get("action_transform") != ACTION_TRANSFORM
            or metadata.get("ready") is not True or identity.get("profile") != PROFILE.id
            or identity.get("model_revision") != PROFILE.revision
            or identity.get("controller_contract") != CONTRACT_ID
            or identity.get("action_transform") != ACTION_TRANSFORM
            or identity.get("model_dtype") != "bfloat16" or identity.get("chunk_size") != 30
            or identity.get("action_width") != 14 or identity.get("num_inference_steps") != 10
            or identity.get("native_rtc_enabled") is not False
            or identity.get("strict_weights_restored") is not True
            or identity.get("compile_model") is not False
            or not isinstance(metadata.get("instance_id"), str) or not metadata["instance_id"]):
        raise ValueError("π0.5 service does not match this pinned native YAM runtime; prepare it again")


class Pi05Transport(HttpTransport):
    """Reuse deadlines/cancellation/wire without adding a model to MA2's catalog.

    The base constructor's profile label is transport metadata only; it never
    selects or transforms a model. Its existing pi05 family is reused, then this
    subclass binds every request, response and readiness to pi05-yam explicitly.
    """

    def __init__(self, app_name: str, **kwargs):
        super().__init__(app_name, "pi05", **kwargs)
        self.profile = PROFILE.id
        self.instance_id = None

    def ready(self, timeout_s: float = 10.0) -> dict:
        metadata = super().ready(timeout_s)
        try:
            validate_readiness(metadata)
            if self.instance_id is not None and metadata["instance_id"] != self.instance_id:
                raise ValueError("π0.5 model instance changed; prepare and qualify again")
            self.instance_id = metadata["instance_id"]
            return metadata
        except ValueError:
            self.close()
            raise

    def predict_chunk(self, request: dict, timeout_s: float) -> dict:
        if self.instance_id is None:
            raise ValueError("Verify the π0.5 service identity before requesting inference")
        validate_request(request, PROFILE)
        response = super().predict_chunk(request, timeout_s)
        validate_response(response, request, PROFILE)
        if (response.get("instance_id") != self.instance_id or response.get("pi05_build_id") != build_id()
                or response.get("action_transform") != ACTION_TRANSFORM
                or response.get("controller_contract") != CONTRACT_ID or len(response["chunk"]) != 30):
            self.cancel()
            raise ValueError("π0.5 response instance, source, contract or full chunk changed")
        return response
