"""Configuration resolved by LeRobot's genuine convention-based factories."""

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig


@PreTrainedConfig.register_subclass("yamkit_remote")
@dataclass
class YamkitRemoteConfig(PreTrainedConfig):
    profile: str = "molmoact2"
    backend: str = "modal"
    modal_app: str = ""
    external_service: str | None = None
    request_timeout_s: float = 10.0
    readiness_timeout_s: float = 120.0
    max_observation_age_s: float = 2.0
    center_crop: bool = False
    image_encoding: str = "rgb8"
    jpeg_quality: int = 85
    call_mode: str = "remote"
    execution_mode: str = "eager"
    controller_mode: str = "async"
    task: str = "put the red cube into the black container"
    prediction_queue_threshold: int | None = None
    supervised_confirmed: bool = False
    mapping_accepted: bool = False
    image_hw: tuple[int, int] = (480, 640)
    device: str = "cpu"
    action_feature_names: list[str] = field(default_factory=list)

    def __post_init__(self):
        import re

        from yamkit.inference.profiles import get_profile

        super().__post_init__()
        if self.backend not in ("modal", "external"):
            raise ValueError("Remote backend must be modal or external")
        if self.backend == "external":
            if (not self.external_service
                    or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", self.external_service)
                    or self.modal_app or self.call_mode != "http"):
                raise ValueError("External inference requires an exact service name and HTTP, without a Modal app")
        elif self.external_service is not None:
            raise ValueError("External service requires the external backend")
        if self.device != "cpu" or self.use_amp or self.use_peft:
            raise ValueError("The remote RPC proxy requires CPU, no AMP and no PEFT")
        if self.pretrained_path is not None:
            raise ValueError("The remote RPC proxy must not point to local model weights/processors")
        if not 0 < self.request_timeout_s <= 120 or not 0 < self.readiness_timeout_s <= 120:
            raise ValueError("Remote request/readiness timeouts must be in (0, 120] seconds")
        if not 0 < self.max_observation_age_s <= 120:
            raise ValueError("Maximum observation age must be in (0, 120] seconds")
        profile = get_profile(self.profile)
        if self.image_encoding not in ("jpeg", "rgb8"):
            raise ValueError("Remote images require jpeg or rgb8 encoding")
        if type(self.jpeg_quality) is not int or not 1 <= self.jpeg_quality <= 100:
            raise ValueError("JPEG quality must be an integer from 1 to 100")
        if self.call_mode not in ("remote", "spawn", "http"):
            raise ValueError("Remote call_mode must be remote, spawn or http")
        if self.execution_mode not in ("eager", "cuda_graph10"):
            raise ValueError("Unknown remote execution mode")
        if self.controller_mode not in ("async", "reference"):
            raise ValueError("Remote controller mode must be async or reference")
        if self.execution_mode == "cuda_graph10" and (
                self.call_mode != "http" or profile.id != "molmoact2" or self.image_encoding != "rgb8"):
            raise ValueError("Production graph10 requires MolmoAct2 over HTTP with raw RGB")
        if self.controller_mode == "reference" and (
                profile.id != "molmoact2" or self.call_mode != "http"
                or self.execution_mode != "cuda_graph10" or self.image_encoding != "rgb8" or self.center_crop):
            raise ValueError("Reference controller requires remote MolmoAct2 over HTTP graph10 with raw RGB and no crop")
        if not isinstance(self.task, str) or not self.task.strip() or len(self.task) > 2048:
            raise ValueError("The actual remote rollout task must contain 1–2048 characters")
        if self.prediction_queue_threshold is not None and (
                type(self.prediction_queue_threshold) is not int
                or not 0 <= self.prediction_queue_threshold <= profile.chunk_size):
            raise ValueError("Prediction queue threshold must be between zero and the chunk size")
        self.profile = profile.id
        self.action_feature_names = list(profile.action_names)
        self.input_features = {
            "observation.state": PolicyFeature(FeatureType.STATE, (len(profile.state_names),)),
        }
        self.set_image_shape(self.image_hw)
        self.output_features = {"action": PolicyFeature(FeatureType.ACTION, (len(profile.action_names),))}

    def set_image_shape(self, image_hw):
        from yamkit.inference.profiles import get_profile
        from yamkit.inference.protocol import MAX_IMAGE_HEIGHT, MAX_IMAGE_WIDTH

        if (len(image_hw) != 2 or any(type(value) is not int for value in image_hw)
                or not 1 <= image_hw[0] <= MAX_IMAGE_HEIGHT or not 1 <= image_hw[1] <= MAX_IMAGE_WIDTH):
            raise ValueError("Remote image dimensions exceed the protocol bounds")
        self.image_hw = tuple(image_hw)
        for name in get_profile(self.profile).image_keys:
            self.input_features[f"observation.images.{name}"] = PolicyFeature(FeatureType.VISUAL, (3, *image_hw))

    @property
    def observation_delta_indices(self):
        return [0]

    @property
    def action_delta_indices(self):
        from yamkit.inference.profiles import get_profile

        return list(range(get_profile(self.profile).chunk_size))

    @property
    def reward_delta_indices(self):
        return None

    def get_optimizer_preset(self):
        raise NotImplementedError("Remote inference proxies cannot be trained")

    def get_scheduler_preset(self):
        raise NotImplementedError("Remote inference proxies cannot be trained")

    def validate_features(self):
        from yamkit.inference.profiles import get_profile

        profile = get_profile(self.profile)
        if (self.output_features["action"].shape != (len(profile.action_names),)
                or self.action_feature_names != list(profile.action_names)):
            raise ValueError("Remote action schema/order does not match the selected profile")
