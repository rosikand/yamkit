"""Configuration resolved by LeRobot's genuine convention-based factories."""

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig


@PreTrainedConfig.register_subclass("yamkit_remote")
@dataclass
class YamkitRemoteConfig(PreTrainedConfig):
    profile: str = "molmoact2"
    modal_app: str = ""
    request_timeout_s: float = 10.0
    readiness_timeout_s: float = 120.0
    max_observation_age_s: float = 2.0
    center_crop: bool = False
    device: str = "cpu"
    action_feature_names: list[str] = field(default_factory=list)

    def __post_init__(self):
        from yamkit.inference.profiles import get_profile

        super().__post_init__()
        if self.device != "cpu" or self.use_amp or self.use_peft:
            raise ValueError("The remote RPC proxy requires CPU, no AMP and no PEFT")
        if self.pretrained_path is not None:
            raise ValueError("The remote RPC proxy must not point to local model weights/processors")
        if not 0 < self.request_timeout_s <= 120 or not 0 < self.readiness_timeout_s <= 120:
            raise ValueError("Remote request/readiness timeouts must be in (0, 120] seconds")
        if not 0 < self.max_observation_age_s <= 120:
            raise ValueError("Maximum observation age must be in (0, 120] seconds")
        profile = get_profile(self.profile)
        self.profile = profile.id
        self.action_feature_names = list(profile.action_names)
        self.input_features = {
            "observation.state": PolicyFeature(FeatureType.STATE, (len(profile.state_names),)),
            **{f"observation.images.{name}": PolicyFeature(FeatureType.VISUAL, (3, 480, 640))
               for name in profile.image_keys},
        }
        self.output_features = {"action": PolicyFeature(FeatureType.ACTION, (len(profile.action_names),))}

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
