"""Native LeRobot methods/processors, fake model tensors, no checkpoint or hardware."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.lerobot_types import TransitionKey
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.processor import UnnormalizerProcessorStep

from yamkit.inference.mapping import YAM_NAMES
from yamkit.pi05.executor import Pi05ReferenceExecutor


def native_policy():
    policy = PI05Policy.__new__(PI05Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(n_action_steps=30, rtc_config=None)
    policy.reset()
    return policy


def postprocessor():
    saved = json.loads((Path(__file__).parent / "fixtures/pi05_yam_normalization.json").read_text())
    return UnnormalizerProcessorStep(
        features={"action": PolicyFeature(FeatureType.ACTION, (14,))},
        norm_map={FeatureType.ACTION: NormalizationMode.QUANTILES}, stats={"action": saved["action"]})


def test_saved_native_unnormalizer_is_identical_whole_chunk_and_native_single_rows():
    post = postprocessor()
    raw = torch.linspace(-0.9, 0.9, 30 * 14).reshape(1, 30, 14)
    chunk = post({TransitionKey.ACTION: raw})[TransitionKey.ACTION]
    rows = torch.stack([post({TransitionKey.ACTION: raw[:, index]})[TransitionKey.ACTION]
                        for index in range(30)], dim=1)
    torch.testing.assert_close(chunk, rows, rtol=0, atol=0)


def test_executor_fifo_matches_actual_native_select_action_queue_for_all_rows():
    from threading import Event

    policy, post, native_calls = native_policy(), postprocessor(), []
    raw = torch.linspace(-0.8, 0.8, 30 * 14).reshape(1, 30, 14)

    def predict_native(batch):
        native_calls.append(batch["observation_index"])
        return raw.clone()

    policy.predict_action_chunk = predict_native
    expected = [post({TransitionKey.ACTION: policy.select_action({"observation_index": index})})[
                    TransitionKey.ACTION].squeeze(0).tolist() for index in range(90)]
    assert native_calls == [0, 30, 60]
    actual, clock = [], [1.0]
    processed_chunk = post({TransitionKey.ACTION: raw})[TransitionKey.ACTION].squeeze(0).tolist()

    def send(action, check):
        check()
        actual.append([action[name] for name in YAM_NAMES])
        return action

    def wait(delay):
        clock[0] += max(delay, 1e-12)

    engine = Pi05ReferenceExecutor(predict=lambda *_: processed_chunk, observe=dict, send=send,
                                  validate_target=lambda _: None, stop=Event(),
                                  clock=lambda: clock[0], wait=wait)
    metrics = engine.run(duration_s=5, max_chunks=3)
    np.testing.assert_array_equal(actual, expected)
    assert metrics["completed_rows"] == 90
    assert metrics["modified_commands"] == 0


def test_native_camera_preprocessing_keeps_checkpoint_order_and_letterbox():
    policy = native_policy()
    policy.anchor = torch.nn.Parameter(torch.zeros(1))
    policy.config.image_features = {f"observation.images.{name}": None for name in ("top", "left", "right")}
    policy.config.image_resolution = (224, 224)
    batch = {key: torch.full((1, 3, 360, 640), value)
             for key, value in zip(policy.config.image_features, (0.25, 0.5, 0.75), strict=True)}
    images, masks = policy._preprocess_images(batch)
    assert [tuple(image.shape) for image in images] == [(1, 3, 224, 224)] * 3
    assert [image[0, 0, 112, 112].item() for image in images] == [-0.5, 0.0, 0.5]
    assert all(image[0, 0, 0, 0].item() == -1.0 for image in images)
    assert all(mask.tolist() == [True] for mask in masks)
