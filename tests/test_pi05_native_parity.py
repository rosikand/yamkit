"""Native LeRobot methods/processors, fake model tensors, no checkpoint or hardware."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
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


class TimingWorld:
    """Identical simulated observation, model and send costs for both runners."""

    def __init__(self, observation_s, inference_s, send_s):
        self.now = 100.0
        self.observation_s, self.inference_s, self.send_s = observation_s, inference_s, send_s
        self.observed, self.predicted, self.sent = [], [], []

    def clock(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds

    def observe(self):
        index = len(self.observed)
        self.observed.append(self.now)
        self.advance(self.observation_s[index % len(self.observation_s)])
        return {"observation_index": index}

    def infer(self, observation):
        self.predicted.append(observation["observation_index"])
        self.advance(self.inference_s[(len(self.predicted) - 1) % len(self.inference_s)])

    def send(self, target):
        self.sent.append((self.now, list(target)))
        self.advance(self.send_s[(len(self.sent) - 1) % len(self.send_s)])


@pytest.mark.parametrize("observation_s,inference_s,send_s", [
    ([0.003], [0.005], [0.002]),  # Entire first tick fits inside its period.
    ([0.003], [0.2], [0.002]),  # Inference overrun: no extra post-send wait.
    ([0.04, 0.002, 0.001], [0.005, 0.15], [0.003]),  # Camera overrun and changing chunk latency.
    ([0.001], [0.2, 0.005], [0.05, 0.002]),  # Send overrun, then ordinary row budget.
])
def test_exact_native_base_strategy_cadence_and_observations(
        monkeypatch, observation_s, inference_s, send_s):
    from threading import Event

    from lerobot.rollout.strategies import base as native_base

    native, actual = (TimingWorld(observation_s, inference_s, send_s) for _ in range(2))
    policy, post = native_policy(), postprocessor()
    raw = torch.linspace(-0.8, 0.8, 30 * 14).reshape(1, 30, 14)

    def native_prediction(observation):
        native.infer(observation)
        return raw.clone()

    policy.predict_action_chunk = native_prediction
    native_stop = Event()

    def native_send_next_action(observation, *_args):
        tensor = post({TransitionKey.ACTION: policy.select_action(observation)})[TransitionKey.ACTION]
        target = tensor.squeeze(0).tolist()
        native.send(target)
        if len(native.sent) == 60:
            native_stop.set()
        return target

    # Execute the actual pinned BaseStrategy.run method, not a copied timing
    # formula. Only its clock, inert I/O and policy/hardware context are fake.
    monkeypatch.setattr(native_base, "time", SimpleNamespace(perf_counter=native.clock))
    monkeypatch.setattr(native_base, "precise_sleep", native.advance)
    monkeypatch.setattr(native_base, "send_next_action", native_send_next_action)
    strategy = native_base.BaseStrategy(SimpleNamespace())
    strategy._engine = SimpleNamespace(resume=lambda: None)
    strategy._interpolator = SimpleNamespace(get_control_interval=lambda fps: 1 / fps)
    strategy._process_observation_and_notify = lambda _processors, observation: observation
    strategy._handle_warmup = lambda *_: False
    strategy._log_telemetry = lambda *_: None
    context = SimpleNamespace(
        runtime=SimpleNamespace(cfg=SimpleNamespace(fps=30, duration=90, use_torch_compile=False),
                                shutdown_event=native_stop),
        hardware=SimpleNamespace(robot_wrapper=SimpleNamespace(get_observation=native.observe)),
        processors=None)
    strategy.run(context)

    processed_chunk = post({TransitionKey.ACTION: raw})[TransitionKey.ACTION].squeeze(0).tolist()

    def predict(observation, _timeout):
        actual.infer(observation)
        return processed_chunk

    def send(target, check):
        check()
        actual.send([target[name] for name in YAM_NAMES])
        return target

    engine = Pi05ReferenceExecutor(predict=predict, observe=actual.observe, send=send,
                                  validate_target=lambda _: None, stop=Event(),
                                  clock=actual.clock, wait=actual.advance)
    result = engine.run(duration_s=90, max_chunks=2)
    assert native.predicted == actual.predicted == [0, 30]
    assert len(actual.observed) == len(native.observed) == 60
    assert result["observations"] == 60
    np.testing.assert_allclose(actual.observed, native.observed, rtol=0, atol=1e-10)
    np.testing.assert_allclose([at for at, _ in actual.sent], [at for at, _ in native.sent], rtol=0, atol=1e-10)
    np.testing.assert_array_equal([row for _, row in actual.sent], [row for _, row in native.sent])
    assert actual.now == pytest.approx(native.now, abs=1e-10)
    if inference_s[0] > 1 / 30 and observation_s[0] < 1 / 30 and send_s[0] < 1 / 30:
        assert actual.sent[1][0] - actual.sent[0][0] < 1 / 30
