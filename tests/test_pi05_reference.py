"""Pure fake I/O coverage for the independent native π0.5 contract."""

from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from yamkit.inference.mapping import MOLMO_NAMES, YAM_NAMES
from yamkit.inference.profiles import get_profile
from yamkit.pi05.contract import PROFILE, validate_checkpoint_config
from yamkit.pi05.executor import Pi05ExecutionFault, Pi05ReferenceExecutor, finite_rows
from yamkit.pi05.runtime import pinned_tokenizer_snapshot, restore_native_weights


def checkpoint_config():
    return {
        "type": "pi05", "n_obs_steps": 1, "chunk_size": 30, "n_action_steps": 30,
        "max_state_dim": 32, "max_action_dim": 32, "num_inference_steps": 10,
        "use_relative_actions": False, "rtc_config": None, "dtype": "bfloat16",
        "empty_cameras": 0, "image_resolution": [224, 224], "tokenizer_max_length": 200,
        "action_feature_names": list(MOLMO_NAMES),
        "normalization_mapping": {"VISUAL": "IDENTITY", "STATE": "QUANTILES", "ACTION": "QUANTILES"},
        "input_features": {**{f"observation.images.{name}": {"type": "VISUAL", "shape": [3, 360, 640]}
                              for name in ("top", "left", "right")},
                           "observation.state": {"type": "STATE", "shape": [14]}},
        "output_features": {"action": {"type": "ACTION", "shape": [14]}},
    }


def test_profile_is_additive_and_preserves_base_checkpoint():
    assert get_profile("pi05").repo_id == "lerobot/pi05_base"
    assert get_profile("molmoact2").revision == "fdade02d1f1c1dd819114b0478f735072fb6b212"
    assert PROFILE.repo_id == "Jiafei1224/molmoact2-yam-pi05"
    assert PROFILE.state_names == PROFILE.action_names == YAM_NAMES
    assert PROFILE.native_image_keys == ("top", "left", "right")
    validate_checkpoint_config(checkpoint_config())


@pytest.mark.parametrize("key,value", [
    ("chunk_size", 50), ("n_action_steps", 15), ("use_relative_actions", True),
    ("rtc_config", {}), ("dtype", "float32"), ("num_inference_steps", 5),
    ("action_feature_names", list(reversed(MOLMO_NAMES))), ("n_obs_steps", True),
    ("normalization_mapping", {"STATE": "MEAN_STD"}), ("max_action_dim", 14),
])
def test_changed_checkpoint_semantics_are_rejected(key, value):
    config = checkpoint_config()
    config[key] = value
    with pytest.raises(ValueError, match="contract"):
        validate_checkpoint_config(config)


def test_camera_dictionary_order_is_contract_not_sorting():
    config = checkpoint_config()
    config["input_features"] = dict(reversed(list(config["input_features"].items())))
    with pytest.raises(ValueError, match="camera order"):
        validate_checkpoint_config(config)


class NativeWeights:
    config = object()

    def _fix_pytorch_state_dict_keys(self, values, config):
        assert config is self.config
        return values

    def load_state_dict(self, values, strict):
        assert strict is True
        self.loaded = values
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])


def test_strict_native_restore_uses_same_key_mapping():
    policy = NativeWeights()
    restore_native_weights(policy, {"a": 1, "model.b": 2})
    assert policy.loaded == {"model.a": 1, "model.b": 2}


def test_native_loader_never_returns_random_weights_after_failure():
    class BrokenWeights(NativeWeights):
        def load_state_dict(self, values, strict):
            raise RuntimeError("missing weights")

    with pytest.raises(RuntimeError, match="missing weights"):
        restore_native_weights(BrokenWeights(), {"a": 1})


def test_native_loader_rejects_missing_returned_keys():
    class PartialWeights(NativeWeights):
        def load_state_dict(self, values, strict):
            return SimpleNamespace(missing_keys=["missing"], unexpected_keys=[])

    with pytest.raises(ValueError, match="completely"):
        restore_native_weights(PartialWeights(), {"a": 1})


def test_native_loader_rejects_ambiguous_prefix_collision():
    with pytest.raises(ValueError, match="duplicate"):
        restore_native_weights(NativeWeights(), {"a": 1, "model.a": 2})


def test_gated_tokenizer_is_actionable_and_never_substituted():
    import httpx
    from huggingface_hub.errors import GatedRepoError

    attempted = []

    def download(repo, **kwargs):
        attempted.append((repo, kwargs["revision"]))
        raise GatedRepoError("private diagnostic must not escape",
                             response=httpx.Response(403, request=httpx.Request("GET", "https://huggingface.co/")))

    with pytest.raises(ValueError, match="Accept its publisher terms") as error:
        pinned_tokenizer_snapshot(download)
    assert "private diagnostic" not in str(error.value)
    assert attempted == [("google/paligemma-3b-pt-224", "35e4f46485b4d07967e7e9935bc3786aad50687c")]


class Clock:
    now = 100.0

    def __call__(self):
        return self.now

    def wait(self, delay):
        self.now += max(delay, 1e-12)


def rows():
    values = np.repeat(np.linspace(0.0, 0.2, 30)[:, None], 14, axis=1)
    values[:, [6, 13]] = 0.5
    return values


def make_executor(*, predictor=None, sender=None, validate=None, stop=None):
    clock, events, sent, observed, stop = Clock(), [], [], [], stop or Event()

    def predict(obs, timeout):
        assert timeout <= 2.0
        clock.wait(0.1)
        return predictor() if predictor else rows()

    def send(target, check):
        check()
        sent.append(target)
        return sender(target) if sender else target

    def observe():
        observed.append(len(sent))
        return {"state": np.zeros(14)}

    engine = Pi05ReferenceExecutor(predict=predict, observe=observe, send=send,
                                  validate_target=validate or (lambda _: None), stop=stop,
                                  clock=clock, wait=clock.wait,
                                  event=lambda name, **record: events.append((name, record)))
    return engine, clock, events, sent, observed


def test_native_full_fifo_rows_are_unchanged_without_molmo_interpolation():
    engine, _, events, sent, observed = make_executor()
    result = engine.run(duration_s=10, max_chunks=3)
    assert observed == [0, 30, 60]
    expected = np.tile(rows(), (3, 1))
    np.testing.assert_array_equal(np.array([[t[n] for n in YAM_NAMES] for t in sent]), expected)
    assert result["predicted_rows"] == result["completed_rows"] == 90
    assert result["completed_chunks"] == 3
    assert result["interpolation_points"] == result["dropped_rows"] == result["modified_commands"] == 0
    assert result["coherence_violations"] == result["reordered_rows"] == result["faults"] == 0
    assert result["execution_rate_hz"] == pytest.approx(30)
    assert len([e for e in events if e[0] == "dispatch"]) == 90


def test_stop_during_rpc_never_dispatches_late_chunk():
    stop = Event()

    def predict():
        stop.set()
        return rows()

    engine, _, _, sent, _ = make_executor(predictor=predict, stop=stop)
    result = engine.run(duration_s=5)
    assert sent == []
    assert result["completed_rows"] == 0
    assert result["predicted_rows"] == result["dropped_rows"] == 30
    assert result["stop_requested"] is True


def test_stop_midchunk_accounts_for_every_unsent_native_row():
    stop, count = Event(), []

    def send(target):
        count.append(1)
        if len(count) == 7:
            stop.set()
        return target

    engine, _, _, _, _ = make_executor(sender=send, stop=stop)
    result = engine.run(duration_s=5)
    assert result["completed_rows"] == 7
    assert result["dropped_rows"] == 23
    assert result["completed_chunks"] == 0
    assert result["prefix_dropped_rows"] == 0


@pytest.mark.parametrize("bad", [np.zeros((29, 14)), np.zeros((30, 13)),
                                  np.full((30, 14), np.nan), np.full((30, 14), np.inf),
                                  np.full((30, 14), 1.01), np.full((30, 14), -0.01),
                                  np.full((30, 14), True), [["0"] * 14] * 30])
def test_bad_chunks_never_dispatch(bad):
    with pytest.raises(Pi05ExecutionFault):
        finite_rows(bad)


def test_whole_chunk_bounds_are_validated_before_first_row():
    def validate(target):
        if target[YAM_NAMES[0]] > 0.15:
            raise Pi05ExecutionFault("bounds")

    engine, _, _, sent, _ = make_executor(validate=validate)
    with pytest.raises(Pi05ExecutionFault, match="bounds"):
        engine.run(duration_s=5)
    assert sent == []
    assert engine.metrics()["dropped_rows"] == 30


def test_command_modification_is_a_fault_not_a_policy_hack():
    def send(target):
        return {**target, YAM_NAMES[0]: target[YAM_NAMES[0]] + 0.001}

    engine, _, _, sent, _ = make_executor(sender=send)
    with pytest.raises(Pi05ExecutionFault, match="differs"):
        engine.run(duration_s=5)
    assert len(sent) == 1
    assert engine.metrics()["modified_commands"] == engine.metrics()["coherence_violations"] == 1


def test_used_engine_cannot_retry():
    engine, *_ = make_executor()
    engine.run(duration_s=5, max_chunks=1)
    with pytest.raises(Pi05ExecutionFault, match="cannot resume"):
        engine.run(duration_s=5, max_chunks=1)
