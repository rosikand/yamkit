"""Explicitly fake native policy and loopback HTTP; no GPU or hardware opens."""

import http.client
import json
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from yamkit.openpi import service
from yamkit.openpi.contract import IMAGE_KEYS
from yamkit.openpi.yam_candidate import CandidateQuantiles, CandidateStatistics, encode_state


@pytest.fixture
def statistics():
    quantiles = CandidateQuantiles(np.full(14, -1), np.full(14, 1), "explicitly fake unit data", "a" * 64)
    return CandidateStatistics(quantiles, quantiles)


@pytest.fixture
def fake_native(statistics):
    calls = []
    raw = np.arange(1600, dtype=np.float32).reshape(50, 32) / 100 - 2

    def infer(observation):
        calls.append(observation)
        return {"actions": raw.copy()}

    runtime = service.NativeRuntime(SimpleNamespace(infer=infer), statistics, expires_at=time.time() + 120,
                                    source_sha="b" * 40, statistics_file_sha256="c" * 64, image_hw=(2, 3))
    return runtime, calls, raw


def observation():
    return service.observation_request({key: np.full((2, 3, 3), index * 53, dtype=np.uint8)
                                        for index, key in enumerate(IMAGE_KEYS)},
                                       [0.2] * 6 + [0.8] + [-0.3] * 6 + [0.1], "put cube into container")


def request(runtime, sequence=1):
    return {**observation(), "wire_version": service.WIRE_VERSION, "client_id": "test-client",
            "sequence_id": sequence, "timeout_s": 2.0, "instance_id": runtime.instance_id,
            "statistics_sha256": runtime.ready()["statistics_sha256"],
            "openpi_service_build_id": service.build_id()}


@contextmanager
def serving(runtime, token="a" * 32):
    server = service.make_server(runtime, token=token, port=0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_native_14d_input_before_padding_and_untouched_output(fake_native, statistics):
    runtime, calls, raw = fake_native
    payload = request(runtime)
    result = runtime.predict(payload, warm=True)
    assert len(calls) == 1
    assert calls[0]["state"].shape == (14,)
    np.testing.assert_array_equal(calls[0]["state"], encode_state(payload["state"], statistics))
    assert set(calls[0]) == {"image", "image_mask", "state", "prompt"}
    assert calls[0]["prompt"] == payload["task"]
    for index, key in enumerate(IMAGE_KEYS):
        np.testing.assert_array_equal(calls[0]["image"][key], np.full((2, 3, 3), index * 53, dtype=np.uint8))
        assert calls[0]["image_mask"][key]
    np.testing.assert_array_equal(result["raw_normalized_chunk"], raw)
    assert result["raw_normalized_chunk"][-1][-1] > 1
    assert result["state"] == payload["state"]
    assert result["native_inference_modified"] is False
    assert result["model_weights_modified"] is False
    assert result["hardware_tested"] is False


def test_warm_is_actual_task_specific_and_replay_rejected(fake_native):
    runtime, calls, _ = fake_native
    with pytest.raises(ValueError, match="warmed"):
        runtime.predict(request(runtime))
    runtime.predict(request(runtime, 2), warm=True)
    runtime.predict(request(runtime, 3))
    with pytest.raises(ValueError, match="stale"):
        runtime.predict(request(runtime, 3))
    payload = request(runtime, 4)
    payload["task"] = "different task"
    with pytest.raises(ValueError, match="warmed"):
        runtime.predict(payload)
    assert len(calls) == 2


@pytest.mark.parametrize("key,value", [("instance_id", "wrong"), ("statistics_sha256", "a" * 64),
                                         ("openpi_service_build_id", "a" * 64), ("sequence_id", True),
                                         ("sequence_id", -1), ("wire_version", True),
                                         ("client_id", "contains spaces"), ("timeout_s", 2.01),
                                         ("timeout_s", float("nan")), ("task", ""), ("task", 123)])
def test_request_identity_schema_and_bounds_fail_before_native(fake_native, key, value):
    runtime, calls, _ = fake_native
    payload = request(runtime)
    payload[key] = value
    with pytest.raises(ValueError):
        runtime.predict(payload)
    assert not calls


@pytest.mark.parametrize("state", [[0] * 13, [0] * 15, [float("inf")] * 14,
                                  [0] * 6 + [1.01] + [0] * 7, [0] * 13 + [-0.01]])
def test_invalid_measured_state_not_repaired(fake_native, state):
    runtime, calls, _ = fake_native
    payload = request(runtime)
    payload["state"] = state
    with pytest.raises(ValueError):
        runtime.predict(payload, warm=True)
    assert not calls


@pytest.mark.parametrize("mutation", ["shape", "encoding", "data", "extra", "role"])
def test_wire_rgb_is_strict(fake_native, mutation):
    runtime, calls, _ = fake_native
    payload = request(runtime)
    image = payload["images"][IMAGE_KEYS[0]]
    if mutation == "shape":
        image["shape"] = [3, 2, 3]
    elif mutation == "encoding":
        image["encoding"] = "jpeg"
    elif mutation == "data":
        image["data"] = "!" * len(image["data"])
    elif mutation == "extra":
        image["crop"] = "center"
    else:
        payload["images"]["wrong"] = payload["images"].pop(IMAGE_KEYS[0])
    with pytest.raises(ValueError):
        runtime.predict(payload, warm=True)
    assert not calls


@pytest.mark.parametrize("raw", [np.zeros((30, 14), dtype=np.float32), np.zeros((50, 32), dtype=np.int64),
                                np.full((50, 32), np.nan), np.full((50, 32), np.inf)])
def test_native_invalid_output_fails_without_mutation(fake_native, raw):
    runtime, _, _ = fake_native
    runtime.policy.infer = lambda _: {"actions": raw}
    with pytest.raises(ValueError):
        runtime.predict(request(runtime), warm=True)
    assert not runtime.ready()["warm_signatures"]
    assert not runtime._lock.locked()


def test_server_serializes_model_and_rejects_expired_results(fake_native):
    runtime, calls, _ = fake_native
    runtime._lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="in flight"):
            runtime.predict(request(runtime), warm=True)
    finally:
        runtime._lock.release()
    assert not calls
    original = runtime.policy.infer

    def expire(observation):
        result = original(observation)
        runtime._deadline = time.monotonic() - 1
        return result

    runtime.policy.infer = expire
    with pytest.raises(TimeoutError, match="expired"):
        runtime.predict(request(runtime), warm=True)
    assert not runtime._warm_signatures


def test_readiness_cannot_mutate_identity(fake_native):
    runtime, _, _ = fake_native
    result = runtime.ready()
    result["model_config"]["action_horizon"] = 30
    assert runtime.ready()["model_config"]["action_horizon"] == 50


@pytest.mark.parametrize("body", [b'{"a":1,"a":2}', b'{"x":NaN}', b'[]', b'', b'{invalid'])
def test_json_parser_rejects_ambiguous_wire(body):
    with pytest.raises(ValueError):
        service.decode_json(body)


def test_statistics_metadata_and_file_hash_bound(tmp_path, statistics):
    path = tmp_path / "statistics.json"
    path.write_text(json.dumps(statistics.metadata()))
    loaded, digest = service.load_statistics(tmp_path, path)
    assert loaded.metadata() == statistics.metadata()
    assert len(digest) == 64
    value = statistics.metadata()
    value["state"]["q01"][0] = -2
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="identity"):
        service.load_statistics(tmp_path, path)


def test_private_token_metadata_checked_without_exposing_value(tmp_path):
    path = tmp_path / "private.token"
    path.write_text("z" * 32)
    path.chmod(0o600)
    assert service.read_token_file(tmp_path, path) == "z" * 32
    path.chmod(0o644)
    with pytest.raises(ValueError, match="private") as exc:
        service.read_token_file(tmp_path, path)
    assert "z" * 32 not in str(exc.value)


def test_http_authentication_and_errors_do_not_echo_values(fake_native):
    runtime, _, _ = fake_native
    with serving(runtime) as port:
        for credential, path, status in [("wrong", "/ready", 401), ("a" * 32, "/unknown", 404),
                                        ("a" * 32, "/ready", 200)]:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            connection.request("GET", path, headers={"Authorization": "Bearer " + credential})
            response = connection.getresponse()
            body = response.read()
            assert response.status == status
            assert credential.encode() not in body
            connection.close()


def test_loader_keeps_official_model_and_changes_only_input_adapter(tmp_path, statistics, monkeypatch):
    path = tmp_path / "statistics.json"
    path.write_text(json.dumps(statistics.metadata()))
    policy = object()
    seen = []

    def load(root):
        seen.append(root)
        return SimpleNamespace(policy=policy, provenance={"packages": {"openpi": "test"},
                                                          "state_source": "old diagnostic zeros"})

    monkeypatch.setattr(service.OfficialPi05Diagnostic, "load", load)
    config = service.ServiceConfig(root=str(tmp_path), statistics=str(path), token_file="private.token",
                                   source_sha="e" * 40)
    runtime = service.NativeRuntime.load(config, expires_at=time.time() + 120)
    assert runtime.policy is policy
    assert seen == [tmp_path]
    assert runtime.ready()["source_sha"] == "e" * 40
    assert "actual supplied measured" in runtime.provenance["state_source"]


@pytest.mark.parametrize("field,value", [("session_seconds", float("nan")), ("session_seconds", 86401),
                                        ("image_height", 721), ("port", 0), ("source_sha", "not-a-sha"),
                                        ("statistics", "../escape.json")])
def test_configuration_is_bounded_and_repo_local(tmp_path, field, value):
    kwargs = {"root": str(tmp_path), "statistics": "stats.json", "token_file": "private.token", field: value}
    with pytest.raises(ValueError):
        service.ServiceConfig(**kwargs).validate()
