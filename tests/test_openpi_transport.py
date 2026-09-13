"""Actual loopback transport with explicit fake native model, including Stop races."""

import threading
import time

import numpy as np
import pytest

from tests import test_openpi_service as shared
from yamkit.inference.client import InvalidatedRequest, RemoteFault
from yamkit.openpi import service, transport

fake_native, statistics = shared.fake_native, shared.statistics
observation, serving = shared.observation, shared.serving


def client(runtime, port, **kwargs):
    return transport.OpenPiTransport(endpoint_url=f"http://127.0.0.1:{port}", token="a" * 32,
                                     statistics_sha256=runtime.ready()["statistics_sha256"], **kwargs)


def test_authenticated_roundtrip_warm_reuse_and_raw_shape(fake_native):
    runtime, calls, raw = fake_native
    with serving(runtime) as port:
        remote = client(runtime, port)
        try:
            ready = remote.ready()
            assert ready["instance_id"] == runtime.instance_id
            assert ready["session_expires_at"] == runtime.expires_at
            warmed = remote.warm(observation())
            first = remote.predict_chunk(observation())
            second = remote.predict_chunk(observation())
            np.testing.assert_array_equal(first["raw_normalized_chunk"], raw)
            assert len(calls) == 3
            assert warmed["instance_id"] == first["instance_id"] == second["instance_id"]
            assert warmed["sequence_id"] < first["sequence_id"] < second["sequence_id"]
            assert first["state"] == observation()["state"]
            assert remote.last_timing["request_retry_count"] == 0
            assert ready["openpi_service_build_id"] == service.build_id()
        finally:
            remote.close()


def test_predict_cannot_skip_identity_or_actual_task_warmup(fake_native):
    runtime, calls, _ = fake_native
    with serving(runtime) as port:
        remote = client(runtime, port)
        try:
            with pytest.raises(ValueError, match="identity"):
                remote.predict_chunk(observation())
            remote.ready()
            with pytest.raises(RemoteFault, match="HTTP 400"):
                remote.predict_chunk(observation())
            assert not calls
            remote.warm(observation())
            remote.predict_chunk(observation())
        finally:
            remote.close()


@pytest.mark.parametrize("endpoint", ["http://localhost:8767", "http://100.85.93.94:8767", "http://127.0.0.1:0",
                                     "http://127.0.0.1:8767/path", "https://127.0.0.1:8767",
                                     "http://user:secret@127.0.0.1:8767", "http://127.0.0.1:65536"])
def test_only_bare_loopback_ssh_origin(endpoint):
    with pytest.raises(ValueError, match="loopback") as exc:
        transport.OpenPiTransport(endpoint_url=endpoint, token="a" * 32, statistics_sha256="b" * 64)
    assert "secret" not in str(exc.value)


@pytest.mark.parametrize("field,value", [("policy", "pi05-yam"), ("runtime_revision", "a" * 40),
                                       ("statistics_sha256", "a" * 64), ("openpi_service_build_id", "a" * 64),
                                       ("native_output_shape", [30, 14]), ("ready", 1),
                                       ("num_inference_steps", 11), ("model_weights_modified", True),
                                       ("session_expires_at", float("nan")), ("instance_id", "changed")])
def test_every_native_identity_is_bound(fake_native, field, value):
    runtime, _, _ = fake_native
    metadata = runtime.ready()
    metadata[field] = value
    with pytest.raises(ValueError):
        transport.validate_readiness(metadata, statistics_sha256=runtime.ready()["statistics_sha256"])


def test_stop_returns_immediately_and_late_reply_is_never_accepted(fake_native):
    runtime, _, raw = fake_native
    entered, release = threading.Event(), threading.Event()
    outcome = []
    with serving(runtime) as port:
        remote = client(runtime, port)
        remote.ready()
        remote.warm(observation())

        def block(_):
            entered.set()
            assert release.wait(2)
            return {"actions": raw}

        runtime.policy.infer = block

        def invoke():
            try:
                outcome.append(remote.predict_chunk(observation()))
            except RemoteFault as exc:
                outcome.append(exc)

        worker = threading.Thread(target=invoke)
        worker.start()
        assert entered.wait(1)
        # The loopback server can consume the body before the client's sendall
        # thread resumes. Wait for the explicit client-side send completion too.
        assert remote.request_sent.wait(1)
        assert remote.last_timing["request_sent_monotonic_s"] <= time.monotonic()
        started = time.monotonic()
        remote.cancel()
        worker.join(0.5)
        assert not worker.is_alive()
        assert time.monotonic() - started < 0.5
        assert len(outcome) == 1 and isinstance(outcome[0], InvalidatedRequest)
        with pytest.raises(RemoteFault, match="still in flight"):
            remote.predict_chunk(observation())
        release.set()
        retire = time.monotonic() + 2
        while remote._busy.locked() and time.monotonic() < retire:
            threading.Event().wait(0.005)
        assert not remote._busy.locked()
        assert len(outcome) == 1
        remote.close()


def test_request_deadline_no_retry_and_session_stop_guard(fake_native):
    runtime, _, raw = fake_native
    with serving(runtime) as port:
        stop = threading.Event()
        remote = client(runtime, port, shutdown_event=stop)
        remote.ready()
        remote.warm(observation())

        def slow(_):
            time.sleep(0.1)
            return {"actions": raw}

        runtime.policy.infer = slow
        started = time.monotonic()
        with pytest.raises(RemoteFault, match="deadline|failed"):
            remote.predict_chunk(observation(), timeout_s=0.03)
        assert time.monotonic() - started < 0.3
        assert remote.last_timing["request_retry_count"] == 0
        stop.set()
        with pytest.raises(InvalidatedRequest):
            remote.ensure_session_active()
        remote.close()


@pytest.mark.parametrize("field,value", [("state", [0.0] * 14), ("instance_id", "changed"),
                                       ("sequence_id", 999), ("raw_normalized_chunk", [[0.0] * 14] * 30),
                                       ("raw_dtype", "object"), ("warm_signature", "wrong"),
                                       ("statistics_file_sha256", "f" * 64)])
def test_response_anchor_shape_and_identity_rejected(fake_native, monkeypatch, field, value):
    runtime, _, _ = fake_native
    with serving(runtime) as port:
        remote = client(runtime, port)
        remote.ready()
        remote.warm(observation())
        invoke = remote._invoke

        def altered(route, payload, timeout):
            result = invoke(route, payload, timeout)
            result[field] = value
            return result

        monkeypatch.setattr(remote, "_invoke", altered)
        with pytest.raises(ValueError, match="validation"):
            remote.predict_chunk(observation())
        remote.close()


def test_overlay_git_sha_is_provenance_not_implicit_false_mismatch(fake_native):
    runtime, _, _ = fake_native
    metadata = runtime.ready()
    transport.validate_readiness(metadata, statistics_sha256=metadata["statistics_sha256"])
    with pytest.raises(ValueError, match="source"):
        transport.validate_readiness(metadata, statistics_sha256=metadata["statistics_sha256"],
                                     expected_source_sha="d" * 40)


def test_transport_repr_cannot_leak_bearer():
    secret = "x" * 32
    remote = transport.OpenPiTransport(endpoint_url="http://127.0.0.1:8767", token=secret,
                                       statistics_sha256="a" * 64)
    assert secret not in repr(remote)
    remote.close()


def test_stop_between_wire_and_validation_rejects_result(fake_native, monkeypatch):
    runtime, _, _ = fake_native
    with serving(runtime) as port:
        remote = client(runtime, port)
        remote.ready()
        remote.warm(observation())
        validate = transport.validate_normalized_chunk

        def stop_after_wire(value):
            remote.cancel()
            return validate(value)

        monkeypatch.setattr(transport, "validate_normalized_chunk", stop_after_wire)
        with pytest.raises(InvalidatedRequest, match="validation"):
            remote.predict_chunk(observation())
        remote.close()


def test_monotonic_session_expiry_and_identity_change_close_transport(fake_native):
    runtime, _, _ = fake_native
    with serving(runtime) as port:
        remote = client(runtime, port)
        remote.ready()
        remote._deadline = time.monotonic() - 1
        with pytest.raises(RemoteFault, match="expired"):
            remote.ensure_session_active()
        remote.close()
        remote = client(runtime, port)
        remote.ready()
        runtime._identity["statistics_file_sha256"] = "f" * 64
        with pytest.raises(ValueError, match="changed"):
            remote.ready()
        assert remote._closed
