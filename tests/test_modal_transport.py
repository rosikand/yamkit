"""Persistent SDK handles must retain bounded requests and local Stop semantics."""

import sys
import threading
from types import SimpleNamespace

import pytest

from yamkit.inference.client import InvalidatedRequest, ModalTransport, RemoteFault


def install_sdk(monkeypatch, get):
    lookups, cancellations = [], []

    def lookup(*args):
        lookups.append(args)
        call = SimpleNamespace(get=get, cancel=lambda: cancellations.append(True))
        method = SimpleNamespace(spawn=lambda *args: call, remote=lambda *args: get(timeout=None))
        return lambda: SimpleNamespace(ready=method, predict_chunk=method)

    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(Cls=SimpleNamespace(from_name=lookup)))
    return lookups, cancellations


@pytest.mark.parametrize("call_mode", ["remote", "spawn"])
def test_ready_and_predictions_reuse_one_service_handle(monkeypatch, call_mode):
    lookups, _ = install_sdk(monkeypatch, lambda **kwargs: {"ok": True})
    transport = ModalTransport("test-app", "molmoact2", call_mode=call_mode)
    assert transport.ready(1) == {"ok": True}
    assert transport.last_timing["handle_reused"] is False
    for _ in range(3):
        assert transport.predict_chunk({}, 1) == {"ok": True}
        assert transport.last_timing["handle_reused"] is True
    assert lookups == [("test-app", "PolicyService")]


@pytest.mark.parametrize("call_mode", ["remote", "spawn"])
def test_stop_rejects_late_cached_handle_response_and_overlapping_request(monkeypatch, call_mode):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    waiting = [False]

    def get(**kwargs):
        if waiting[0]:
            entered.set()
            assert release.wait(2)
        return {"ok": True}

    lookups, cancelled = install_sdk(monkeypatch, get)
    stop = threading.Event()
    transport = ModalTransport("test-app", "molmoact2", shutdown_event=stop, call_mode=call_mode)
    transport.ready(1)
    waiting[0] = True
    errors = []

    def predict():
        try:
            transport.predict_chunk({}, 1)
        except RemoteFault as exc:
            errors.append(exc)
        finally:
            finished.set()

    worker = threading.Thread(target=predict)
    worker.start()
    try:
        assert entered.wait(1)
        with pytest.raises(RemoteFault, match="still in flight"):
            transport.predict_chunk({}, 1)
        stop.set()
        assert finished.wait(0.5), "Stop must return before the RPC completes"
        assert len(errors) == 1 and isinstance(errors[0], InvalidatedRequest)
    finally:
        release.set()
        worker.join(2)
    # The SDK worker releases its busy lock only after handling cancellation.
    assert transport._busy.acquire(timeout=1)
    transport._busy.release()
    assert bool(cancelled) is (call_mode == "spawn") and len(lookups) == 1
    with pytest.raises(InvalidatedRequest):
        transport.predict_chunk({}, 1)


@pytest.mark.parametrize("call_mode", ["remote", "spawn"])
def test_deadline_retains_busy_ownership_until_sdk_worker_retires(monkeypatch, call_mode):
    release = threading.Event()

    def get(**kwargs):
        assert release.wait(2)
        return {}

    _, cancelled = install_sdk(monkeypatch, get)
    transport = ModalTransport("test-app", "molmoact2", call_mode=call_mode)
    try:
        with pytest.raises(RemoteFault, match="deadline"):
            transport.predict_chunk({}, 0.05)
        with pytest.raises(RemoteFault, match="still in flight"):
            transport.predict_chunk({}, 1)
    finally:
        release.set()
    assert transport._busy.acquire(timeout=1)
    transport._busy.release()
    assert bool(cancelled) is (call_mode == "spawn")


def test_local_invalidation_rejects_uncancellable_remote_reply(monkeypatch):
    """Reset has no shutdown event, but still invalidates the outstanding call."""
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def get(**kwargs):
        entered.set()
        assert release.wait(2)
        return {"late": True}

    _, cancelled = install_sdk(monkeypatch, get)
    transport = ModalTransport("test-app", "molmoact2")
    errors = []

    def request():
        try:
            transport.predict_chunk({}, 1)
        except RemoteFault as exc:
            errors.append(exc)
        finally:
            finished.set()

    worker = threading.Thread(target=request)
    worker.start()
    try:
        assert entered.wait(1)
        transport.cancel()
        assert finished.wait(0.5)
        assert isinstance(errors[0], InvalidatedRequest)
        with pytest.raises(RemoteFault, match="still in flight"):
            transport.ready(1)
    finally:
        release.set()
        worker.join(2)
    assert transport._busy.acquire(timeout=1)
    transport._busy.release()
    assert not cancelled
