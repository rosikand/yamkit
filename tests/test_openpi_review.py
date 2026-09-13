"""Independent final-review regressions; explicit fake devices/model only."""

import signal

import numpy as np
import pytest

from tests import test_openpi_rollout as lifecycle_shared
from tests import test_openpi_service as service_shared
from tests import test_openpi_transport as transport_shared
from yamkit.inference.client import RemoteFault
from yamkit.openpi import artifacts, rollout

lifecycle = lifecycle_shared.lifecycle
fake_native, statistics = service_shared.fake_native, service_shared.statistics


@pytest.mark.parametrize("telemetry", ["capture_end", "execution_metrics", "release_phase"])
@pytest.mark.parametrize("control_fault", [False, True])
def test_final_telemetry_failure_cannot_bypass_release_or_signal_restoration(
    lifecycle, monkeypatch, telemetry, control_fault
):
    """Telemetry can fail precisely when RAM pressure makes release essential."""
    if control_fault:
        lifecycle.options["engine_failure"] = ValueError("explicit fake control failure")
    if telemetry == "capture_end":
        original = artifacts.OpenPiCapture.end

        def fail_end(self):
            if "run" in lifecycle.calls:
                raise MemoryError("explicit fake capture telemetry exhaustion")
            original(self)

        monkeypatch.setattr(artifacts.OpenPiCapture, "end", fail_end)
    elif telemetry == "execution_metrics":
        def fail_metrics(_self):
            raise MemoryError("explicit fake metrics telemetry exhaustion")

        monkeypatch.setattr(rollout.OpenPiYamExecutor, "metrics", fail_metrics)
    else:
        original = artifacts.rollout_phase

        def fail_release_phase(phase):
            if phase == "releasing":
                raise MemoryError("explicit fake phase display exhaustion")
            original(phase)

        monkeypatch.setattr(artifacts, "rollout_phase", fail_release_phase)
    before = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
    outcome = None
    try:
        try:
            outcome = rollout.run_rollout(lifecycle.transport, **lifecycle.args)
        except BaseException as exc:  # noqa: BLE001 — capture failure for safety assertions below
            outcome = exc
        after = {number: signal.getsignal(number) for number in before}
    finally:
        # A failing version of the implementation must not contaminate pytest.
        for number, handler in before.items():
            signal.signal(number, handler)
    assert "disconnect_no_home" in lifecycle.calls
    assert "fake_guard_exit" in lifecycle.calls
    assert lifecycle.sdk.closed
    assert after == before
    assert isinstance(outcome, dict), "Telemetry failure must produce a sanitized failed lifecycle report"
    assert outcome["released"] is True
    assert outcome["exit_status"] != 0
    if control_fault:
        assert outcome["failure_type"] == "ValueError", "Preserve the original control failure"


@pytest.mark.parametrize("raw", [np.full((50, 32), np.nan), np.full((50, 32), np.inf),
                                  np.zeros((49, 32), dtype=np.float32)])
def test_server_rejected_native_anomaly_never_reaches_client_raw_capture(fake_native, raw):
    """Raw-first client retention does not imply retention before HTTP validation."""
    runtime, _, _ = fake_native
    runtime.policy.infer = lambda _observation: {"actions": raw}
    captured = []
    with service_shared.serving(runtime) as port:
        remote = transport_shared.client(runtime, port)
        try:
            remote.ready()
            with pytest.raises(RemoteFault, match="HTTP 400") as failure:
                captured.append(remote.warm(service_shared.observation()))
            assert not captured
            assert str(failure.value) == "Official OpenPI request failed (HTTP 400)"
            assert remote.last_timing["request_retry_count"] == 0
            assert not runtime.ready()["warm_signatures"]
        finally:
            remote.close()
