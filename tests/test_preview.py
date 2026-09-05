"""Publisher tests use synthetic arrays and loopback sockets; no camera is opened."""

import gc
import http.client
import json
import socket
import threading
import time
import weakref
from contextlib import contextmanager

import cv2
import numpy as np
import pytest

from yamkit import preview


def test_stale_image_stays_stale_when_viewer_reconnects():
    slot = preview._Slot("top", "rgb")
    slot.set_latest(preview.Encoded(1, 3, b"jpeg", 10.0, 10.1, (2, 2, 3)))
    slot.viewers = 1
    assert slot.status(12.0)["state"] == "stale"
    slot.viewers = 0
    assert slot.status(13.0)["state"] == "stale"
    assert slot.status(13.0)["age_s"] == 3.0


def wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition did not become true")
        time.sleep(0.005)


@contextmanager
def publisher(**kwargs):
    pub = preview.PreviewPublisher("test-session", "test-secret", {"top": "rgb"}, **kwargs).start()
    try:
        yield pub
    finally:
        pub.close()


def connect(pub, path="/cameras/top/stream", token="test-secret"):
    connection = http.client.HTTPConnection("127.0.0.1", pub.port, timeout=3)
    headers = {preview.TOKEN_HEADER: token} if token is not None else {}
    connection.request("GET", path, headers=headers)
    return connection, connection.getresponse()


def read_part(response):
    assert response.readline() == f"--{preview.MJPEG_BOUNDARY}\r\n".encode()
    headers = {}
    while (line := response.readline()) != b"\r\n":
        key, value = line.decode().split(":", 1)
        headers[key] = value.strip()
    image = response.read(int(headers["Content-Length"]))
    assert response.read(2) == b"\r\n"
    return headers, image


def manual_publisher(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(preview.time, "perf_counter", lambda: now[0])
    pub = preview.PreviewPublisher("s", "t", {"top": "rgb"})
    return pub, pub.slot("top"), now


def test_demand_and_rate_precede_copy_or_pixel_access(monkeypatch):
    class GuardedArray(np.ndarray):
        def __getattribute__(self, name):
            if name == "nbytes":
                raise AssertionError("pixels inspected before demand/rate checks")
            return super().__getattribute__(name)

    pub, slot, now = manual_publisher(monkeypatch)
    forbidden = np.zeros((16, 16, 3), dtype=np.uint8).view(GuardedArray)
    pub.offer("top", forbidden)
    assert slot.offered == slot.copied == slot.offer_errors == 0
    assert pub.acquire_viewer(slot)
    pub.offer("top", np.zeros((16, 16, 3), dtype=np.uint8))
    now[0] += 0.01
    pub.offer("top", forbidden)
    assert slot.accepted == slot.copied == slot.rate_skipped == 1
    assert slot.offer_errors == 0


def test_thirty_hz_observations_target_ten_hz_preview(monkeypatch):
    pub, slot, now = manual_publisher(monkeypatch)
    assert pub.acquire_viewer(slot)
    for index in range(30):
        now[0] = 10.0 + index / 30.0
        pub.offer("top", np.full((8, 8, 3), index, dtype=np.uint8))
        slot.take()
    assert slot.accepted == slot.copied == 10
    assert slot.rate_skipped == 20


@pytest.mark.parametrize("kind", ["owning", "view", "strided"])
def test_snapshot_survives_buffer_reuse_without_mutating_observation(monkeypatch, kind):
    pub, slot, _ = manual_publisher(monkeypatch)
    assert pub.acquire_viewer(slot)
    storage = np.full((24, 32, 3), 29, dtype=np.uint8)
    frame = storage if kind == "owning" else storage.view() if kind == "view" else storage[:, ::2]
    original = frame.copy()
    pub.offer("top", frame)
    np.testing.assert_array_equal(frame, original)
    pending = slot.take()
    assert not np.shares_memory(pending.frame, frame)
    assert pending.frame.flags.c_contiguous
    frame[:] = 190  # simulate producer recycling even its own OWNDATA array
    np.testing.assert_array_equal(pending.frame, original)
    assert slot.copied == 1


def test_snapshot_does_not_retain_original_camera_buffer(monkeypatch):
    pub, slot, _ = manual_publisher(monkeypatch)
    assert pub.acquire_viewer(slot)
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    reference = weakref.ref(frame)
    pub.offer("top", frame)
    del frame
    gc.collect()
    assert reference() is None
    assert slot._pending is not None


def test_pending_replaces_old_and_busy_slot_drops_before_copy(monkeypatch):
    pub, slot, now = manual_publisher(monkeypatch)
    assert pub.acquire_viewer(slot)
    for value in range(8):
        pub.offer("top", np.full((12, 12, 3), value, dtype=np.uint8))
        now[0] += 0.101
    assert slot.accepted == slot.copied == 8
    assert slot.dropped_pending == 7
    np.testing.assert_array_equal(slot._pending.frame, 7)
    with slot._lock:
        pub.offer("top", np.zeros((12, 12, 3), dtype=np.uint8))
    assert slot.dropped_busy == 1
    assert slot.copied == 8
    assert slot.take() is not None
    assert slot.take() is None


def test_offer_errors_are_contained_without_conversion_or_logging(monkeypatch):
    class NotAnImage:
        def __array__(self):
            pytest.fail("offer must not convert the observation")

    class BrokenImage(np.ndarray):
        @property
        def nbytes(self):
            raise RuntimeError("bad array metadata")

    pub, slot, _ = manual_publisher(monkeypatch)
    assert pub.acquire_viewer(slot)
    monkeypatch.setattr(preview.log, "warning", lambda *a, **k: pytest.fail("offer must not log"))
    pub.offer("top", NotAnImage())
    pub.offer("top", np.zeros((2, 2, 3), dtype=np.uint8).view(BrokenImage))
    pub.offer("not-configured", NotAnImage())
    assert slot.offer_errors == 2
    assert slot.copied == 0


def test_replay_and_acquisition_pause_keep_source_age(monkeypatch):
    pub, slot, now = manual_publisher(monkeypatch)
    assert pub.acquire_viewer(slot)
    assert slot.status(now[0])["state"] == "waiting"
    frame = np.zeros((12, 12, 3), dtype=np.uint8)
    pub.offer("top", frame, source_time=9.8)
    pending = slot.take()
    slot.set_latest(preview.Encoded(1, pending.source_seq, b"jpeg", pending.t_src, now[0], frame.shape))
    now[0] = 10.2
    pub.offer("top", frame)  # fallback when camera metadata lock was busy
    assert slot.accepted == 1
    assert slot.status(now[0])["age_s"] == 0.4
    now[0] = 12.0  # acquisition paused during dataset save/reset
    pub.offer("top", frame, source_time=9.8)
    status = slot.status(now[0])
    assert status["state"] == "stale"
    assert status["age_s"] == 2.2
    assert status["source_seq"] == 1
    assert slot.take() is None
    frame[:] = 50
    pub.offer("top", frame, source_time=11.9)  # producer reuses same array for a new acquisition
    pending = slot.take()
    assert pending.source_seq == 2
    assert pending.t_src == 11.9
    assert slot.accepted == 2


def test_timestamp_becoming_available_does_not_republish_a_replay(monkeypatch):
    pub, slot, now = manual_publisher(monkeypatch)
    assert pub.acquire_viewer(slot)
    frame = np.zeros((12, 12, 3), dtype=np.uint8)
    pub.offer("top", frame)  # camera's metadata lock was briefly busy
    assert slot.take().t_src == 10.0
    now[0] = 10.2
    pub.offer("top", frame, source_time=9.8)
    now[0] = 10.4
    pub.offer("top", frame, source_time=9.8)
    assert slot.source_seq == slot.accepted == 1
    assert slot.take() is None
    # A later acquisition into that same ndarray is still recognized as new.
    pub.offer("top", frame, source_time=10.3)
    assert slot.take().source_seq == 2


def test_status_source_sequence_and_age_describe_displayed_frame(monkeypatch):
    pub, slot, now = manual_publisher(monkeypatch)
    assert pub.acquire_viewer(slot)
    assert slot.status(now[0])["source_seq"] is None
    pub.offer("top", np.zeros((12, 12, 3), dtype=np.uint8), source_time=9.8)
    pending = slot.take()
    slot.set_latest(preview.Encoded(1, pending.source_seq, b"jpeg", pending.t_src, now[0], (12, 12, 3)))
    # New observations replace the pending slot while the worker is behind. Viewers still
    # see the first JPEG, whose acquisition identity and age must remain paired in status.
    for index in range(3):
        now[0] += 0.11
        pub.offer("top", np.full((12, 12, 3), index + 1, dtype=np.uint8), source_time=now[0] - 0.01)
    status = slot.status(now[0])
    assert status["seq"] == 1
    assert status["source_seq"] == 1
    assert status["age_s"] == 0.53
    assert status["observed_source_seq"] == 4
    assert status["dropped"] == 2


@pytest.mark.parametrize("mode,color", [("rgb", (255, 0, 0)), ("bgr", (0, 0, 255))])
def test_encoding_preserves_color_and_input(mode, color):
    pub = preview.PreviewPublisher("s", "t", {"top": mode})
    pub._cv2 = cv2
    frame = np.full((32, 32, 3), color, dtype=np.uint8)
    original = frame.copy()
    jpeg = pub._encode(frame, mode)
    decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded[:, :, 2].mean() > 245
    assert decoded[:, :, :2].mean() < 8
    np.testing.assert_array_equal(frame, original)


def test_multiple_viewers_share_one_encode(monkeypatch):
    with publisher() as pub:
        encoded_on = []
        original_encode = pub._encode

        def encode(frame, mode):
            encoded_on.append(threading.current_thread().name)
            return original_encode(frame, mode)

        monkeypatch.setattr(pub, "_encode", encode)
        clients = [connect(pub) for _ in range(3)]
        try:
            assert all(response.status == 200 for _, response in clients)
            wait_until(lambda: pub.viewers == 3)
            observation = np.full((32, 32, 3), (255, 0, 0), dtype=np.uint8)
            pub.offer("top", observation)
            parts = [read_part(response) for _, response in clients]
            assert len(encoded_on) == 1
            assert encoded_on == ["yamkit-preview-encoder"]
            assert all(part[1] == parts[0][1] for part in parts)
            assert all(part[0]["X-Yamkit-Source-Seq"] == "1" for part in parts)
            assert pub.status()["cameras"]["top"]["encoded"] == 1
            np.testing.assert_array_equal(observation[:, :, 0], 255)
        finally:
            for connection, response in clients:
                response.close()
                connection.close()
        wait_until(lambda: pub.viewers == 0)


def test_loopback_auth_unknown_camera_and_registration():
    with publisher(owner="camera-lease") as pub:
        assert pub._server.server_address[0] == "127.0.0.1"
        prefix, body = pub.registration_line().split(" ", 1)
        assert prefix == "@yamkit-preview/1"
        assert "test-secret" not in body
        registration = json.loads(body)
        assert registration["session"] == "test-session"
        assert registration["owner"] == "camera-lease"
        assert registration["port"] == pub.port
        for path, token, status in [
            ("/status", None, 401),
            ("/status", "wrong", 401),
            ("/cameras/unknown/stream", "test-secret", 404),
            ("/status", "test-secret", 200),
        ]:
            connection, response = connect(pub, path, token)
            assert response.status == status
            response.read()
            response.close()
            connection.close()
        assert pub.viewers == 0


def test_viewer_limit_is_atomic_and_disconnect_before_first_frame():
    with publisher(max_viewers=1) as pub:
        barrier = threading.Barrier(3)
        outcomes = []

        def open_viewer():
            barrier.wait()
            outcomes.append(connect(pub))

        threads = [threading.Thread(target=open_viewer) for _ in range(2)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(3)
            assert not thread.is_alive()
        try:
            assert sorted(response.status for _, response in outcomes) == [200, 503]
            assert pub.viewers == 1
        finally:
            for connection, response in outcomes:
                response.close()
                connection.close()
        wait_until(lambda: pub.viewers == 0)


def test_connection_limit_applies_before_header_read_and_cleanup(monkeypatch):
    monkeypatch.setattr(preview, "SEND_TIMEOUT_S", 0.2)
    with publisher(max_viewers=1) as pub:
        server = pub._server
        limit = pub.max_viewers + preview.CONTROL_CONNECTIONS
        sockets = [socket.create_connection(("127.0.0.1", pub.port), timeout=2) for _ in range(limit + 4)]
        try:
            wait_until(lambda: len(server._connections) > 0)
            assert len(server._connections) <= limit
            wait_until(lambda: len(server._connections) == 0)
        finally:
            for sock in sockets:
                sock.close()


def test_stalled_viewer_is_dropped_without_holding_up_offer(monkeypatch):
    monkeypatch.setattr(preview, "SEND_TIMEOUT_S", 0.15)
    monkeypatch.setattr(preview, "SOCKET_BUFFER_BYTES", 4096)
    with publisher(fps=30) as pub:
        client = socket.create_connection(("127.0.0.1", pub.port), timeout=2)
        client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        client.sendall(
            b"GET /cameras/top/stream HTTP/1.0\r\n"
            b"X-Yamkit-Preview-Token: test-secret\r\n\r\n"
        )
        try:
            wait_until(lambda: pub.viewers == 1)
            random = np.random.default_rng(0)
            frame = random.integers(0, 256, (480, 640, 3), dtype=np.uint8)
            pub.offer("top", frame)
            # The peer reads nothing. A JPEG exceeds both deliberately small socket buffers;
            # the serving thread times out while acquisition can keep offering new frames.
            time.sleep(0.04)
            pub.offer("top", frame.copy())
            assert pub.slot("top").accepted == 2
            wait_until(lambda: pub.viewers == 0)
            assert pub.slot("top").offer_errors == 0
        finally:
            client.close()


def test_encoder_failure_does_not_break_offer_and_recovers(monkeypatch):
    with publisher() as pub:
        connection, response = connect(pub)
        try:
            original_encode = pub._encode
            monkeypatch.setattr(pub, "_encode", lambda *a: (_ for _ in ()).throw(RuntimeError("broken JPEG")))
            pub.offer("top", np.zeros((16, 16, 3), dtype=np.uint8))
            wait_until(lambda: pub.slot("top").encode_errors == 1)
            assert pub.status()["cameras"]["top"]["state"] == "unavailable"
            monkeypatch.setattr(pub, "_encode", original_encode)
            time.sleep(0.1)
            pub.offer("top", np.full((16, 16, 3), 100, dtype=np.uint8))
            read_part(response)
            assert pub.status()["cameras"]["top"]["state"] == "live"
        finally:
            response.close()
            connection.close()


def test_close_bounds_stuck_encoder_and_unblocks_waiting_clients(monkeypatch):
    pub = preview.PreviewPublisher("test-session", "test-secret", {"top": "rgb"}).start()
    entered = threading.Event()
    release = threading.Event()
    original_encode = pub._encode

    def slow_encode(frame, mode):
        entered.set()
        release.wait(3)
        return original_encode(frame, mode)

    monkeypatch.setattr(pub, "_encode", slow_encode)
    connection, response = connect(pub)
    server = pub._server
    try:
        pub.offer("top", np.zeros((16, 16, 3), dtype=np.uint8))
        assert entered.wait(2)
        t0 = time.monotonic()
        pub.close(timeout=0.1)
        assert time.monotonic() - t0 < 0.5
        assert response.read() == b""
        assert pub.slot("top")._pending is None
        assert pub.slot("top").latest is None
        wait_until(lambda: pub.viewers == 0 and not server._connections)
        assert not pub._server_thread.is_alive()
        release.set()
        pub._worker.join(2)
        assert not pub._worker.is_alive()
        assert pub.slot("top").latest is None
        pub.close()
    finally:
        release.set()
        response.close()
        connection.close()
        pub.close()


def test_environment_activation_failure_cleans_up(monkeypatch):
    assert isinstance(preview.start_from_env({"top": "rgb"}, {}), preview.NullPreview)
    preview.NullPreview().offer("top", object(), source_time=1.0)
    publishers = []

    def fail_announce(pub):
        publishers.append(pub)
        raise OSError("stdout closed")

    monkeypatch.setattr(preview.PreviewPublisher, "announce", fail_announce)
    environment = {preview.ENV_SESSION: "test-session", preview.ENV_TOKEN: "test-secret"}
    assert isinstance(preview.start_from_env({"top": "rgb"}, environment, owner="owner"), preview.NullPreview)
    assert len(publishers) == 1
    assert publishers[0].port is None
    assert not publishers[0]._worker.is_alive()
    assert not publishers[0]._server_thread.is_alive()


def test_partial_startup_thread_failure_closes_listener(monkeypatch):
    publishers = []
    original_start = preview.PreviewPublisher.start

    def remember_start(pub):
        publishers.append(pub)
        return original_start(pub)

    def fail_thread_start(thread):
        raise RuntimeError("thread limit")

    monkeypatch.setattr(preview.PreviewPublisher, "start", remember_start)
    monkeypatch.setattr(threading.Thread, "start", fail_thread_start)
    env = {preview.ENV_SESSION: "s", preview.ENV_TOKEN: "t"}
    assert isinstance(preview.start_from_env({"top": "rgb"}, env), preview.NullPreview)
    assert publishers[0].port is None
