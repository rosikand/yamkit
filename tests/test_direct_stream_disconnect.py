"""A stalled camera must not strand HTTP viewers in the API's shared worker pool."""

import socket
import threading
import time
import urllib.request

import anyio
import pytest
import uvicorn

from yamkit.preview import MJPEG_MEDIA_TYPE
from yamkit.ui.camstream import CameraStreamingResponse, _Camera
from yamkit.ui.server import create_app


def wait_for(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail("stalled camera viewer did not release")


@pytest.fixture
def stalled_cameras(monkeypatch):
    cameras = []

    def loop(camera):
        cameras.append(camera)
        with camera.cond:
            camera.frame = b"synthetic first JPEG; no physical acquisition"
            camera.frame_t = time.time()
            camera.cond.notify_all()
        camera._stop.wait()  # emulate cap.read() producing no more frames
        with camera.cond:
            camera.cond.notify_all()

    monkeypatch.setattr(_Camera, "_loop", loop)
    yield cameras
    for camera in cameras:
        camera.stop(join=True, disable=True)


@pytest.mark.parametrize("asgi_version", ["2.3", "2.4"])
@pytest.mark.parametrize("cancelled", [False, True])
def test_disconnect_or_asgi_cancellation_wakes_stalled_iterator(stalled_cameras, asgi_version, cancelled):
    camera = _Camera("top", {"index_or_path": 999})

    async def run():
        frame_sent = anyio.Event()
        disconnected = anyio.Event()

        async def receive():
            await disconnected.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body":
                frame_sent.set()

        response = CameraStreamingResponse(camera, MJPEG_MEDIA_TYPE)
        with anyio.fail_after(2):
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(response, {"type": "http", "asgi": {"spec_version": asgi_version}}, receive, send)
                await frame_sent.wait()
                await anyio.sleep(0.05)  # worker enters next() waiting for a fresh image
                assert camera.clients == 1
                if cancelled:
                    tasks.cancel_scope.cancel()
                else:
                    disconnected.set()
        assert camera.clients == 0
        assert camera.running and not camera._stop.is_set()

    anyio.run(run)


def test_repeated_http_disconnects_leave_control_api_and_shared_capture_available(rig, tmp_path, stalled_cameras):
    rig.cameras = {"top": {"type": "opencv", "index_or_path": 999, "fps": 30}}
    rig.save()
    app = create_app(rig.path, datasets_dir=tmp_path / "datasets", outputs_dir=tmp_path / "outputs")
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    base = f"http://127.0.0.1:{listener.getsockname()[1]}"
    server = uvicorn.Server(uvicorn.Config(app, log_level="critical", timeout_graceful_shutdown=1))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    survivor = None
    try:
        wait_for(lambda: server.started)
        survivor = urllib.request.urlopen(base + "/api/cameras/top/stream", timeout=2)
        survivor.read(1)
        wait_for(lambda: len(stalled_cameras) == 1)
        camera = stalled_cameras[0]
        for _ in range(45):  # more stale-camera retries than AnyIO's default 40 worker slots
            with urllib.request.urlopen(base + "/api/cameras/top/stream", timeout=2) as response:
                response.read(1)
        wait_for(lambda: camera.clients == 1)
        for path in ("/api/cameras", "/api/session"):
            with urllib.request.urlopen(base + path, timeout=1) as response:
                assert response.status == 200
        assert len(stalled_cameras) == 1 and camera.running and not camera._stop.is_set()
        survivor.close()
        survivor = None
        wait_for(lambda: camera.clients == 0)
    finally:
        if survivor is not None:
            survivor.close()
        for camera in stalled_cameras:
            camera.stop(disable=True)
        server.should_exit = True
        thread.join(timeout=5)
        listener.close()
        assert not thread.is_alive()
