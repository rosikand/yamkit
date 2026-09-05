#!/usr/bin/env python3
"""Actual Chrome/HTTP UI smoke with synthetic cameras and harmless session children.

Run: TMPDIR="$PWD/.context/tmp" .venv/bin/python scripts/browser_smoke.py
Requires the normal project environment (including websockets) and google-chrome.
No physical camera/arm, model weights, Hub upload, or paid API is used. The recording
child simulates phases; installed LeRobot recording/reset coverage lives in
tests/test_preview_plugins.py. All writable artifacts stay inside this repository.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = ("top", "left_wrist", "right_wrist")


def wait_for(predicate, *, timeout=12, description="condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {description}")


def child(mode: str, control: Path) -> None:
    """Use the real ownership handshake and publisher; never import a robot driver."""
    if mode not in ("record", "teleop"):
        print('[yamkit-result] ' + json.dumps({"fixture": True, "ready": True, "operation": mode}), flush=True)
        time.sleep(30 if mode == "rollout" else 0.4)
        return
    import numpy as np

    from yamkit.camera_ownership import claim_from_env
    from yamkit.preview import start_from_env

    lease = claim_from_env(list(CAMERAS))
    preview = start_from_env(dict.fromkeys(CAMERAS, "rgb"), owner=lease.owner)
    prior = None
    seq = 0
    try:
        while True:
            phase = control.read_text().strip()
            if phase != prior:
                if phase == "record":
                    print("Recording episode 0", flush=True)
                elif phase == "reset":
                    print("Reset the environment", flush=True)
                elif phase == "failed":
                    preview.close()  # keep the lease: competing capture must remain blocked
                elif phase == "upload":
                    preview.close()
                    lease.release()
                    print("[yamkit] recording finished — uploading browser fixture", flush=True)
                prior = phase
            if phase in ("record", "reset"):
                seq += 1
                for index, name in enumerate(CAMERAS):
                    frame = np.full((120, 160, 3), (seq + index * 70) % 255, dtype=np.uint8)
                    preview.offer(name, frame, source_time=time.perf_counter())
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass
    finally:
        preview.close()
        lease.release()


class Browser:
    """Minimal synchronous Chrome DevTools client; no downloaded browser or driver."""

    def __init__(self, work: Path):
        from websockets.sync.client import connect

        executable = shutil.which("google-chrome")
        if executable is None:
            raise RuntimeError("google-chrome is required for this optional browser check")
        profile = work / "chrome-profile"
        self.log = (work / "chrome.log").open("w")
        self.proc = subprocess.Popen([
            executable, "--headless=new", "--no-sandbox", "--disable-dev-shm-usage",
            "--disable-background-networking", "--disable-component-update", "--no-first-run",
            "--no-default-browser-check", "--disable-sync", "--disable-extensions", "--no-proxy-server",
            "--disable-breakpad", "--remote-debugging-port=0", "--remote-allow-origins=http://localhost",
            "--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE localhost, EXCLUDE 127.0.0.1",
            f"--user-data-dir={profile}", "about:blank",
        ], stdout=self.log, stderr=subprocess.STDOUT)
        try:
            active = profile / "DevToolsActivePort"
            wait_for(active.exists, description="Chrome debugging port")
            port = active.read_text().splitlines()[0]
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=5) as response:
                target = next(target for target in json.load(response) if target["type"] == "page")
            self.ws = connect(target["webSocketDebuggerUrl"], origin="http://localhost", max_size=None)
            self.serial = 0
            self.exceptions = []
            self.requests = []
            self.call("Runtime.enable")
            self.call("Network.enable")
            self.call("Page.enable")
            self.call("Page.addScriptToEvaluateOnNewDocument", {"source": """
                window.smokeDialogs = []; window.smokeAccept = false;
                window.confirm = message => { smokeDialogs.push(message); return smokeAccept; };
                window.alert = message => { smokeDialogs.push(message); };
            """})
        except BaseException:
            self.close()
            raise

    def call(self, method, params=None):
        self.serial += 1
        request_id = self.serial
        self.ws.send(json.dumps({"id": request_id, "method": method, "params": params or {}}))
        while True:
            message = json.loads(self.ws.recv(timeout=20))
            if message.get("method") == "Runtime.exceptionThrown":
                self.exceptions.append(message["params"]["exceptionDetails"])
            if message.get("method") == "Network.requestWillBeSent":
                self.requests.append(message["params"]["request"])
            if message.get("id") == request_id:
                if "error" in message:
                    raise AssertionError(message["error"])
                return message.get("result", {})

    def evaluate(self, expression):
        result = self.call("Runtime.evaluate", {"expression": expression, "returnByValue": True, "awaitPromise": True})
        if "exceptionDetails" in result:
            raise AssertionError(result["exceptionDetails"])
        return result.get("result", {}).get("value")

    def wait(self, expression):
        return wait_for(lambda: self.evaluate(f"Boolean({expression})"), description=expression)

    def click(self, selector):
        self.evaluate(f"document.querySelector({json.dumps(selector)}).click()")

    def value(self, selector, value):
        self.evaluate(f"""(() => {{const el = document.querySelector({json.dumps(selector)});
            el.value = {json.dumps(value)}; el.dispatchEvent(new Event('input', {{bubbles: true}}));
            el.dispatchEvent(new Event('change', {{bubbles: true}}));}})()""")

    def close(self):
        if hasattr(self, "ws"):
            self.ws.close()
        self.proc.terminate()
        try:
            self.proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)
        self.log.close()


def run(work: Path) -> dict:
    import cv2
    import numpy as np
    import uvicorn

    from yamkit import arm, hub, modal_ops
    from yamkit.config import ArmSpec, PairSpec, RigConfig
    from yamkit.inference.service import ModelRuntime
    from yamkit.ui import camstream, server
    from yamkit.ui.sessions import SessionManager

    checks, forbidden_calls, seen, direct_starts = [], [], [], []
    control = work / "phase.txt"
    control.write_text("record")
    rig = RigConfig(
        arms={f"{side}_{role}": ArmSpec(
            name=f"{side}_{role}", role=role, side=side, can_serial=f"fake-{side}-{role}",
            gripper="yam_teaching_handle" if role == "leader" else "linear_4310",
            gripper_limits=None if role == "leader" else [0, 6.5],
        ) for side in ("left", "right") for role in ("leader", "follower")},
        pairs=[PairSpec(f"{side}_leader", f"{side}_follower") for side in ("left", "right")],
        cameras={name: {"type": "opencv", "index_or_path": 900 + index, "width": 640, "height": 480, "fps": 30}
                 for index, name in enumerate(CAMERAS)},
    )
    rig.save(work / "rig.yaml")
    np.savez(work / "snapshot.npz", fixture=np.zeros(1))
    manager = SessionManager()

    def forbidden(*args, **kwargs):
        forbidden_calls.append("hardware/model/paid service attempted")
        raise AssertionError(forbidden_calls[-1])

    original_connect = socket.socket.connect

    def local_connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6) and address[0] not in ("127.0.0.1", "::1", "localhost"):
            return forbidden()
        return original_connect(sock, address)

    def camera_loop(camera):
        direct_starts.append(camera.name)
        seq = 0
        while not camera._stop.is_set():
            seq += 1
            frame = np.full((120, 160, 3), seq % 255, dtype=np.uint8)
            ok, jpg = cv2.imencode(".jpg", frame)
            assert ok
            with camera.cond:
                camera.frame = jpg.tobytes()
                camera.frame_t = time.time()
                camera.cond.notify_all()
            camera._stop.wait(0.1)

    def argv(*args):
        seen.append(args)
        return [sys.executable, "-u", str(Path(__file__).resolve()), "--child", args[0], "--control", str(control)]

    manager.yamkit_argv = argv
    with contextlib.ExitStack() as stack:
        for target, attribute, replacement in (
            (arm.YamArm, "connect", forbidden), (cv2, "VideoCapture", forbidden),
            (ModelRuntime, "load", forbidden), (modal_ops, "prepare", forbidden),
            (modal_ops, "service_handle", forbidden), (modal_ops, "owned_service", lambda: None),
            (hub, "get_token", lambda: None), (camstream._Camera, "_loop", camera_loop),
            (socket.socket, "connect", local_connect),
        ):
            stack.enter_context(patch.object(target, attribute, replacement))
        stack.enter_context(patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "HF_TOKEN": "", "MODAL_TOKEN_ID": "",
                                                   "MODAL_TOKEN_SECRET": "", "YAMKIT_OPENAI_API_KEY": ""}))
        hub.clear_cache()
        app = server.create_app(rig.path, datasets_dir=work / "datasets", outputs_dir=work / "outputs",
                                frontend_dir=ROOT / "ui", session_manager=manager)
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        base = f"http://127.0.0.1:{listener.getsockname()[1]}"
        ui_server = uvicorn.Server(uvicorn.Config(app, log_level="error", timeout_graceful_shutdown=2))
        thread = threading.Thread(target=ui_server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        wait_for(lambda: ui_server.started, description="fixture HTTP server")
        browser = Browser(work)

        def check(name, condition=True):
            assert condition, name
            checks.append(name)
            print(f"PASS {name}", flush=True)

        def stop():
            browser.evaluate("fetch('/api/session/stop', {method: 'POST'}).then(r => r.json())")
            wait_for(lambda: not manager.active, description="local child stop")
            manager.wait(timeout=5)
            browser.wait("!session.active")

        def cameras(source, state="live"):
            browser.wait(f"overview?.cameras?.length === 3 && overview.cameras.every(c => c.preview_source === '{source}' && c.preview_state === '{state}')")
            browser.wait("[...document.querySelectorAll('.cam img')].length === 3 && [...document.querySelectorAll('.cam img')].every(img => img.naturalWidth === 160)")

        def finish_operation(button, command):
            browser.wait(f"!document.querySelector({json.dumps(button)}).disabled")
            before = len(seen)
            browser.click(button)
            wait_for(lambda: len(seen) > before and seen[-1][0] == command, description=command)
            wait_for(lambda: manager.wait(timeout=0.05) is not None, description=f"{command} finish")
            operation = json.dumps(manager.meta["operation_id"])
            browser.wait(f"!session.active && session.meta.operation_id === {operation} && document.querySelector('#inf-result').textContent.includes('fixture')")

        try:
            navigation = browser.call("Page.navigate", {"url": base + "/#/inference"})
            assert "errorText" not in navigation, navigation
            browser.wait("document.querySelector('#inf-backend') && pages.inference._profiles.length === 3")
            cameras("direct")
            check("Inference page load starts no session, model, motors or paid API", not seen and not forbidden_calls)
            check("Local backend default and unsupported base mapping visible", browser.evaluate("document.querySelector('#inf-backend').value === 'local' && document.querySelector('#btn-ro').disabled && document.querySelector('#inf-profile-note').textContent.length > 30"))
            finish_operation("#btn-pc", "policy-check")
            check("Policy check uses shared CLI and displays fixture readiness")
            browser.value("#inf-task", "changed instruction")
            check("Changed selection invalidates displayed readiness", browser.evaluate("document.querySelector('#inf-result').textContent === ''"))
            browser.value("#inf-preset", "custom")
            browser.value("#inf-policy", "outputs/custom-checkpoint")
            check("Custom checkpoint selection remains available", browser.evaluate("document.querySelector('#inf-policy').value === 'outputs/custom-checkpoint'"))
            browser.value("#inf-preset", "molmoact2")
            browser.value("#inf-backend", "modal")
            check("Modal selects GPU and disables guided RTC", browser.evaluate("document.querySelector('#inf-rtc').disabled && !document.querySelector('#inf-rtc').checked && !document.querySelector('#btn-prepare').disabled"))
            finish_operation("#btn-prepare", "modal-prepare")
            check("Prepare Modal uses fixture child and displays result")
            browser.value("#inf-saved", str((work / "snapshot.npz").relative_to(ROOT)))
            finish_operation("#btn-probe-saved", "policy-probe")
            check("Saved probe passes saved path through shared CLI", "--saved" in seen[-1] and "--live" not in seen[-1])
            before = len(seen)
            browser.click("#btn-probe-live")
            check("Declining active-read confirmation starts no child", len(seen) == before and browser.evaluate("smokeDialogs.at(-1).includes('ACTIVE READ')"))
            browser.evaluate("smokeAccept = true")
            finish_operation("#btn-probe-live", "policy-probe")
            check("Confirmed active-read uses explicit CLI approval", "--approve-active-read" in seen[-1])
            browser.click("#btn-ro")
            wait_for(lambda: manager.active and manager.mode == "rollout", description="fixture rollout")
            browser.wait("session.active && !document.querySelector('#btn-inf-stop').disabled")
            browser.click("#btn-inf-stop")
            wait_for(lambda: not manager.active, description="Stop local execution")
            manager.wait(timeout=5)
            browser.wait("!session.active")
            check("Start and Stop control only the local fixture rollout child", seen[-1][0] == "rollout")

            browser.evaluate("location.hash = '#/record'")
            browser.wait("document.querySelector('#btn-record')")
            cameras("direct")
            browser.click("#btn-teleop")
            cameras("session")
            check("Dynamic camera ownership works for camera-owning teleop")
            stop()
            cameras("direct")
            browser.value("#rec-name", "browser_fixture")
            browser.value("#rec-task", "synthetic recording")
            browser.click("#btn-record")
            cameras("session")
            check("Recording displays recorder-owned JPEGs through browser /stream")
            browser.evaluate("window.smokeImages = [...document.querySelectorAll('.cam img')]; window.smokeURLs = smokeImages.map(i => i.src)")
            time.sleep(3.2)
            check("Healthy MJPEG nodes and URLs survive normal polling", browser.evaluate("smokeImages.every((img, i) => img === document.querySelectorAll('.cam img')[i] && img.src === smokeURLs[i])"))
            control.write_text("reset")
            browser.wait("session.parsed?.phase === 'reset'")
            cameras("session")
            check("Reset retains recorder ownership and fresh previews")
            browser.call("Page.reload")
            browser.wait("document.querySelector('#btn-record')")
            cameras("session")
            check("Browser reload reconnects to active recorder preview")
            control.write_text("pause")
            browser.wait("overview.cameras.every(c => c.preview_state === 'stale')")
            age = browser.evaluate("overview.cameras[0].frame_age_s")
            time.sleep(1.4)
            browser.wait(f"overview.cameras[0].frame_age_s > {age}")
            check("Paused acquisition shows increasing stale image age")
            control.write_text("record")
            cameras("session")
            before = len(direct_starts)
            control.write_text("failed")
            browser.wait("overview.cameras.every(c => c.preview_state === 'unavailable')")
            check("Failed publisher never opens competing direct cameras", len(direct_starts) == before and manager.cameras_owned)
            control.write_text("upload")
            browser.wait("session.active && session.parsed?.phase === 'upload'")
            cameras("direct")
            check("Released recorder returns to direct preview during simulated upload", not manager.cameras_owned and manager.active)
            stop()
            cameras("direct")
            check("Stop leaves idle previews working and no session child")
            check("Zero JavaScript exceptions", not browser.exceptions)
            check("Zero real camera, arm, model-load, or paid-service attempts", not forbidden_calls)
            check("Frontend requests stay on fixture loopback origin", all(r["url"].startswith(base) or r["url"] == "about:blank" for r in browser.requests))
            return {"passed": len(checks), "checks": checks, "javascript_exceptions": browser.exceptions,
                    "forbidden_calls": forbidden_calls, "child_commands": [args[0] for args in seen],
                    "physical_hardware": False, "real_services": False,
                    "scope": "Chrome, actual UI/HTTP/session/preview paths; synthetic acquisition and phase simulation"}
        except BaseException:
            print(json.dumps({"body": browser.evaluate("document.body.innerText.slice(-1600)"),
                              "url": browser.evaluate("location.href"),
                              "exceptions": browser.exceptions, "session": manager.status()}, default=str), flush=True)
            raise
        finally:
            browser.close()
            manager.stop(grace_s=0.1)
            manager.wait(timeout=5)
            ui_server.should_exit = True
            thread.join(timeout=8)
            listener.close()
            assert not thread.is_alive(), "fixture HTTP server did not stop"
            hub.clear_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".context/browser-smoke.json")
    parser.add_argument("--child", choices=("record", "teleop", "policy-check", "policy-probe", "modal-prepare", "rollout"))
    parser.add_argument("--control", type=Path)
    args = parser.parse_args()
    if args.child:
        child(args.child, args.control)
        return
    temp_root = ROOT / ".context/tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(temp_root)
    with tempfile.TemporaryDirectory(prefix="browser-smoke-", dir=temp_root) as directory:
        result = run(Path(directory))
    output = args.output.resolve()
    if ROOT not in output.parents:
        raise ValueError("output must stay inside this repository")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{result['passed']} browser checks passed; report: {output}")


if __name__ == "__main__":
    main()
