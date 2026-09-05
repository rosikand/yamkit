#!/usr/bin/env python3
"""Measure the preview handoff using synthetic images and real loopback MJPEG viewers.

No robot, camera, dataset, or Hub connection is created. Each scenario runs in a fresh
process so RSS high-water marks are comparable. See docs/preview-benchmark.md.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import platform
import resource
import secrets
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from yamkit.preview import TOKEN_HEADER, PreviewPublisher

CAMERAS = ("top", "left_wrist", "right_wrist")
SCENARIOS = {"off": (0, False), "on": (1, False), "slow": (1, True), "multiple": (3, False)}
COUNTERS = ("offered", "accepted", "copied", "encoded", "rate_skipped", "replayed", "dropped", "errors")


def percentiles(samples: list[float], scale: float = 1.0) -> dict[str, float]:
    values = np.percentile(samples, [50, 95, 99]) if samples else [0, 0, 0]
    return {name: round(float(value) * scale, 4) for name, value in zip(("p50", "p95", "p99"), values)}


def rss_mib() -> float:
    # Linux current RSS; resource alone only reports the lifetime high-water mark.
    resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)


class Viewer:
    """One real HTTP viewer, with bounded reads and no image queue or JPEG decoding."""

    def __init__(self, port: int, token: str, camera: str, slow: bool) -> None:
        self.port, self.token, self.camera, self.slow = port, token, camera, slow
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.connection: http.client.HTTPConnection | None = None
        self.frames = self.bytes = self.connections = self.disconnects = 0
        self.errors: list[str] = []
        self.thread = threading.Thread(target=self.run, name=f"benchmark-viewer-{camera}", daemon=True)

    def run(self) -> None:
        while not self.stop.is_set():
            conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
            self.connection = conn
            try:
                conn.connect()
                if self.slow:
                    conn.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
                conn.request("GET", f"/cameras/{self.camera}/stream", headers={TOKEN_HEADER: self.token})
                response = conn.getresponse()
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}")
                self.connections += 1
                self.ready.set()
                if self.slow:
                    # 10 KiB/s is far below these 640x480 JPEG streams. Small receive
                    # buffers exercise server backpressure, timeout, and reconnects.
                    while not self.stop.is_set():
                        block = response.read(1024)
                        if not block:
                            break
                        self.bytes += len(block)
                        self.stop.wait(0.1)
                else:
                    while not self.stop.is_set():
                        line = response.readline(4096)
                        if not line:
                            break
                        if not line.startswith(b"--"):
                            continue
                        length = None
                        while line := response.readline(4096):
                            if line == b"\r\n":
                                break
                            key, _, value = line.partition(b":")
                            if key.lower() == b"content-length":
                                length = int(value)
                        if length is None or not 0 < length <= 4 * 1024 * 1024:
                            raise RuntimeError("missing or oversized MJPEG frame")
                        remaining = length
                        while remaining and not self.stop.is_set():
                            block = response.read(min(16384, remaining))
                            if not block:
                                break
                            self.bytes += len(block)
                            remaining -= len(block)
                        if remaining:
                            break
                        self.frames += 1
                response.close()
            except (OSError, http.client.HTTPException, RuntimeError, ValueError) as exc:
                if not self.stop.is_set() and len(self.errors) < 5:
                    self.errors.append(type(exc).__name__ + ": " + str(exc))
            finally:
                conn.close()
                self.connection = None
                self.disconnects += 1
            self.stop.wait(0.05)

    def close(self) -> None:
        self.stop.set()
        conn = self.connection
        if conn and conn.sock:
            try:
                conn.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.thread.join(2.5)


def run_scenario(name: str, duration: float, warmup: float) -> dict:
    import cv2

    viewer_count, slow = SCENARIOS[name]
    token = secrets.token_urlsafe(32)
    initial_rss = rss_mib()
    pub = PreviewPublisher("synthetic-benchmark", token, dict.fromkeys(CAMERAS, "rgb"), fps=10).start()
    viewers = [Viewer(pub.port, token, camera, slow) for camera in CAMERAS for _ in range(viewer_count)]
    # Fixed textured RGB backgrounds with a moving bar. Allocate a distinct frame
    # per camera/tick; no preallocated frame ring masks the handoff's memory costs.
    rng = np.random.default_rng(20260905)
    bases = [rng.integers(0, 256, (480, 640, 3), dtype=np.uint8) for _ in CAMERAS]
    handoff, loops, intervals, rss_samples = [], [], [], []
    ticks = 0
    close_s = None
    try:
        for viewer in viewers:
            viewer.thread.start()
        for viewer in viewers:
            if not viewer.ready.wait(3):
                raise RuntimeError(f"viewer failed to connect: {viewer.errors}")
        phase_start = time.perf_counter()
        next_tick = phase_start
        measurement_start = None
        previous_tick = None
        before = None
        source_tick = 0
        while True:
            now = time.perf_counter()
            if measurement_start is None and now - phase_start >= warmup:
                before = pub.status()["cameras"]
                measurement_start = now
                cpu_start = time.process_time()
                measurement_rss = rss_mib()
                viewer_start = [(v.frames, v.bytes, v.connections, v.disconnects) for v in viewers]
            measuring = measurement_start is not None
            if measuring and now - measurement_start >= duration:
                break
            loop_start = time.perf_counter_ns()
            if measuring and previous_tick is not None:
                intervals.append((loop_start - previous_tick) * 1e-6)
            previous_tick = loop_start if measuring else None
            for camera, base in zip(CAMERAS, bases):
                frame = base.copy()
                x = (source_tick * 11) % 620
                frame[:, x : x + 20] = (255, 64, 16)
                started = time.perf_counter_ns()
                pub.offer(camera, frame)
                if measuring:
                    handoff.append(time.perf_counter_ns() - started)
            if measuring:
                loops.append(time.perf_counter_ns() - loop_start)
                ticks += 1
                if ticks % 3 == 0:
                    rss_samples.append(rss_mib())
            source_tick += 1
            next_tick += 1 / 30
            time.sleep(max(0, next_tick - time.perf_counter()))
        elapsed = time.perf_counter() - measurement_start
        cpu_s = time.process_time() - cpu_start
        after = pub.status()["cameras"]
        counters = {
            camera: {key: after[camera].get(key, 0) - before[camera].get(key, 0) for key in COUNTERS}
            for camera in CAMERAS
        }
        totals = {key: sum(camera[key] for camera in counters.values()) for key in COUNTERS}
        viewer_results = [
            {
                "camera": viewer.camera,
                "frames": viewer.frames - start[0],
                "bytes": viewer.bytes - start[1],
                "connections": viewer.connections - start[2],
                "disconnects": viewer.disconnects - start[3],
                "errors": viewer.errors,
            }
            for viewer, start in zip(viewers, viewer_start)
        ]
        result = {
            "scenario": name,
            "duration_s": round(elapsed, 3),
            "ticks": ticks,
            "source_hz": round(ticks / elapsed, 3),
            "viewer_count": len(viewers),
            "active_viewers_at_end": {camera: after[camera]["viewers"] for camera in CAMERAS},
            "handoff_us": percentiles(handoff, 1e-3),
            "loop_work_ms": percentiles(loops, 1e-6),
            "loop_interval_ms": percentiles(intervals),
            "cpu_s": round(cpu_s, 3),
            "cpu_percent_one_core": round(cpu_s / elapsed * 100, 2),
            "rss_initial_mib": round(initial_rss, 2),
            "rss_measurement_start_mib": round(measurement_rss, 2),
            "rss_peak_sampled_mib": round(max(rss_samples, default=measurement_rss), 2),
            "rss_final_mib": round(rss_mib(), 2),
            "rss_process_high_water_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
            "counters": totals,
            "cameras": counters,
            "viewers": viewer_results,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        }
    finally:
        started = time.perf_counter()
        pub.close()
        for viewer in viewers:
            viewer.close()
        close_s = time.perf_counter() - started
    result["cleanup_s"] = round(close_s, 3)
    result["viewer_threads_alive_after_cleanup"] = sum(viewer.thread.is_alive() for viewer in viewers)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=10, help="measured seconds per scenario (default: 10)")
    parser.add_argument("--warmup", type=float, default=1, help="unmeasured seconds per scenario (default: 1)")
    parser.add_argument("--output", type=Path, help="optional JSON report path inside this repository")
    parser.add_argument("--scenario", choices=SCENARIOS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.duration <= 0 or args.warmup < 0:
        parser.error("duration must be positive and warmup must be nonnegative")
    if args.scenario:
        print(json.dumps(run_scenario(args.scenario, args.duration, args.warmup)))
        return
    report = {
        "environment": {"platform": platform.platform(), "python": platform.python_version(), "cpus": os.cpu_count()},
        "method": {
            "sources": 3,
            "shape": [480, 640, 3],
            "color": "RGB uint8",
            "source_hz": 30,
            "preview_fps_target": 10,
            "warmup_s": args.warmup,
            "duration_s": args.duration,
            "seed": 20260905,
            "slow_viewer_bytes_per_second": 10240,
            "slow_viewer_receive_buffer_requested_bytes": 4096,
            "cpu_scope": "publisher plus synthetic acquisition and loopback client threads",
        },
        "results": [],
    }
    for scenario in SCENARIOS:
        print(f"Benchmarking {scenario}...", file=sys.stderr, flush=True)
        completed = subprocess.run(
            [sys.executable, __file__, "--scenario", scenario,
             "--duration", str(args.duration), "--warmup", str(args.warmup)],
            check=True, capture_output=True, text=True, timeout=args.duration + args.warmup + 30,
        )
        report["results"].append(json.loads(completed.stdout))
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        root = Path(__file__).resolve().parents[1]
        if not args.output.resolve().is_relative_to(root):
            parser.error("output must stay inside this repository")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
