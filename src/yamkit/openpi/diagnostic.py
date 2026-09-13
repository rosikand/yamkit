"""Bounded saved-RGB official inference; never a YAM qualification or rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .assets import local_path
from .contract import SAVED_IMAGE_MAP, identity
from .runtime import OfficialPi05Diagnostic, normalized_observation, validate_normalized_chunk

MAX_SAVED_OBSERVATIONS = 50
MAX_SAVED_FILE_BYTES = 16 * 1024 * 1024


def load_saved_images(path: Path) -> dict:
    """Read RGB only. Discarding recorded joints is explicit, not disguised normalization."""
    if path.suffix != ".npz" or not path.is_file() or path.stat().st_size > MAX_SAVED_FILE_BYTES:
        raise ValueError("Saved observations must be NPZ files of at most 16 MiB")
    # Check the archive before NumPy allocates/decompresses its members. This also
    # bounds hidden/unused state members without loading their potentially pickled data.
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if (len(entries) > 16 or sum(item.file_size for item in entries) > MAX_SAVED_FILE_BYTES
                or len({item.filename for item in entries}) != len(entries)):
            raise ValueError("Saved observation archive exceeds bounded unique member sizes")
    with np.load(path, allow_pickle=False) as saved:
        if not set(SAVED_IMAGE_MAP).issubset(saved.files):
            raise ValueError("Saved observation lacks top/left_wrist/right_wrist RGB")
        images = {native: saved[source].copy() for source, native in SAVED_IMAGE_MAP.items()}
    normalized_observation(images, "validate image schema")
    return images


def _write_json(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def run_diagnostic(runtime, images: list[dict], task: str, requests: int, output: Path) -> dict:
    if (type(requests) is not int or not 1 <= requests <= 50
            or not 1 <= len(images) <= MAX_SAVED_OBSERVATIONS):
        raise ValueError("Official diagnostic needs 1..50 bounded calls and saved RGB observations")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        **identity(), **runtime.provenance, "checked_at": datetime.now(UTC).isoformat(),
        "task": task, "requested_warm_samples": requests, "completed_warm_samples": 0,
        "saved_rgb_observations": len(images), "saved_measured_states_used": False,
        "software_inference_passed": False, "qualified_for_yam": False,
        "fake_yam_rollout_attempted": False, "physical_task_success": None,
        "model_rows_returned": 0, "model_rows_sent_to_robot": 0,
        "unexpected_rows_dropped": 0, "output_modifications": 0,
    }
    try:
        # Fixed input noise makes the wrapper/native comparison reproducible without
        # changing official flow matching, its steps, or any action values.
        noise = np.random.default_rng(42).standard_normal((50, 32), dtype=np.float32)
        _write_json(output / "parity-noise.json", {"noise": noise.tolist()})
        started = time.monotonic()
        wrapped = runtime.predict(images[0], task, noise=noise)
        report["cold_s"] = time.monotonic() - started
        native = validate_normalized_chunk(runtime.policy.infer(
            normalized_observation(images[0], task), noise=noise)["actions"])
        report["same_noise_native_wrapper_parity"] = {
            "bitwise_equal": bool(np.array_equal(wrapped, native)),
            "max_abs_error": float(np.max(np.abs(wrapped - native))),
            "native_shape": list(native.shape), "noise_sha256": hashlib.sha256(noise.tobytes()).hexdigest(),
        }
        _write_json(output / "native-parity-chunks.json", {"wrapped": wrapped.tolist(), "native": native.tolist()})
        if not np.array_equal(wrapped, native):
            raise ValueError("Official wrapper output differs from the actual native policy at fixed noise")
        durations = []
        for index in range(requests):
            started = time.monotonic()
            chunk = runtime.predict(images[index % len(images)], task)
            elapsed = time.monotonic() - started
            durations.append(elapsed)
            _write_json(output / f"native-chunk-{index:03d}.json", {
                "index": index, "saved_rgb_index": index % len(images), "round_trip_s": elapsed,
                "output_space": "raw normalized model space; no physical meaning established",
                "chunk": chunk.tolist(),
            })
            report["completed_warm_samples"] += 1
            report["model_rows_returned"] += 50
            print(json.dumps({"sample": index + 1, "total": requests, "round_trip_s": elapsed,
                              "native_shape": list(chunk.shape), "physical_ready": False}), flush=True)
        report["warm_round_trip_s"] = {
            "sample_count": len(durations), "p50": float(np.percentile(durations, 50)),
            "p95": float(np.percentile(durations, 95)), "max": max(durations), "min": min(durations),
        }
        report["software_inference_passed"] = True
    except Exception as error:  # noqa: BLE001 — preserve failure evidence without echoing secrets
        # Do not echo dependency/HTTP exception text which could contain credentials.
        report["failure_type"] = type(error).__name__
    finally:
        # Readiness cannot be overwritten by a supplied runtime or a plausible array.
        report.update(physical_ready=False, hardware_tested=False, qualified_for_yam=False)
        _write_json(output / "diagnostic.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--saved-observation", type=Path, action="append", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--requests", type=int, default=5, choices=range(1, 51))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        output = local_path(root, args.output)
    except ValueError:
        parser.error("Diagnostic output must stay inside this checkout")
    if output.exists():
        parser.error("Diagnostic output already exists; preserve it and select a new directory")
    phase = "saved_input_validation"
    try:
        if not 1 <= len(args.saved_observation) <= MAX_SAVED_OBSERVATIONS:
            raise ValueError("Use at most 50 saved observation files")
        paths = [local_path(root, path) for path in args.saved_observation]
        images = [load_saved_images(path) for path in paths]
        normalized_observation(images[0], args.task)
        phase = "official_runtime_load"
        runtime = OfficialPi05Diagnostic.load(root)
    except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001 — dependency failures must not echo secrets
        output.mkdir(parents=True, exist_ok=False)
        report = {
            **identity(), "checked_at": datetime.now(UTC).isoformat(), "failure_phase": phase,
            "failure_type": type(error).__name__, "software_inference_passed": False,
            "qualified_for_yam": False, "physical_task_success": None,
            "model_rows_sent_to_robot": 0, "saved_measured_states_used": False,
            "reason": ("Saved inputs were rejected before model loading; use 1–50 repo-local NPZ files, "
                       "at most 16 MiB each including uncompressed members, with three uint8 RGB views."
                       if phase == "saved_input_validation" else
                       "Official model load failed; verify the isolated pinned runtime, asset receipt, "
                       "and available GPU memory. No YAM physical contract was established."),
        }
        _write_json(output / "diagnostic.json", report)
        print(json.dumps(report))
        raise SystemExit(130 if isinstance(error, KeyboardInterrupt) else 2) from None
    report = run_diagnostic(runtime, images, args.task, args.requests, output)
    print(json.dumps({"software_inference_passed": report["software_inference_passed"],
                      "physical_ready": False, "report": str(output / "diagnostic.json")}))
    raise SystemExit(0 if report["software_inference_passed"] else 2)


if __name__ == "__main__":
    main()
