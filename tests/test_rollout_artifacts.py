"""Offline snapshot privacy/integrity and private-Hub retry boundaries."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from huggingface_hub.errors import RepositoryNotFoundError

from yamkit.rollout_artifacts import (
    CAMERAS,
    package_rollout,
    sanitize,
    sanitize_text,
    upload_rollout,
    validate_bundle,
)


def write_json(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def recording(tmp_path):
    run = tmp_path / "20260908-184743-rollout-test"
    trace = tmp_path / "trace"
    run.mkdir()
    trace.mkdir()
    write_json(run / "meta.json", {
        "id": run.name, "ended_at": 100, "returncode": 0, "status": "success", "task_success": False,
    })
    write_json(trace / "summary.json", {
        "status": "TRACE_SAVED", "resources_released": True, "frame_count": 2, "video_fps": 30,
        "counts": {"frames_dropped": 0},
    })
    write_json(trace / "trace.json", {"events": [{"kind": "observation", "positions": [0.1] * 14}],
                                    "chunks": [], "unexpected_token": "should-not-escape"})
    write_json(trace / "metrics.json", {"executed_actions": 1, "home_completed": True})
    write_json(trace / "frame_timestamps.json", [1.0, 1.033])
    write_json(trace / "video_timeline.json", {"nominal_fps": 30, "frames": [{"source_index": 0}]})
    write_json(trace / "plan.json", {"fps": 30, "private_endpoint": "https://private.modal.run"})
    (run / "log.txt").write_text("INFO ready\nAuthorization: Bearer arbitrary-long-credential\n")
    (trace / "report.html").write_text('<video src="top.mp4"></video><a href="trace.json">Trace</a>')
    for side in ("left", "right"):
        (trace / f"joints-{side}.png").write_bytes(b"plot")
    for camera in CAMERAS:
        (trace / f"{camera}.mp4").write_bytes(b"unchanged-video")
        frames = trace / "frames" / camera
        frames.mkdir(parents=True)
        for index in range(2):
            (frames / f"frame-{index:06d}.png").write_bytes(b"original-rgb" + bytes([index]))
    return run, trace


def test_complete_portable_bundle_preserves_media_and_originals(recording):
    run, trace = recording
    original = (run / "log.txt").read_bytes()
    bundle = package_rollout(run, trace_dir=trace, metadata={
        "model": {"repo_id": "allenai/MolmoAct2", "revision": "abcd1234", "api_key": "never"},
        "rig": {"control": {"max_joint_speed": 0.9}}, "unexpected": "not-allowlisted",
        "known_missing_data": ["INFO phase logs missing in this historical run."],
    })
    manifest = validate_bundle(bundle)
    assert manifest["frame_counts"] == dict.fromkeys(CAMERAS, 2)
    assert not any(item.startswith("Missing artifact:") for item in manifest["missing_data"])
    assert (bundle / "report.html").read_bytes() == (trace / "report.html").read_bytes()
    assert (run / "log.txt").read_bytes() == original
    assert "arbitrary-long-credential" not in (bundle / "log.txt").read_text()
    assert json.loads((bundle / "meta.json").read_text())["task_success"] is False
    assert "should-not-escape" not in (bundle / "trace.json").read_text()
    assert "private_endpoint" not in (bundle / "plan.json").read_text()
    metadata = json.loads((bundle / "run_metadata.json").read_text())
    assert metadata["model"] == {"repo_id": "allenai/MolmoAct2", "revision": "abcd1234"}
    assert "unexpected" not in metadata
    assert "INFO phase logs missing" in (bundle / "README.md").read_text()
    for camera in CAMERAS:
        assert (bundle / f"{camera}.mp4").read_bytes() == b"unchanged-video"
        for index in range(2):
            relative = f"frames/{camera}/frame-{index:06d}.png"
            assert (bundle / relative).read_bytes() == (trace / relative).read_bytes()
            assert manifest["files"][relative]["sha256"] == hashlib.sha256((trace / relative).read_bytes()).hexdigest()


def test_snapshot_idempotency_and_conflicting_rebuild(recording):
    run, trace = recording
    bundle = package_rollout(run, trace_dir=trace)
    assert package_rollout(run, trace_dir=trace) == bundle
    (run / "log.txt").write_text("new log")
    with pytest.raises(ValueError, match="different bundle already exists"):
        package_rollout(run, trace_dir=trace)
    assert "new log" not in (bundle / "log.txt").read_text()
    assert package_rollout(run) == bundle  # Existing immutable snapshot can be retried without raw-frame paths.


@pytest.mark.parametrize("change", [{"ended_at": None}, {"returncode": None}, {"active": True}, {"status": "running"}])
def test_only_finalized_runs_package(recording, change):
    run, trace = recording
    meta = json.loads((run / "meta.json").read_text())
    meta.update(change)
    write_json(run / "meta.json", meta)
    with pytest.raises(ValueError, match="finalized run"):
        package_rollout(run, trace_dir=trace)
    assert not (run / "bundle").exists()


@pytest.mark.parametrize("change", [{"resources_released": False}, {"status": "EXPORTING"}])
def test_export_and_hardware_release_required(recording, change):
    run, trace = recording
    summary = json.loads((trace / "summary.json").read_text())
    summary.update(change)
    write_json(trace / "summary.json", summary)
    with pytest.raises(ValueError, match="resources must be released"):
        package_rollout(run, trace_dir=trace)


def test_partial_capture_explicitly_lists_missing_files_and_data(recording):
    run, trace = recording
    (trace / "metrics.json").unlink()
    (trace / "frames/top/frame-000000.png").unlink()
    write_json(trace / "summary.json", {
        "status": "TRACE_SAVED_WITH_EXPORT_ERRORS", "resources_released": True,
        "frame_count": 2, "counts": {"frames_dropped": 2, "trace_errors": 1},
        "video_export_errors": {"top": "TimeoutError"},
    })
    bundle = package_rollout(run, trace_dir=trace)
    missing = validate_bundle(bundle)["missing_data"]
    assert "Missing artifact: metrics.json" in missing
    assert "Original RGB count mismatch: top has 1, expected 2." in missing
    assert "Original RGB frame indices have gaps: top." in missing
    assert "Capture reported frames_dropped=2." in missing
    assert any("Configuration/model-version metadata" in entry for entry in missing)


@pytest.mark.parametrize("source", ["meta", "capture", "unknown"])
def test_explicit_incomplete_log_is_flagged_without_changing_legacy_bundles(recording, source):
    run, trace = recording
    metadata = None
    if source == "meta":
        meta = json.loads((run / "meta.json").read_text())
        meta["log_complete"] = False
        write_json(run / "meta.json", meta)
    elif source == "capture":
        metadata = {"capture": {"log_complete": False}}
    bundle = package_rollout(run, trace_dir=trace, metadata=metadata)
    missing = validate_bundle(bundle)["missing_data"]
    warning = "Session log is incomplete; retained output may be truncated."
    assert (warning in missing) == (source != "unknown")


@pytest.mark.parametrize("relative", ["trace.json", "frames/top/frame-000000.png"])
def test_rejects_artifact_symlinks(recording, relative):
    run, trace = recording
    source = trace / relative
    source.unlink()
    source.symlink_to(run / "log.txt")
    with pytest.raises(ValueError, match="symlinks"):
        package_rollout(run, trace_dir=trace)


def test_rejects_symlink_parent_and_traversal(recording):
    run, trace = recording
    link = run.parent / "linked-trace"
    link.symlink_to(trace, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        package_rollout(run, trace_dir=link)
    with pytest.raises(ValueError, match="parent directories"):
        package_rollout(run, trace_dir=trace / ".." / "trace")


def test_excludes_unrelated_secret_files(recording):
    run, trace = recording
    for directory in (run, trace, trace / "frames/top"):
        (directory / "secret.env").write_text("PASSWORD=must-not-upload")
        (directory / "ignored-symlink").symlink_to(run / "log.txt")
    bundle = package_rollout(run, trace_dir=trace)
    assert not any("secret" in name for name in validate_bundle(bundle)["files"])


def test_known_credentials_redacted_without_recognizable_prefix(recording, monkeypatch):
    run, trace = recording
    monkeypatch.setattr("huggingface_hub.get_token", lambda: "unprefixed-live-credential")
    monkeypatch.setenv("MODAL_TOKEN_ID", "unprefixed-modal-credential")
    write_json(trace / "metrics.json", {"innocent": "unprefixed-live-credential", "another": "unprefixed-modal-credential"})
    bundle = package_rollout(run, trace_dir=trace)
    assert "unprefixed" not in (bundle / "metrics.json").read_text()


@pytest.mark.parametrize("flag", [True, False])
def test_reference_endpoint_boolean_survives_only_at_recorded_schema_paths(flag):
    sample = {"endpoint": flag, "http_endpoint": "http://private/service"}
    value = {"reference_execution": {"dispatch_samples": [sample]},
             "events": [{"kind": "reference_dispatch", **sample}, sample],
             "endpoint": flag, "other": sample}
    clean = sanitize(value)
    assert clean["reference_execution"]["dispatch_samples"] == [{"endpoint": flag}]
    assert clean["events"] == [{"kind": "reference_dispatch", "endpoint": flag}, {}]
    assert "endpoint" not in clean and clean["other"] == {}
    sample["endpoint"] = "https://secret-endpoint/service"
    assert sanitize(value)["reference_execution"]["dispatch_samples"] == [{}]


def test_reference_bundle_reports_committed_join_and_uncommitted_send(recording):
    run, trace = recording
    sample = {"dispatch_index": 0, "chunk_index": 0, "row_index": 0, "point_index": 0, "endpoint": False}
    write_json(trace / "metrics.json", {"reference_execution": {
        "dispatch_samples": [sample], "dispatch_samples_dropped": 1}})
    bundle = package_rollout(run, trace_dir=trace)
    assert json.loads((bundle / "metrics.json").read_text())["reference_execution"]["dispatch_samples"] == [sample]
    missing = validate_bundle(bundle)["missing_data"]
    assert not any("index join was not captured" in entry for entry in missing)
    assert any("omits 1 send(s)" in entry for entry in missing)


@pytest.mark.parametrize("raw, secret", [
    ("Authorization: Bearer arbitrary-credential", "arbitrary-credential"),
    ('{"password": "quoted-password"}', "quoted-password"),
    ("--hf-token=free-form-token", "free-form-token"),
    ("MODAL_TOKEN_ID=free-form-modal-token", "free-form-modal-token"),
    ("https://user:password@private.modal.run/path?secret=yes", "private.modal.run"),
    ("hf_aSecretThatShouldNeverLeave123456", "hf_aSecretThatShouldNeverLeave123456"),
    ("-----BEGIN PRIVATE KEY-----\nsensitive material\n-----END PRIVATE KEY-----", "sensitive material"),
])
def test_secret_values_are_redacted_inside_arbitrary_strings(raw, secret):
    assert secret not in sanitize_text(raw)
    assert secret not in json.dumps(sanitize({"unremarkable_key": raw}))


class FakeHub:
    def __init__(self, *, private=True, missing=False, remote_files=(), remote_manifest=None):
        self.private = private
        self.missing = missing
        self.remote_files = list(remote_files)
        self.remote_manifest = remote_manifest
        self.creates = []
        self.uploads = []
        self.failure = None

    def dataset_info(self, **kwargs):
        if self.missing:
            raise RepositoryNotFoundError("not found", response=httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co")))
        return SimpleNamespace(private=self.private, sha="before-upload")

    def create_repo(self, **kwargs):
        self.creates.append(kwargs)
        self.missing = False

    def list_repo_files(self, **kwargs):
        return self.remote_files

    def hf_hub_download(self, **kwargs):
        destination = Path(kwargs["local_dir"]) / "manifest.json"
        destination.write_bytes(self.remote_manifest)
        return str(destination)

    def upload_folder(self, **kwargs):
        self.uploads.append(kwargs)
        if self.failure:
            raise self.failure
        return SimpleNamespace(oid="completed-commit")


def test_private_create_and_upload_after_finalize_preserves_paths(recording):
    run, trace = recording
    bundle = package_rollout(run, trace_dir=trace)
    api = FakeHub(missing=True)
    result = upload_rollout(bundle, repo_id="owner/rollouts", api=api)
    assert result["status"] == "uploaded"
    assert result["revision"] == "completed-commit"
    assert result["path_in_repo"] == f"runs/{run.name}"
    assert api.creates == [{"repo_id": "owner/rollouts", "repo_type": "dataset", "private": True, "exist_ok": True}]
    assert api.uploads[0]["folder_path"] == bundle
    assert api.uploads[0]["parent_commit"] == "before-upload"
    assert "frames/top/frame-000001.png" in api.uploads[0]["allow_patterns"]
    assert "hf-upload.json" not in api.uploads[0]["allow_patterns"]
    assert json.loads((run / "hf-upload.json").read_text()) == result


def test_public_repo_refused_even_when_created_concurrently(recording):
    run, trace = recording
    api = FakeHub(private=False, missing=True)
    with pytest.raises(ValueError, match="private dataset"):
        upload_rollout(package_rollout(run, trace_dir=trace), repo_id="owner/rollouts", api=api)
    assert not api.uploads
    assert json.loads((run / "hf-upload.json").read_text())["status"] == "failed"


def test_upload_retry_same_manifest_is_idempotent(recording):
    run, trace = recording
    bundle = package_rollout(run, trace_dir=trace)
    manifest = validate_bundle(bundle)
    remote = [f"runs/{run.name}/{name}" for name in manifest["files"]] + [f"runs/{run.name}/manifest.json"]
    api = FakeHub(remote_files=remote, remote_manifest=(bundle / "manifest.json").read_bytes())
    result = upload_rollout(bundle, repo_id="owner/rollouts", api=api)
    assert result["status"] == "already_uploaded"
    assert not api.uploads
    api.remote_manifest = b"different"
    with pytest.raises(ValueError, match="different content"):
        upload_rollout(bundle, repo_id="owner/rollouts", api=api)
    assert not api.uploads


def test_existing_incomplete_remote_run_never_overwritten(recording):
    run, trace = recording
    api = FakeHub(remote_files=[f"runs/{run.name}/top.mp4"])
    with pytest.raises(ValueError, match="different or incomplete"):
        upload_rollout(package_rollout(run, trace_dir=trace), repo_id="owner/rollouts", api=api)
    assert not api.uploads


@pytest.mark.parametrize("change", ["extra", "modified", "symlink", "manifest_path"])
def test_bundle_tampering_refused_before_network(recording, change):
    run, trace = recording
    bundle = package_rollout(run, trace_dir=trace)
    if change == "extra":
        (bundle / "token.env").write_text("secret")
    elif change == "modified":
        (bundle / "trace.json").write_text("different")
    elif change == "symlink":
        (bundle / "top.mp4").unlink()
        (bundle / "top.mp4").symlink_to(trace / "top.mp4")
    else:
        manifest = json.loads((bundle / "manifest.json").read_text())
        manifest["files"]["../log.txt"] = {}
        write_json(bundle / "manifest.json", manifest)
    api = FakeHub()
    with pytest.raises(ValueError):
        upload_rollout(bundle, repo_id="owner/rollouts", api=api)
    assert not api.uploads
    assert not api.creates


def test_failure_status_is_safe_and_retry_keeps_bundle(recording):
    run, trace = recording
    bundle = package_rollout(run, trace_dir=trace)
    api = FakeHub()
    api.failure = RuntimeError("failed at https://private.modal.run token=plain-sensitive-token")
    with pytest.raises(RuntimeError):
        upload_rollout(bundle, repo_id="owner/rollouts", api=api)
    status = (run / "hf-upload.json").read_text()
    assert "plain-sensitive-token" not in status
    assert "private.modal.run" not in status
    assert json.loads(status)["status"] == "failed"
    validate_bundle(bundle)
    api.failure = None
    assert upload_rollout(bundle, repo_id="owner/rollouts", api=api)["status"] == "uploaded"
