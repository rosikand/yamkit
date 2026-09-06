"""Recording wrapper storage and upload boundaries; recorder and Hub are fake."""

import pytest
from typer.testing import CliRunner

from yamkit import cli, hub


@pytest.fixture
def recording(rig, tmp_path, monkeypatch):
    datasets = tmp_path / "datasets"
    root = datasets / "cube"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text('{"total_episodes": 1}')
    rig.hub.username = "tester"
    rig.save()
    monkeypatch.setattr(cli, "DATASETS_DIR", datasets)
    runs, uploads = [], []

    def run(script, args):
        runs.append((script, args))
        return 0

    def upload(name, **kwargs):
        uploads.append((name, kwargs))
        return "https://huggingface.co/datasets/other/target"

    monkeypatch.setattr(cli, "_run_lerobot", run)
    monkeypatch.setattr(hub, "push_dataset", upload)
    args = ["record", "--rig", str(rig.path), "--name", "cube", "--task", "fixture", "--to", "hub"]
    return root, runs, uploads, args


@pytest.mark.parametrize("status", [1, 2, 130, -2, -15])
def test_failed_recorder_preserves_existing_dataset_without_upload(recording, monkeypatch, status):
    root, _, uploads, args = recording
    before = (root / "meta" / "info.json").read_bytes()
    monkeypatch.setattr(cli, "_run_lerobot", lambda *args: status)

    result = CliRunner().invoke(cli.app, [*args, "--resume"])

    assert result.exit_code == (status if status > 0 else 128 - status)
    assert "no upload or deletion" in result.output
    assert not uploads
    assert (root / "meta" / "info.json").read_bytes() == before


@pytest.mark.parametrize("metadata", ['{"total_episodes": 0}', '{}', 'invalid',
                                      '{"total_episodes": -1}', '{"total_episodes": true}'])
@pytest.mark.parametrize("manual", [False, True])
def test_empty_or_invalid_dataset_never_uploads_or_deletes(recording, metadata, manual):
    root, runs, uploads, args = recording
    (root / "meta" / "info.json").write_text(metadata)
    if manual:
        args = ["push-dataset", "cube", "--rig", args[2], "--remove-local"]

    result = CliRunner().invoke(cli.app, args)

    assert result.exit_code == 1
    assert "no upload or deletion" in " ".join(result.output.split())
    assert not uploads
    assert (root / "meta" / "info.json").read_text() == metadata
    assert len(runs) == (0 if manual else 1)


@pytest.mark.parametrize("destination", ["hub", "both"])
@pytest.mark.parametrize("repo", [None, "other/target"])
def test_success_uploads_exact_recorder_root_and_requested_repo(recording, destination, repo):
    root, runs, uploads, args = recording
    args[-1] = destination
    if repo:
        args += ["--repo-id", repo]

    result = CliRunner().invoke(cli.app, args)

    assert result.exit_code == 0, result.output
    assert len(runs) == len(uploads) == 1
    assert f"--dataset.root={root}" in runs[0][1]
    assert f"--dataset.repo_id={repo or 'yamkit/cube'}" in runs[0][1]
    assert uploads == [(repo or "cube", {"private": True, "rig_username": "tester", "root": root})]
    assert root.exists() is (destination == "both")


@pytest.mark.parametrize("extra", [
    ["--dataset.root=other"], ["--dataset.root", "other"],
    ["--dataset.repo_id=other/target"], ["--dataset.repo_id", "other/target"],
    ["--dataset.push_to_hub=true"], ["--dataset.no_stamp=false"], ["--dataset={}"],
])
def test_storage_passthrough_rejected_before_recorder_or_upload(recording, extra):
    root, runs, uploads, args = recording

    result = CliRunner().invoke(cli.app, [*args, *extra])

    assert result.exit_code == 2
    assert "record controls" in result.output
    assert not runs and not uploads and root.exists()


@pytest.mark.parametrize("name", ["", ".", "..", "../outside", "/outside", "nested/cube"])
def test_record_name_cannot_redirect_storage(recording, name):
    root, runs, uploads, args = recording
    args[args.index("--name") + 1] = name

    result = CliRunner().invoke(cli.app, args)

    assert result.exit_code == 2
    assert "dataset name, not a path" in result.output
    assert not runs and not uploads and root.exists()


def test_record_rejects_symlink_dataset(recording):
    root, runs, uploads, args = recording
    root.with_name("link").symlink_to(root, target_is_directory=True)
    args[args.index("--name") + 1] = "link"

    result = CliRunner().invoke(cli.app, args)

    assert result.exit_code == 2
    assert "symlink" in result.output
    assert not runs and not uploads and root.exists()


def test_upload_failure_keeps_recording(recording, monkeypatch):
    root, runs, _, args = recording

    def fail(*args, **kwargs):
        raise ConnectionError("fixture offline")

    monkeypatch.setattr(hub, "push_dataset", fail)
    result = CliRunner().invoke(cli.app, [*args, "--repo-id", "other/target"])

    assert result.exit_code == 2
    assert "upload failed" in result.output
    assert "--repo-id other/target" in " ".join(result.output.split())
    assert len(runs) == 1 and root.exists()


def test_record_passes_encoding_options_through(recording):
    _, runs, _, args = recording

    result = CliRunner().invoke(cli.app, [*args, "--dataset.streaming_encoding=true"])

    assert result.exit_code == 0, result.output
    assert "--dataset.streaming_encoding=true" in runs[0][1]


def test_manual_upload_retry_preserves_explicit_hub_destination(recording):
    root, runs, uploads, args = recording
    result = CliRunner().invoke(cli.app, ["push-dataset", "cube", "--rig", args[2],
                                         "--repo-id", "other/target"])

    assert result.exit_code == 0, result.output
    assert not runs and root.exists()
    assert uploads == [("other/target", {"private": True, "rig_username": "tester", "root": root})]
