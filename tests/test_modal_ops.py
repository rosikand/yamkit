"""Ownership operations are serialized across processes; no tests contact Modal."""

import multiprocessing
from types import SimpleNamespace

import pytest

from yamkit import modal_ops
from yamkit.inference.profiles import get_profile


def ready(profile_id="smolvla"):
    return {**get_profile(profile_id).metadata(), "ready": True, "fresh_chunk": True,
            "saved_processors": True}


@pytest.mark.parametrize("profile_id", ["smolvla", "molmoact2", "pi05"])
def test_readiness_accepts_native_and_robot_unit_profiles(profile_id):
    modal_ops._validate_ready(ready(profile_id), get_profile(profile_id))


@pytest.mark.parametrize("changes", [
    {"ready": False}, {"ready": 1}, {"fresh_chunk": False}, {"saved_processors": False},
    {"model_revision": "main"}, {"repo_id": "other/checkpoint"},
    {"state_names": []}, {"action_names": ["unverified"]},
    {"image_keys": ["camera3", "camera2", "camera1"]}, {"native_image_keys": []},
    {"chunk_size": 51}, {"max_chunk_steps": 0}, {"fps": 100}, {"fps": True},
    {"mapping_verified": True}, {"action_units": "robot"}, {"supports_rtc": True},
])
def test_readiness_rejects_incomplete_or_incompatible_contract(changes):
    with pytest.raises(ValueError):
        modal_ops._validate_ready({**ready(), **changes}, get_profile("smolvla"))


def test_readiness_requires_saved_processor_and_fresh_chunk_flags():
    metadata = ready()
    del metadata["fresh_chunk"]
    with pytest.raises(ValueError, match="fresh_chunk"):
        modal_ops._validate_ready(metadata, get_profile("smolvla"))


def _prepare_in_child(results):
    try:
        receipt = modal_ops.prepare("smolvla", development=True)
        results.put(("ok", receipt["app_id"]))
    except BaseException as error:  # noqa: BLE001 — report any child-process failure to the parent test
        results.put(("error", type(error).__name__))


def test_competing_process_prepares_and_shutdown_cannot_create_second_pool(monkeypatch, tmp_path):
    from yamkit.inference import modal_service

    context = multiprocessing.get_context("fork")
    started, release = context.Event(), context.Event()
    results = context.Queue()
    monkeypatch.setattr(modal_ops, "receipt_path", lambda: tmp_path / "owned-service.json")

    class App:
        app_id = "ap-only-one"

        def __init__(self):
            async def deploy(*, name):
                started.set()
                if not release.wait(5):
                    raise TimeoutError("test did not release the first deployment")
            self.deploy = SimpleNamespace(aio=deploy)

    monkeypatch.setattr(modal_service, "create_app", lambda **kwargs: App())
    monkeypatch.setattr(modal_ops, "service_handle", lambda *args: SimpleNamespace(ready=lambda: ready()))
    monkeypatch.setattr(modal_ops, "call", lambda method, **kwargs: method())
    monkeypatch.setattr(modal_ops.subprocess, "run", lambda *args, **kwargs:
                        pytest.fail("no shutdown may cross another process's ownership operation"))
    process = context.Process(target=_prepare_in_child, args=(results,))
    process.start()
    try:
        assert started.wait(5), "first mocked deployment never began"
        # The second process sees an in-progress receipt, but lock exclusion must
        # happen even before inspecting/reusing it or creating a different pool.
        for profile_id in ("smolvla", "pi05"):
            with pytest.raises(RuntimeError, match="already in progress"):
                modal_ops.prepare(profile_id, development=True)
        with pytest.raises(RuntimeError, match="already in progress"):
            modal_ops.shutdown()
        release.set()
        process.join(5)
        assert not process.is_alive()
        assert results.get(timeout=2) == ("ok", "ap-only-one")
        monkeypatch.setattr(modal_service, "create_app", lambda **kwargs: pytest.fail("second pool created"))
        assert modal_ops.prepare("smolvla")["app_id"] == "ap-only-one"
        with pytest.raises(ValueError, match="different model"):
            modal_ops.prepare("pi05")
    finally:
        if process.is_alive():
            process.terminate()
        process.join(5)
        results.close()


def test_prepare_failure_cleanup_retains_lock_without_reentrant_deadlock(monkeypatch, tmp_path):
    from yamkit.inference import modal_service

    monkeypatch.setattr(modal_ops, "receipt_path", lambda: tmp_path / "owned-service.json")

    def fail(**kwargs):
        with pytest.raises(RuntimeError, match="already in progress"):
            modal_ops.shutdown()
        raise ValueError("mocked image construction failed")

    monkeypatch.setattr(modal_service, "create_app", fail)
    monkeypatch.setattr(modal_ops.subprocess, "run", lambda argv, **kwargs:
                        pytest.fail("deployment was never invoked; no cloud resource exists"))
    with pytest.raises(ValueError, match="mocked image"):
        modal_ops.prepare("smolvla")
    assert modal_ops.owned_service()["status"] == "stopped"
    assert modal_ops.owned_service()["shutdown_verification"] == "deployment was never invoked"
    assert modal_ops.shutdown()["status"] == "stopped"  # lock released despite exception


def test_failed_cleanup_releases_lock_and_blocks_replacement(monkeypatch, tmp_path):
    from yamkit.inference import modal_service

    monkeypatch.setattr(modal_ops, "receipt_path", lambda: tmp_path / "owned-service.json")

    class App:
        app_id = None

        def __init__(self):
            async def fail(*, name):
                raise ValueError("deployment failed after submission")
            self.deploy = SimpleNamespace(aio=fail)

    monkeypatch.setattr(modal_service, "create_app", lambda **kwargs: App())
    monkeypatch.setattr(modal_ops.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=1, stdout=""))
    with pytest.raises(RuntimeError, match="shutdown inventory failed"):
        modal_ops.prepare("smolvla")
    assert modal_ops.owned_service()["status"] == "shutdown_unverified"
    with pytest.raises(ValueError, match="different model"):
        modal_ops.prepare("pi05")


def _hold_lock(started, release):
    with modal_ops._ownership_lock():
        started.set()
        release.wait(5)


def test_process_exit_releases_lock_without_removing_lock_file(monkeypatch, tmp_path):
    context = multiprocessing.get_context("fork")
    started, release = context.Event(), context.Event()
    monkeypatch.setattr(modal_ops, "receipt_path", lambda: tmp_path / "owned-service.json")
    process = context.Process(target=_hold_lock, args=(started, release))
    process.start()
    try:
        assert started.wait(5)
        with pytest.raises(RuntimeError, match="already in progress"):
            modal_ops.shutdown()
        process.terminate()
        process.join(5)
        assert modal_ops.shutdown()["status"] == "stopped"
        assert (tmp_path / "owned-service.lock").exists()
    finally:
        if process.is_alive():
            process.terminate()
        process.join(5)


def _save_incomplete(monkeypatch, tmp_path, **changes):
    monkeypatch.setattr(modal_ops, "receipt_path", lambda: tmp_path / "owned-service.json")
    modal_ops._save({"app_name": "yamkit-vla-owned", "app_id": None, "profile_id": "smolvla",
                     "revision": get_profile("smolvla").revision, "status": "preparing",
                     "deployment_started": True, **changes})


def test_missing_app_id_is_recovered_by_exact_name_before_stop_and_verification(monkeypatch, tmp_path):
    import json

    _save_incomplete(monkeypatch, tmp_path)
    seen = []

    def run(argv, **kwargs):
        args = argv[3:]
        seen.append(args)
        if args == ["app", "list", "--json"]:
            rows = [{"description": "yamkit-vla-owned-copy", "app_id": "ap-unrelated"},
                    {"description": "yamkit-vla-owned", "app_id": "ap-owned"}]
            return SimpleNamespace(returncode=0, stdout=json.dumps(rows))
        return SimpleNamespace(returncode=0, stdout="[]")

    monkeypatch.setattr(modal_ops.subprocess, "run", run)
    receipt = modal_ops.shutdown()
    assert receipt["status"] == "stopped" and receipt["app_id"] == "ap-owned"
    assert receipt["remaining_containers"] == 0
    assert seen == [["app", "list", "--json"], ["app", "stop", "--yes", "ap-owned"],
                    ["container", "list", "--app-id", "ap-owned", "--json"]]


@pytest.mark.parametrize("rows", [[],
    [{"description": "yamkit-vla-owned-copy", "app_id": "ap-other"}],
    [{"description": "yamkit-vla-owned", "app_id": "ap-a"},
     {"description": "yamkit-vla-owned", "app_id": "ap-b"}],
    [{"description": "yamkit-vla-owned", "app_id": "--not-an-id"}],
])
def test_missing_or_ambiguous_app_cannot_be_claimed_stopped(monkeypatch, tmp_path, rows):
    import json

    _save_incomplete(monkeypatch, tmp_path)
    seen = []
    monkeypatch.setattr(modal_ops.subprocess, "run", lambda argv, **kwargs:
                        seen.append(argv[3:]) or SimpleNamespace(returncode=0, stdout=json.dumps(rows)))
    with pytest.raises(RuntimeError, match="unverified"):
        modal_ops.shutdown()
    assert seen == [["app", "list", "--json"]]
    assert modal_ops.owned_service()["status"] == "shutdown_unverified"
    with pytest.raises(ValueError, match="different model"):
        modal_ops.prepare("pi05")


def test_shutdown_timeouts_are_sanitized_and_do_not_mark_stopped(monkeypatch, tmp_path):
    _save_incomplete(monkeypatch, tmp_path, app_id="ap-owned")

    def timeout(*args, **kwargs):
        raise modal_ops.subprocess.TimeoutExpired(["private-data-must-not-surface"], 30,
                                                 output="private-output", stderr="private-stderr")

    monkeypatch.setattr(modal_ops.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="timed out") as error:
        modal_ops.shutdown()
    assert "private" not in str(error.value)
    assert modal_ops.owned_service()["status"] == "shutdown_unverified"


@pytest.mark.parametrize("state,expected", [("stopped", "stopped"), ("deployed", "shutdown_unverified")])
def test_nonzero_stop_needs_control_plane_stopped_state_not_just_idle_containers(monkeypatch, tmp_path,
                                                                                state, expected):
    import json

    _save_incomplete(monkeypatch, tmp_path, app_id="ap-owned")

    def run(argv, **kwargs):
        if argv[3:5] == ["app", "stop"]:
            return SimpleNamespace(returncode=1, stdout="")
        if argv[3:5] == ["app", "list"]:
            return SimpleNamespace(returncode=0, stdout=json.dumps([{"app_id": "ap-owned", "state": state}]))
        return SimpleNamespace(returncode=0, stdout="[]")

    monkeypatch.setattr(modal_ops.subprocess, "run", run)
    if state == "stopped":
        modal_ops.shutdown()
    else:
        with pytest.raises(RuntimeError, match="stop was not confirmed"):
            modal_ops.shutdown()
    assert modal_ops.owned_service()["status"] == expected


def test_remaining_containers_prevent_replacement_after_stop_acknowledged(monkeypatch, tmp_path):
    _save_incomplete(monkeypatch, tmp_path, app_id="ap-owned")
    seen = []
    monkeypatch.setattr(modal_ops.subprocess, "run", lambda argv, **kwargs:
                        seen.append(argv[3:]) or SimpleNamespace(returncode=0, stdout='[{"id": "still-running"}]'))
    monkeypatch.setattr(modal_ops.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="have not retired"):
        modal_ops.shutdown()
    assert len(seen) == 7  # one stop and six bounded retirement polls
    assert modal_ops.owned_service()["status"] == "shutdown_unverified"
