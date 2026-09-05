"""Real Linux process/descriptor lifetime tests; no CAN or motor is opened."""

import os
import select
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

from yamkit import ownership

BOOTSTRAP = """
import os, sys
from pathlib import Path
from yamkit import ownership
ownership.LOCK_DIR = Path(sys.argv[1])
ownership.adapter_identity = lambda channel: 'usb-serial:' + channel
"""

PROBE = """
try:
    lease = ownership.ArmOwnership.acquire(sys.argv[2])
except RuntimeError:
    print('busy', flush=True)
else:
    lease.release()
    print('free', flush=True)
"""


@pytest.fixture
def locks(tmp_path, monkeypatch):
    root = tmp_path / "locks"
    monkeypatch.setattr(ownership, "LOCK_DIR", root)
    monkeypatch.setattr(ownership, "adapter_identity", lambda channel: f"usb-serial:{channel}")
    yield root
    for lease in list(ownership._leases):
        if lease.path.parent == root:
            lease.release()


def _probe(locks, adapter="A", cwd=None):
    result = subprocess.run(
        [sys.executable, "-c", BOOTSTRAP + PROBE, str(locks), adapter],
        cwd=cwd, text=True, capture_output=True, timeout=10, check=True,
    )
    return result.stdout.strip()


def _line(process):
    assert select.select([process.stdout], [], [], 10)[0], "subprocess failed to report readiness"
    line = process.stdout.readline().strip()
    assert line, process.stderr.read()
    return line


def _start(locks, code, cwd=None):
    return subprocess.Popen(
        [sys.executable, "-c", BOOTSTRAP + code, str(locks)],
        cwd=cwd, text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def _stop(process):
    if process.poll() is None:
        process.kill()
    process.communicate(timeout=10)


def test_live_identity_ignores_rig_serial_or_interface_selection(monkeypatch):
    monkeypatch.setattr(ownership, "find_by_name", lambda name: SimpleNamespace(serial="actual-adapter"))
    assert ownership.adapter_identity("can0") == ownership.adapter_identity("renamed0")
    assert ownership.adapter_identity("can0") == "usb-serial:actual-adapter"


def test_no_serial_uses_canonical_physical_device(tmp_path, monkeypatch):
    device = tmp_path / "devices" / "usb-port" / "interface"
    device.mkdir(parents=True)
    for name in ("can0", "renamed0"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "device").symlink_to(device, target_is_directory=True)
    monkeypatch.setattr(ownership, "SYS_NET", tmp_path)
    monkeypatch.setattr(ownership, "find_by_name", lambda name: SimpleNamespace(serial=None))
    assert ownership.adapter_identity("can0") == ownership.adapter_identity("renamed0")


def test_virtual_interface_uses_namespace_and_ifindex(tmp_path, monkeypatch):
    monkeypatch.setattr(ownership, "SYS_NET", tmp_path)
    monkeypatch.setattr(ownership, "find_by_name", lambda name: SimpleNamespace(serial=None))
    monkeypatch.setattr(ownership.socket, "if_nametoindex", lambda name: 42)
    assert ownership.adapter_identity("vcan0").endswith(":ifindex:42")


def test_missing_adapter_fails_without_creating_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(ownership, "LOCK_DIR", tmp_path / "unused")
    monkeypatch.setattr(ownership, "find_by_name", lambda name: None)
    with pytest.raises(RuntimeError, match="Cannot identify"):
        ownership.ArmOwnership.acquire("absent")
    assert not (tmp_path / "unused").exists()


def test_duplicate_connection_conflicts_even_in_same_process(locks):
    lease = ownership.ArmOwnership.acquire("A")
    with pytest.raises(RuntimeError, match="already owned"):
        ownership.ArmOwnership.acquire("A")
    assert not os.get_inheritable(lease._fd)
    lease.release()
    lease.release()
    with pytest.raises(RuntimeError, match="released"):
        lease.check_owner()
    assert lease.path.exists()  # permanent inode, not an unlinkable stale PID marker
    assert _probe(locks) == "free"


def test_worktrees_and_overlapping_rigs_conflict(locks, tmp_path):
    first, second = tmp_path / "worktree-a", tmp_path / "worktree-b"
    first.mkdir()
    second.mkdir()
    process = _start(locks, "lease = ownership.ArmOwnership.acquire('A')\nprint('ready', flush=True)\ninput()\nlease.release()\n", cwd=first)
    try:
        assert _line(process) == "ready"
        assert _probe(locks, cwd=second) == "busy"
        assert _probe(locks, "B", cwd=second) == "free"
        process.communicate("close\n", timeout=10)
        assert process.returncode == 0
        assert _probe(locks, cwd=second) == "free"
    finally:
        _stop(process)


def test_crash_releases_all_arm_locks(locks):
    process = _start(locks, "a = ownership.ArmOwnership.acquire('A')\nb = ownership.ArmOwnership.acquire('B')\nprint('ready', flush=True)\ninput()\n")
    try:
        assert _line(process) == "ready"
        assert _probe(locks, "A") == _probe(locks, "B") == "busy"
        process.send_signal(signal.SIGKILL)
        process.wait(timeout=10)
        assert _probe(locks, "A") == _probe(locks, "B") == "free"
    finally:
        _stop(process)


def test_exec_releases_lease_when_hardware_process_is_replaced(locks):
    process = _start(locks, """
lease = ownership.ArmOwnership.acquire('A')
print('ready', flush=True)
input()
os.execv(sys.executable, [sys.executable, '-c', "print('exec-ready', flush=True); input()"])
""")
    try:
        assert _line(process) == "ready"
        assert _probe(locks) == "busy"
        process.stdin.write("exec\n")
        process.stdin.flush()
        assert _line(process) == "exec-ready"
        assert process.poll() is None
        assert _probe(locks) == "free"
    finally:
        _stop(process)


def test_fork_child_cannot_release_parent_or_extend_ownership(locks):
    code = """
import subprocess
lease = ownership.ArmOwnership.acquire('A')
ready_read, ready_write = os.pipe()
finish_read, finish_write = os.pipe()
pid = os.fork()
if pid == 0:
    os.close(ready_read)
    os.close(finish_write)
    try:
        lease.check_owner()
    except RuntimeError:
        lease.release()  # must never unlock parent's shared open description
        os.write(ready_write, b'ok')
    os.read(finish_read, 1)
    os._exit(0)
os.close(ready_write)
os.close(finish_read)
assert os.read(ready_read, 2) == b'ok'
print('child-ready', flush=True)
input()
lease.release()
print('parent-released-child-alive', flush=True)
input()
os.write(finish_write, b'x')
assert os.waitpid(pid, 0)[1] == 0
"""
    process = _start(locks, code)
    try:
        assert _line(process) == "child-ready"
        assert _probe(locks) == "busy"
        process.stdin.write("release\n")
        process.stdin.flush()
        assert _line(process) == "parent-released-child-alive"
        assert _probe(locks) == "free"  # child is alive but has no inherited FD
        process.communicate("finish\n", timeout=10)
        assert process.returncode == 0
    finally:
        _stop(process)


def test_partial_multi_arm_acquisition_releases_only_new_leases(locks):
    first = ownership.ArmOwnership.acquire("B")
    process = _start(locks, """
from contextlib import ExitStack
try:
    with ExitStack() as stack:
        for name in ('A', 'B'):
            lease = ownership.ArmOwnership.acquire(name)
            stack.callback(lease.release)
except RuntimeError:
    print('conflict-cleaned', flush=True)
input()
""")
    try:
        assert _line(process) == "conflict-cleaned"
        assert _probe(locks, "A") == "free"
        assert _probe(locks, "B") == "busy"
        first.check_owner()
    finally:
        _stop(process)


def test_dropping_reference_keeps_protection_until_explicit_shutdown(locks):
    import gc

    ownership.ArmOwnership.acquire("A")
    gc.collect()
    assert _probe(locks) == "busy"


def test_symlink_directory_is_rejected(locks, tmp_path):
    target = tmp_path / "other"
    target.mkdir(mode=0o777)
    locks.symlink_to(target, target_is_directory=True)
    with pytest.raises(RuntimeError, match="real shared sticky directory"):
        ownership.ArmOwnership.acquire("A")


def test_symlink_lock_file_is_rejected(locks, tmp_path):
    lease = ownership.ArmOwnership.acquire("A")
    path = lease.path
    lease.release()
    path.unlink()  # no live owners; set up the hostile path
    target = tmp_path / "other"
    target.touch()
    path.symlink_to(target)
    with pytest.raises(OSError):
        ownership.ArmOwnership.acquire("A")
