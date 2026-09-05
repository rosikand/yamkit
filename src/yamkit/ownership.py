"""Cooperative Linux ownership of a physical CAN adapter before motor activation.

All worktrees use /tmp/yamkit-arm-locks. Files are permanent lock inodes, not stale
PID files: never unlink them. flock is released by close, process death, or exec.
Drivers that do not use these locks can still access the same hardware.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import os
import socket
import stat
import threading
from pathlib import Path

from .can import SYS_NET, find_by_name

LOCK_DIR = Path("/tmp/yamkit-arm-locks")

# Keep leases alive until explicit release, even if a wrapper is garbage collected
# while the SDK's background transmitters are still running.
_leases: set[ArmOwnership] = set()
_registry_lock = threading.RLock()


def adapter_identity(channel: str) -> str:
    """Resolve the *live* adapter, regardless of rig names or serial/name selection."""
    iface = find_by_name(channel)
    if iface is None:
        raise RuntimeError(f"Cannot identify CAN adapter {channel!r} for ownership")
    if iface.serial:
        return f"usb-serial:{iface.serial}"
    device = SYS_NET / channel / "device"
    if device.exists():
        return f"sysfs-device:{device.resolve()}"
    # Virtual CAN has no USB identity. ifindex survives a rename; namespace
    # distinguishes interfaces in separate Linux network namespaces.
    namespace = os.stat("/proc/self/ns/net").st_ino
    return f"netns:{namespace}:ifindex:{socket.if_nametoindex(channel)}"


def _lock_directory() -> Path:
    try:
        LOCK_DIR.mkdir(mode=0o1777)
    except FileExistsError:
        pass
    else:
        # mkdir respects umask; allow other local users to share the same locks.
        LOCK_DIR.chmod(0o1777)
    mode = LOCK_DIR.lstat().st_mode
    if not stat.S_ISDIR(mode) or mode & 0o1777 != 0o1777:
        raise RuntimeError(f"{LOCK_DIR} must be a real shared sticky directory (mode 1777)")
    return LOCK_DIR


class ArmOwnership:
    """One adapter lease, held from before SDK construction until SDK shutdown.

    A fork child may not use the inherited arm. Its copied lock FD is closed
    without LOCK_UN (which would unlock the parent's shared open description).
    The parent's lease remains held, and a child must acquire a fresh lease.
    """

    def __init__(self, identity: str, path: Path, fd: int):
        self.identity = identity
        self.path = path
        self._fd: int | None = fd
        self._pid = os.getpid()

    @classmethod
    def acquire(cls, channel: str) -> ArmOwnership:
        identity = adapter_identity(channel)
        digest = hashlib.sha256(identity.encode()).hexdigest()
        path = _lock_directory() / f"{digest}.lock"
        with _registry_lock:
            # Read-only flock also works on Linux, allowing a different user to
            # acquire the same permanent inode. O_NOFOLLOW rejects symlink paths.
            fd = os.open(path, os.O_CREAT | os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, 0o644)
            try:
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise RuntimeError(f"Ownership lock is not a regular file: {path}")
                # Ensure readers are permitted even with a restrictive umask.
                if os.fstat(fd).st_uid == os.getuid():
                    os.fchmod(fd, 0o644)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    if exc.errno in (errno.EACCES, errno.EAGAIN):
                        raise RuntimeError(
                            f"CAN adapter {channel!r} ({identity}) is already owned by another yamkit "
                            f"arm connection; stop that connection before retrying. Lock: {path}"
                        ) from exc
                    raise
                lease = cls(identity, path, fd)
                _leases.add(lease)
                return lease
            except BaseException:
                os.close(fd)
                raise

    def check_owner(self) -> None:
        if self._pid != os.getpid():
            raise RuntimeError("An inherited arm cannot be used after fork; open it in its owning process")
        if self._fd is None:
            raise RuntimeError("The arm ownership lease has already been released")

    def release(self) -> None:
        """Release only after all hardware transmitters have been stopped."""
        with _registry_lock:
            if self._pid != os.getpid():
                # The fork callback already closed our copy; never unlock parent.
                return
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
            _leases.discard(self)


def _after_fork_child() -> None:
    try:
        for lease in _leases:
            if lease._fd is not None:
                os.close(lease._fd)
                lease._fd = None
        _leases.clear()
    finally:
        _registry_lock.release()


os.register_at_fork(
    before=_registry_lock.acquire,
    after_in_parent=_registry_lock.release,
    after_in_child=_after_fork_child,
)
