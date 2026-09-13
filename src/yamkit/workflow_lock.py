"""Cross-process exclusion for software model preparation and managed Start."""

from __future__ import annotations

import fcntl
import os
import stat
import time
from contextlib import contextmanager
from contextvars import ContextVar

from .backend_workflow import WorkflowError
from .paths import ROOT

_HELD = ContextVar("yamkit_workflow_locks", default=())
INHERITED_FD_ENV = "YAMKIT_INFERENCE_WORKFLOW_FD"


def _path(root):
    return root / ".context/inference-workflow.lock"


def _open(root, *, create):
    path = _path(root)
    if path.parent.resolve() != path.parent.absolute():
        raise WorkflowError("Inference preparation lock requires repository-local directories without symlinks")
    if create:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.parent.resolve() != path.parent.absolute():
        raise WorkflowError("Inference preparation lock requires repository-local directories without symlinks")
    fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK | (os.O_CREAT if create else 0), 0o600)
    details = os.fstat(fd)
    if not stat.S_ISREG(details.st_mode) or details.st_uid != os.geteuid() or details.st_mode & 0o077:
        os.close(fd)
        raise WorkflowError("Inference preparation lock requires a private owned regular file")
    return fd


def _inherited_fd(root):
    raw = os.environ.get(INHERITED_FD_ENV)
    if raw is None:
        return None
    try:
        fd = int(raw)
        details, expected = os.fstat(fd), _path(root).stat(follow_symlinks=False)
        if (fd < 3 or (details.st_dev, details.st_ino) != (expected.st_dev, expected.st_ino)
                or not stat.S_ISREG(details.st_mode) or details.st_uid != os.geteuid() or details.st_mode & 0o077):
            raise ValueError
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except (ValueError, OSError):
        raise WorkflowError("Inherited inference ownership is invalid; no new inference operation was started") from None


@contextmanager
def workflow_lock(*, root=None, wait_s=0):
    """Reentrant in one call context; mutually exclusive across CLI and UI children."""
    root = ROOT if root is None else root
    key = str(_path(root))
    for held_key, held_fd in _HELD.get():
        if key == held_key:
            yield held_fd
            return
    inherited = _inherited_fd(root)
    fd = inherited if inherited is not None else _open(root, create=True)
    deadline = time.monotonic() + wait_s
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise WorkflowError("Another CLI or UI inference operation is preparing or running; wait for it to finish") from None
                time.sleep(0.05)
        token = _HELD.set((*_HELD.get(), (key, fd)))
        try:
            yield fd
        finally:
            _HELD.reset(token)
    finally:
        if inherited is None:
            os.close(fd)


def assert_workflow_available(*, root=None):
    """Read-only availability check for UI preflight; launch must acquire atomically."""
    root = ROOT if root is None else root
    try:
        fd = _open(root, create=False)
    except FileNotFoundError:
        return
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise WorkflowError("Another CLI or UI inference operation is preparing or running; wait for it to finish") from None
    finally:
        os.close(fd)
