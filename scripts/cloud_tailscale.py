#!/usr/bin/env python3
"""A repository-local, ephemeral Tailscale client for Conductor Cloud.

Uses only Python's standard library; never installs a service or changes the host
network. Credentials come from Conductor's secret environment variables.
"""

import contextlib
import errno
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request


VERSION = "1.102.3"
SECRET_ENV = "YAMKIT_TAILSCALE_OAUTH_SECRET"
TAGS_ENV = "YAMKIT_TAILSCALE_TAGS"
ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / ".tools" / "tailscale"


class SetupError(Exception):
    """An actionable setup failure whose message contains no credentials."""


def cloud_environment(environ):
    """Return False on a Mac workspace; reject ambiguous/non-cloud execution."""
    if environ.get("CONDUCTOR_IS_LOCAL") == "1":
        return False
    if environ.get("CONDUCTOR_IS_LOCAL") != "0" or platform.system() != "Linux":
        raise SetupError("This helper requires a Linux Conductor Cloud workspace (CONDUCTOR_IS_LOCAL=0).")
    return True


def workspace_hostname(environ, root=ROOT):
    identity = environ.get("CONDUCTOR_WORKSPACE_ID") or f"{socket.gethostname()}:{root.resolve()}"
    suffix = hashlib.sha256(identity.encode()).hexdigest()[:12]
    return "conductor-yamkit-cloud-" + suffix


def local_environment(base, environ):
    # Do not inherit Conductor secrets into the long-lived daemon or SSH command.
    names = ("PATH", "LANG", "LC_ALL", "TERM", "SSL_CERT_FILE", "SSL_CERT_DIR",
             "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
             "http_proxy", "https_proxy", "all_proxy", "no_proxy")
    env = {name: environ[name] for name in names if name in environ}
    env.update({
        "XDG_CONFIG_HOME": str(base / "config"),
        "XDG_CACHE_HOME": str(base / "cache"),
        "XDG_DATA_HOME": str(base / "data"),
        "TMPDIR": str(base / "tmp"),
        "TS_LOGS_DIR": str(base / "state"),
        # Vercel's link-local Internet route is otherwise mistaken for offline.
        "TS_ASSUME_NETWORK_UP_FOR_TEST": "true",
        "TS_FORCE_NOISE_443": "true",
    })
    return env


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def architecture():
    arches = {"x86_64": "amd64", "aarch64": "arm64", "arm64": "arm64"}
    try:
        return arches[platform.machine()]
    except KeyError:
        raise SetupError("Tailscale bootstrap supports Linux amd64 and arm64 only.") from None


def installed(base, arch):
    try:
        manifest = json.loads((base / "installed.json").read_text())
        return (manifest["version"] == VERSION and manifest["arch"] == arch
                and all(os.access(base / name, os.X_OK)
                        and sha256(base / name) == manifest["sha256"][name]
                        for name in ("tailscale", "tailscaled")))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def install(base):
    """Download pinned official static binaries and verify the published checksum."""
    arch = architecture()
    if installed(base, arch):
        return
    prefix = f"tailscale_{VERSION}_{arch}"
    url = f"https://pkgs.tailscale.com/stable/{prefix}.tgz"
    try:
        with tempfile.TemporaryDirectory(prefix="download-", dir=base / "tmp") as temp:
            temp = Path(temp)
            archive = temp / "tailscale.tgz"
            with urllib.request.urlopen(url + ".sha256", timeout=45) as response:
                fields = response.read(4096).decode().split()
            if not fields or not re.fullmatch(r"[0-9a-fA-F]{64}", fields[0]):
                raise SetupError("The Tailscale download checksum has an unexpected format.")
            checksum = fields[0]
            with urllib.request.urlopen(url, timeout=45) as source, archive.open("wb") as target:
                shutil.copyfileobj(source, target)
            if sha256(archive) != checksum.lower():
                raise SetupError("Tailscale archive checksum mismatch; no binaries were installed.")
            hashes = {}
            with tarfile.open(archive, "r:gz") as package:
                for name in ("tailscale", "tailscaled"):
                    member = package.getmember(f"{prefix}/{name}")
                    if not member.isfile() or member.size > 256 * 1024 * 1024:
                        raise SetupError("The Tailscale archive contains an unexpected binary entry.")
                    with package.extractfile(member) as source, (temp / name).open("wb") as target:
                        shutil.copyfileobj(source, target)
                    (temp / name).chmod(0o700)
                    hashes[name] = sha256(temp / name)
            for name in hashes:
                os.replace(temp / name, base / name)
            manifest = {"version": VERSION, "arch": arch, "sha256": hashes}
            (temp / "installed.json").write_text(json.dumps(manifest) + "\n")
            os.replace(temp / "installed.json", base / "installed.json")
    except (OSError, ValueError, KeyError, tarfile.TarError):
        raise SetupError("Could not download or unpack Tailscale; "
                         "check this workspace's Internet access.") from None


def command(base):
    return [str(base / "tailscale"), "--socket=" + str(base / "tailscaled.sock")]


def status(base, env):
    if not (base / "tailscaled.sock").exists():
        return None
    try:
        result = subprocess.run(command(base) + ["status", "--json"], env=env, cwd=base,
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            current = json.loads(result.stdout)
            if isinstance(current, dict):
                return current
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return None


def process_identity(pid):
    """Linux start time protects against a recycled PID in our saved PID file."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return fields[19]
    except (OSError, IndexError):
        return None


def owned_process(base):
    try:
        record = json.loads((base / "daemon.pid").read_text())
        pid = int(record["pid"])
        if pid <= 1 or process_identity(pid) != record["start"]:
            return None
        args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        expected = [str(base / "tailscaled"), "--socket=" + str(base / "tailscaled.sock"), "--state=mem:"]
        if all(arg.encode() in args for arg in expected):
            return pid
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def clear_stale_daemon(base):
    """Stop only the daemon recorded by this helper; never unlink a live foreign socket."""
    pid = owned_process(base)
    if pid:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            if owned_process(base) != pid:
                break
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, sig)
            for _ in range(30):
                if owned_process(base) != pid:
                    break
                time.sleep(0.1)
        if owned_process(base) == pid:
            raise SetupError("The workspace Tailscale daemon did not stop; "
                             "inspect .tools/tailscale/daemon.log.")
    sock = base / "tailscaled.sock"
    if sock.exists():
        with socket.socket(socket.AF_UNIX) as probe:
            probe.settimeout(1)
            try:
                probe.connect(str(sock))
            except OSError as exc:
                if exc.errno not in (errno.ECONNREFUSED, errno.ENOENT):
                    raise SetupError("Cannot safely replace the workspace Tailscale socket.") from None
            else:
                raise SetupError("An unrecognized daemon owns the workspace socket; it was left running.")
        sock.unlink(missing_ok=True)
    (base / "daemon.pid").unlink(missing_ok=True)


def start_daemon(base, env):
    clear_stale_daemon(base)
    args = [str(base / "tailscaled"), "--tun=userspace-networking", "--state=mem:",
            "--statedir=" + str(base / "state"), "--socket=" + str(base / "tailscaled.sock"),
            "--port=0", "--no-logs-no-support"]
    with (base / "daemon.log").open("ab") as log:
        process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                   start_new_session=True, cwd=base, env=env)
    record = {"pid": process.pid, "start": process_identity(process.pid)}
    (base / "daemon.pid").write_text(json.dumps(record) + "\n")
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SetupError("Tailscale daemon exited; inspect .tools/tailscale/daemon.log.")
        current = status(base, env)
        if current is not None:
            return current
        time.sleep(0.1)
    raise SetupError("Tailscale daemon did not become ready; inspect .tools/tailscale/daemon.log.")


def credentials(environ):
    secret = environ.get(SECRET_ENV, "").strip()
    if not secret:
        raise SetupError(f"Set the secret {SECRET_ENV} in Conductor Cloud environment settings; "
                         "see docs/CONDUCTOR_CLOUD.md. No interactive login is used.")
    if not re.fullmatch(r"tskey-client-[A-Za-z0-9_-]+", secret):
        raise SetupError(f"{SECRET_ENV} must contain the OAuth client secret (tskey-client-...), "
                         "without a URL or extra options.")
    tags = environ.get(TAGS_ENV, "tag:conductor-yamkit")
    if not re.fullmatch(r"tag:[a-zA-Z][a-zA-Z0-9-]*(,tag:[a-zA-Z][a-zA-Z0-9-]*)*", tags):
        raise SetupError(f"{TAGS_ENV} must be a comma-separated list of tag:name values.")
    return secret, tags


def enroll(base, env, environ, secret, tags):
    # Tailscale accepts OAuth secrets through its file: auth-key syntax. The CLI
    # exchanges this for a fresh ephemeral node key, then discards the secret.
    fd, filename = tempfile.mkstemp(prefix="oauth-", dir=base / "tmp")
    try:
        with os.fdopen(fd, "w") as target:
            target.write(secret + "?ephemeral=true&preauthorized=true")
        args = command(base) + ["up", "--reset", "--auth-key=file:" + filename,
                                "--advertise-tags=" + tags,
                                "--hostname=" + workspace_hostname(environ),
                                "--accept-dns=false", "--accept-routes=false",
                                "--ssh=false", "--shields-up", "--timeout=45s"]
        try:
            result = subprocess.run(args, env=env, cwd=base, capture_output=True, timeout=55)
        except subprocess.TimeoutExpired:
            raise SetupError("Tailscale enrollment timed out; "
                             "check cloud Internet access and tailnet policy.") from None
        if result.returncode:
            # Authentication errors may echo OAuth URLs or tokens. Never forward
            # their output to terminal logs, exceptions, or the daemon log.
            raise SetupError("Tailscale enrollment failed. Check the Conductor OAuth secret, its "
                             "auth_keys write scope and allowed tags, and tailnet device approval.")
    finally:
        Path(filename).unlink(missing_ok=True)


def ensure_online(base=BASE, environ=None):
    environ = os.environ if environ is None else environ
    if not cloud_environment(environ):
        return False
    # UNIX domain sockets have a short path limit; never fall back to /tmp.
    if len(os.fsencode(base / "tailscaled.sock")) >= 104:
        raise SetupError("Workspace path is too long for a repository-local Tailscale socket.")
    os.umask(0o077)
    for folder in (base, *(base / name for name in ("config", "cache", "data", "tmp", "state"))):
        folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    env = local_environment(base, environ)
    # Import after the Mac guard: no platform-specific actions in local setup.
    import fcntl
    with (base / "setup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = status(base, env)
        if current and current.get("BackendState") == "Running":
            return True
        secret, tags = credentials(environ)
        install(base)
        if current is None:
            current = start_daemon(base, env)
        enroll(base, env, environ, secret, tags)
        current = status(base, env)
        if not current or current.get("BackendState") != "Running":
            raise SetupError("Tailscale has not reached Running; check tailnet device approval and policy.")
    return True


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in ("setup", "run"):
        print("Usage: python3 scripts/cloud_tailscale.py setup | run <tailscale arguments>", file=sys.stderr)
        return 2
    try:
        online = ensure_online()
        if not online:
            if argv[0] == "setup":
                return 0
            raise SetupError("scripts/tailscale is only for Conductor Cloud workspaces.")
        if argv[0] == "setup":
            print("Conductor Cloud Tailscale ready: " + workspace_hostname(os.environ))
            return 0
        env = local_environment(BASE, os.environ)
        # exec preserves SSH's streams, signals, and remote exit code.
        os.execve(str(BASE / "tailscale"), command(BASE) + argv[1:], env)
    except SetupError as exc:
        print(f"Cloud Tailscale: {exc}", file=sys.stderr)
        return 1
    except OSError:
        print("Cloud Tailscale: a local file or process operation failed; "
              "check .tools/tailscale permissions.",
              file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
