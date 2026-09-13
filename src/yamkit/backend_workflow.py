"""Configured existing-GPU lifecycle, independent of policy execution.

No robot/camera imports, provisioning, SSH configuration edits, or process takeovers.
An existing listener is authenticated before reuse; an unknown listener is never killed.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from .paths import ROOT

CONFIG_RELATIVE = "data/inference/backends.json"


class WorkflowError(ValueError):
    """An actionable operator error, safe to display without a traceback."""


def _name(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", value):
        raise WorkflowError(f"{label} must be a short lowercase name")
    return value


def _local_path(value, *, root=None):
    root = ROOT if root is None else root
    if not isinstance(value, str) or not value or "\x00" in value:
        raise WorkflowError("Backend file paths must be nonempty repository-local paths")
    original = Path(value) if Path(value).is_absolute() else Path(root) / value
    resolved = original.resolve()
    if not resolved.is_relative_to(Path(root).resolve()) or original.absolute() != resolved:
        raise WorkflowError("Backend file paths must stay inside this checkout without symlinks")
    return resolved


@dataclass(frozen=True)
class BackendTarget:
    backend: str
    policy: str
    service: str
    endpoint: str
    token_file: Path | None = None
    ssh: dict | None = None
    remote: dict | None = None
    saved_observations: tuple[Path, ...] = ()

    @property
    def port(self):
        return int(self.endpoint.rsplit(":", 1)[1])


def load_target(backend: str, policy: str, *, config: Path | None = None) -> BackendTarget:
    """Resolve only local metadata; never reads token or private-key contents."""
    from .external_ops import _read_json, owned_service
    from .inference.http_transport import validate_endpoint_url
    from .policy_selection import canonical_policy

    _name(backend, "Backend")
    policy = _name(canonical_policy(policy), "Policy")
    path = _local_path(str(config or (ROOT / CONFIG_RELATIVE)))
    if not path.exists():
        # Existing installations need no new configuration to reuse a single attachment.
        candidates = []
        directory = ROOT / "data/inference/external"
        for receipt_path in directory.glob("*/receipt.json"):
            try:
                receipt = owned_service(receipt_path.parent.name)
                if (receipt and receipt.get("status") == "ready" and receipt.get("provider") == backend
                        and receipt.get("profile_id") == policy):
                    candidates.append(receipt)
            except ValueError:
                continue
        if len(candidates) == 1:
            receipt = candidates[0]
            return BackendTarget(backend, policy, receipt["name"], receipt["http_endpoint"])
        raise WorkflowError(f"Configure {CONFIG_RELATIVE} for backend {backend}, policy {policy}; "
                            "see docs/INFERENCE_CLI.md (no VM is created)")
    try:
        document = _read_json(path)
    except ValueError:
        raise WorkflowError(f"{CONFIG_RELATIVE} must be a private owned JSON file (chmod 600)") from None
    if set(document) != {"version", "backends"} or document["version"] != 1 or not isinstance(document["backends"], dict):
        raise WorkflowError("Backend configuration requires version 1 and a backends mapping")
    selected = document["backends"].get(backend)
    if not isinstance(selected, dict) or set(selected) - {"policies", "ssh"}:
        raise WorkflowError(f"Configure the policies and optional ssh mapping for backend {backend}")
    policies = selected.get("policies")
    entry = policies.get(policy) if isinstance(policies, dict) else None
    if not isinstance(entry, dict) or set(entry) - {"service", "endpoint", "token_file", "remote", "saved_observations"}:
        raise WorkflowError(f"Backend {backend} has no valid {policy} policy configuration")
    service = _name(entry.get("service"), "Service")
    try:
        endpoint = validate_endpoint_url(entry.get("endpoint"), http_ingress="ssh")
    except (ValueError, TypeError):
        raise WorkflowError("Backend endpoint must be http://127.0.0.1:PORT") from None
    for other_name, other in policies.items():
        if other_name != policy and isinstance(other, dict) and (
                other.get("service") == service or other.get("endpoint") == endpoint):
            raise WorkflowError("Each configured policy needs its own service name and loopback port; "
                                "different models cannot share a service identity or listener")
    token_file = _local_path(entry["token_file"]) if entry.get("token_file") else None
    ssh = selected.get("ssh")
    if ssh is not None:
        if not isinstance(ssh, dict) or set(ssh) - {"host", "identity_file", "known_hosts_file"}:
            raise WorkflowError("ssh accepts host (an SSH config alias or user@host), identity_file and known_hosts_file")
        host = ssh.get("host")
        if not isinstance(host, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@:-]{0,253}", host):
            raise WorkflowError("Set a bounded SSH host alias or user@host; shell syntax is not allowed")
        ssh = dict(ssh)
        for key in ("identity_file", "known_hosts_file"):
            if key in ssh:
                ssh[key] = str(_local_path(ssh[key]))
    remote = entry.get("remote")
    if remote is not None:
        allowed = {"repo", "token_file", "region", "session_seconds", "gpu"}
        if not ssh or not isinstance(remote, dict) or set(remote) - allowed:
            raise WorkflowError("Remote startup requires ssh and repo/token_file/region settings")
        remote = dict(remote)
        repo, token = remote.get("repo"), remote.get("token_file")
        if (not isinstance(repo, str) or not repo.startswith("/") or "\x00" in repo
                or not isinstance(token, str) or not token or Path(token).is_absolute()
                or ".." in Path(token).parts or "\x00" in token):
            raise WorkflowError("Remote repo must be absolute; remote token_file must be inside that repo")
        if not isinstance(remote.get("region"), str) or not re.fullmatch(r"[A-Za-z0-9 _-]{1,80}", remote["region"]):
            raise WorkflowError("Declare the existing GPU region in remote.region")
        seconds, gpu = remote.get("session_seconds", 28800), remote.get("gpu", 0)
        if type(seconds) is not int or not 300 <= seconds <= 86400 or type(gpu) is not int or not 0 <= gpu <= 15:
            raise WorkflowError("Remote session_seconds must be 300–86400; gpu must be an integer 0–15")
        remote.update(session_seconds=seconds, gpu=gpu)
    saved = entry.get("saved_observations", [])
    if not isinstance(saved, list) or len(saved) > 50:
        raise WorkflowError("saved_observations must list at most 50 repository-local NPZ paths")
    return BackendTarget(backend, policy, service, endpoint, token_file, ssh, remote,
                         tuple(_local_path(value) for value in saved))


def assert_ui_idle(*, endpoint="http://127.0.0.1:8400", own_preparation_dir=None, require_cameras_idle=False):
    """Do not disturb a user-owned UI job, including saving or prompt preparation."""
    try:
        with urlopen(endpoint + "/api/session", timeout=2) as response:
            raw = response.read(262145)
        if len(raw) > 262144:
            raise WorkflowError("UI session response is invalid; inspect the UI before inference preparation")
        state = json.loads(raw)
        if not isinstance(state, dict) or type(state.get("active")) is not bool:
            raise WorkflowError("Port 8400 did not provide yamkit session status; inspect that listener")
        own = (own_preparation_dir is not None and state["active"] and state.get("pid") == os.getpid()
               and state.get("mode") == "inference-prepare" and not state.get("cameras_owned")
               and state.get("meta", {}).get("preparation_dir") == str(own_preparation_dir))
        if not own and (state["active"] or state.get("cameras_owned")):
            raise WorkflowError("A UI session owns the robot or is still preparing/saving; wait for it to finish")
        if require_cameras_idle and not own and state.get("direct_cameras_open"):
            raise WorkflowError("The UI has direct camera previews open; close Live/camera previews or use UI Start for its camera handoff")
    except URLError as exc:
        if isinstance(exc.reason, ConnectionRefusedError):
            return  # A terminal-only installation need not run the UI.
        raise WorkflowError("Cannot verify the existing UI is idle; check its status before preparing inference") from None
    except WorkflowError:
        raise
    except (TimeoutError, OSError, ValueError, UnicodeError):
        raise WorkflowError("Cannot verify the existing UI is idle; check its status before preparing inference") from None


def ssh_args(target: BackendTarget) -> list[str]:
    if not target.ssh:
        raise WorkflowError("This backend has no SSH connection configured")
    args = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=15"]
    if target.ssh.get("identity_file"):
        args += ["-i", target.ssh["identity_file"]]
    if target.ssh.get("known_hosts_file"):
        args += ["-o", "UserKnownHostsFile=" + target.ssh["known_hosts_file"]]
    return args


def _ssh(target, arguments, *, timeout=25):
    try:
        result = subprocess.run([*ssh_args(target), *arguments], stdin=subprocess.DEVNULL,
                                capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        raise WorkflowError("SSH did not finish; check the configured SSH alias, agent/key and verified host key") from None
    if result.returncode:
        # SSH diagnostics may reflect user configuration; never echo arbitrary stderr.
        raise WorkflowError(f"SSH exited {result.returncode}; check the configured SSH alias, agent/key and verified host key. "
                            "Unknown host keys are never accepted automatically")
    return result.stdout


def _listening(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def _managed_forward_present(target):
    """Recognize only our UID's exact ssh destination and loopback forward argv.

    Read command paths only after matching listener ownership; never inspect or
    display unrelated process command lines. Unknown ownership is a hard stop.
    """
    if not target.ssh:
        return False
    try:
        rows = Path("/proc/net/tcp").read_text().splitlines()[1:]
        address = f"0100007F:{target.port:04X}"
        inodes = {row.split()[9] for row in rows if row.split()[1] == address
                  and row.split()[3] == "0A" and int(row.split()[7]) == os.geteuid()}
        if not inodes:
            return False
        forward = f"127.0.0.1:{target.port}:127.0.0.1:{target.port}"
        for proc in Path("/proc").iterdir():
            if not proc.name.isdecimal():
                continue
            try:
                if proc.stat().st_uid != os.geteuid() or (proc / "comm").read_text().strip() != "ssh":
                    continue
                if not any(fd.readlink().as_posix() in {f"socket:[{inode}]" for inode in inodes}
                           for fd in (proc / "fd").iterdir()):
                    continue
                args = (proc / "cmdline").read_bytes().decode().split("\x00")
                forwards = [args[i + 1] for i, arg in enumerate(args[:-1]) if arg == "-L"]
                forwards += [arg[2:] for arg in args if arg.startswith("-L") and arg != "-L"]
                if forward in forwards and target.ssh["host"] in args:
                    return True
            except (OSError, UnicodeError):
                continue
    except (OSError, ValueError, IndexError):
        pass
    return False


def _probe_service(target, token):
    from .inference.http_transport import HttpTransport

    transport = HttpTransport(target.service, target.policy, endpoint_url=target.endpoint, token=token,
                              http_ingress="ssh", http_session_expires_at=time.time() + 30)
    try:
        return transport._invoke("ready", None, 5)
    finally:
        transport.close()


# This bootstrap uses only stdlib filesystem/process/socket operations on the existing GPU.
# The inherited flock stays held by the detached supervisor, including model loading.
# It cannot stop an existing process, overwrite source, copy tokens, or claim an occupied port.
_REMOTE_BOOTSTRAP = '''import fcntl,json,os,socket,stat,subprocess,sys
from pathlib import Path
c=json.loads(sys.argv[1]); root=Path.cwd().resolve(); directory=root/'data/inference/managed'/c['service']
for parent in [root/'data',root/'data/inference',root/'data/inference/managed',directory]:
 parent.mkdir(mode=0o700,exist_ok=True)
 if parent.is_symlink() or not parent.is_dir(): raise ValueError('Unsafe managed-service directory')
fd=os.open(directory/'service.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
details=os.fstat(fd)
if not stat.S_ISREG(details.st_mode) or details.st_uid!=os.geteuid() or details.st_mode&0o077: raise ValueError('Unsafe service lock')
try: fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:
 print(json.dumps({'status':'starting_or_running'})); sys.exit(0)
sock=socket.socket()
try: sock.bind(('127.0.0.1',c['port']))
except OSError:
 print(json.dumps({'status':'listener_present'})); sys.exit(0)
finally: sock.close()
token=root/c['token_file']
if token.resolve()!=token or not token.is_file(): raise ValueError('Existing repo-local token required')
logfd=os.open(directory/'service.log',os.O_WRONLY|os.O_CREAT|os.O_APPEND|os.O_NOFOLLOW,0o600)
env=dict(os.environ); env['CUDA_VISIBLE_DEVICES']=str(c['gpu'])
module='yamkit.pi05.service' if c['policy']=='pi05-yam' else 'yamkit.inference.standalone_service'
command=[str(root/'.venv-inference/bin/python'),'-m',module,
 '--service-id',c['service'],'--region',c['region'],'--port',str(c['port']),
 '--token-file',c['token_file'],'--session-seconds',str(c['session_seconds']),'--task',c['task']]
if c['policy']=='molmoact2': command+=['--provider','lambda']
child=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=logfd,stderr=logfd,
 start_new_session=True,pass_fds=(fd,))
print(json.dumps({'status':'started','pid':child.pid}))
'''


def _start_remote(target, task):
    if target.policy not in ("molmoact2", "pi05-yam"):
        raise WorkflowError("Automatic startup for this policy requires its separate native runtime; no MolmoAct2 substitution is allowed")
    values = {**target.remote, "service": target.service, "port": target.port, "task": task, "policy": target.policy}
    repo = values.pop("repo")
    command = ("cd " + shlex.quote(repo) + " && . data/inference/env.sh && "
               + shlex.join([".venv-inference/bin/python", "-c", _REMOTE_BOOTSTRAP,
                              json.dumps(values, allow_nan=False)]))
    raw = _ssh(target, [target.ssh["host"], command])
    try:
        result = json.loads(raw)
        if result.get("status") not in ("started", "starting_or_running", "listener_present"):
            raise ValueError
        return result["status"]
    except (ValueError, AttributeError):
        raise WorkflowError("GPU startup returned an unexpected response; inspect its repo-local managed service log") from None


def ensure_backend(target: BackendTarget, task: str, *, progress=lambda _value: None, startup_timeout=900,
                   own_preparation_dir=None) -> dict:
    """Authenticate/reuse or connect/start once. Never terminates services or VMs."""
    from .external_ops import (
        _validated_metadata,
        attach_service,
        http_credentials,
        owned_service,
        update_ready,
    )

    def idle():
        if own_preparation_dir is None:
            assert_ui_idle()
        else:
            assert_ui_idle(own_preparation_dir=own_preparation_dir)

    idle()
    if target.backend != "lambda" or target.policy != "molmoact2":
        raise WorkflowError("This workflow currently prepares the reviewed Lambda MolmoAct2 runtime; other policies need their native workflow")
    if not isinstance(task, str) or not task.strip() or len(task) > 2048:
        raise WorkflowError("task must contain 1–2048 characters")

    def probe():
        receipt = owned_service(target.service)
        # Existing receipt is retained when identity is unchanged, keeping qualification reusable.
        if receipt and receipt.get("status") == "ready":
            try:
                auth = http_credentials(target.service)
                if auth["endpoint_url"] != target.endpoint:
                    raise WorkflowError("Configured endpoint differs from the existing attachment; resolve that configuration explicitly")
                metadata = _validated_metadata(target.service, target.endpoint,
                                               _probe_service(target, auth["token"]), "lambda")
                if metadata.get("instance_id") == receipt["metadata"].get("instance_id"):
                    return (receipt if metadata == receipt["metadata"] else update_ready(
                        target.service, metadata, expected_instance_id=metadata["instance_id"]))
            except WorkflowError:
                raise
            except ValueError:
                pass  # Expired receipt can be replaced only using the configured private token file.
        if not target.token_file:
            raise WorkflowError("The saved service is unavailable or expired; configure token_file and optional SSH startup in " + CONFIG_RELATIVE)
        # Bounded readiness comes before the existing attachment API's generous
        # bootstrap deadline, so a wrong/stalled local listener fails promptly.
        from .external_ops import _read_private

        token = _read_private(target.token_file, 257).strip()
        _validated_metadata(target.service, target.endpoint, _probe_service(target, token), "lambda")
        return attach_service(target.service, target.endpoint, target.token_file)

    return connect_configured_runtime(target, task, probe, progress=progress, startup_timeout=startup_timeout,
                                      own_preparation_dir=own_preparation_dir)


def connect_configured_runtime(target, task, probe, *, progress=lambda _value: None, startup_timeout=900,
                               own_preparation_dir=None):
    """Shared lifecycle only; each policy supplies its own exact readiness validation."""
    def idle():
        if own_preparation_dir is None:
            assert_ui_idle()
        else:
            assert_ui_idle(own_preparation_dir=own_preparation_dir)

    idle()
    progress("Checking the configured model service")
    try:
        return probe()
    except WorkflowError:
        raise
    except ValueError:
        raise WorkflowError("Model identity or credential metadata was rejected; verify matching source, policy and private token file. "
                            "No service or tunnel was changed") from None
    except Exception as exc:  # noqa: BLE001 — private transport exceptions must not leak their payload.
        first_error = type(exc).__name__
    if not target.ssh:
        raise WorkflowError(f"Model readiness failed ({first_error}); check the existing service/tunnel or configure ssh in {CONFIG_RELATIVE}")
    if _listening(target.port) and not _managed_forward_present(target):
        raise WorkflowError("The configured loopback port has an unrecognized listener; no service or tunnel was changed. "
                            "Resolve the port/destination explicitly in the backend configuration")
    idle()
    if target.remote:
        progress("Starting or reusing the existing GPU service (no VM provisioning)")
        _start_remote(target, task)
    if not _listening(target.port):
        progress("Connecting the configured SSH forward")
        _ssh(target, ["-f", "-N", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=15",
                      "-o", "ServerAliveCountMax=3", "-L",
                      f"127.0.0.1:{target.port}:127.0.0.1:{target.port}", target.ssh["host"]])
    deadline = time.monotonic() + (startup_timeout if target.remote else 5)
    while True:
        idle()
        try:
            return probe()
        except WorkflowError:
            raise
        except ValueError:
            raise WorkflowError("Model identity or credential metadata was rejected; verify matching source, policy and private token file") from None
        except Exception as exc:  # noqa: BLE001 — do not emit credential-bearing response bodies.
            last_error = type(exc).__name__
        if time.monotonic() >= deadline:
            raise WorkflowError(f"Model readiness failed ({last_error}); an existing listener was left untouched. "
                                "Inspect data/inference/managed/<service>/service.log on the GPU and verify matching source/token")
        time.sleep(2)
