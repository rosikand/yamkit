"""Hardware-free cloud bootstrap checks, also runnable without yamkit dependencies.

    python3 -m unittest tests.test_cloud_tailscale -v
"""

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import tarfile
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("cloud_tailscale", ROOT / "scripts/cloud_tailscale.py")
cloud = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cloud)
FAKE_SECRET = "tskey-client-test-not-a-real-credential"


class CloudTailscaleTests(unittest.TestCase):
    def setUp(self):
        (ROOT / ".tools").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="ts-test-", dir=ROOT / ".tools")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        (self.base / "tmp").mkdir()
        self.environ = {
            "CONDUCTOR_IS_LOCAL": "0",
            "CONDUCTOR_WORKSPACE_ID": "test-workspace",
            cloud.SECRET_ENV: FAKE_SECRET,
        }
        self.env = cloud.local_environment(self.base, self.environ)

    def test_local_setup_does_nothing(self):
        unused = self.base / "unused"
        with mock.patch.object(cloud, "install") as install:
            self.assertFalse(cloud.ensure_online(unused, {"CONDUCTOR_IS_LOCAL": "1"}))
        self.assertFalse(unused.exists())
        install.assert_not_called()

    def test_ambiguous_environment_does_not_install(self):
        unused = self.base / "unused"
        with self.assertRaises(cloud.SetupError):
            cloud.ensure_online(unused, {})
        self.assertFalse(unused.exists())

    def test_running_client_needs_no_new_secret_or_enrollment(self):
        with mock.patch.object(cloud, "status", return_value={"BackendState": "Running"}), \
                mock.patch.object(cloud, "enroll") as enroll, mock.patch.object(cloud, "install") as install:
            self.assertTrue(cloud.ensure_online(self.base, {"CONDUCTOR_IS_LOCAL": "0"}))
        enroll.assert_not_called()
        install.assert_not_called()

    def test_missing_secret_fails_before_install_or_daemon(self):
        with mock.patch.object(cloud, "status", return_value=None), \
                mock.patch.object(cloud, "start_daemon") as start, \
                mock.patch.object(cloud, "install") as install:
            with self.assertRaisesRegex(cloud.SetupError, cloud.SECRET_ENV):
                cloud.ensure_online(self.base, {"CONDUCTOR_IS_LOCAL": "0"})
        start.assert_not_called()
        install.assert_not_called()

    def test_sleep_recovery_starts_and_enrolls_a_new_daemon(self):
        with mock.patch.object(cloud, "status", side_effect=[None, {"BackendState": "Running"}]), \
                mock.patch.object(cloud, "start_daemon", return_value={"BackendState": "NeedsLogin"}) as start, \
                mock.patch.object(cloud, "enroll") as enroll, mock.patch.object(cloud, "install"):
            self.assertTrue(cloud.ensure_online(self.base, self.environ))
        start.assert_called_once()
        enroll.assert_called_once()

    def test_child_environment_contains_no_enrollment_or_conductor_secrets(self):
        environ = dict(self.environ, CONDUCTOR_API_TOKEN="private-conductor-token",
                       UNRELATED_SECRET="private-other-token", PATH="/usr/bin:/bin")
        env = cloud.local_environment(self.base, environ)
        self.assertNotIn(cloud.SECRET_ENV, env)
        self.assertNotIn("CONDUCTOR_API_TOKEN", env)
        self.assertNotIn("UNRELATED_SECRET", env)
        for key in ("XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "TMPDIR", "TS_LOGS_DIR"):
            self.assertTrue(Path(env[key]).is_relative_to(self.base))

    def test_hostnames_are_stable_and_distinct_across_workspaces_and_vms(self):
        first = cloud.workspace_hostname({"CONDUCTOR_WORKSPACE_ID": "first"})
        self.assertEqual(first, cloud.workspace_hostname({"CONDUCTOR_WORKSPACE_ID": "first"}))
        self.assertNotEqual(first, cloud.workspace_hostname({"CONDUCTOR_WORKSPACE_ID": "second"}))
        with mock.patch.object(cloud.socket, "gethostname", return_value="vm-a"):
            fallback = cloud.workspace_hostname({}, root=ROOT)
        with mock.patch.object(cloud.socket, "gethostname", return_value="vm-b"):
            self.assertNotEqual(fallback, cloud.workspace_hostname({}, root=ROOT))

    def test_oauth_file_is_private_and_deleted_without_leaking_on_failure(self):
        observed = []

        def fake_run(args, **kwargs):
            self.assertNotIn(FAKE_SECRET, " ".join(args))
            self.assertNotIn(FAKE_SECRET, str(kwargs["env"]))
            reference = next(arg for arg in args if arg.startswith("--auth-key=file:"))
            secret_file = Path(reference.removeprefix("--auth-key=file:"))
            observed.append(secret_file)
            self.assertEqual(stat.S_IMODE(secret_file.stat().st_mode), 0o600)
            self.assertEqual(secret_file.read_text(), FAKE_SECRET + "?ephemeral=true&preauthorized=true")
            self.assertIn("--shields-up", args)
            return subprocess.CompletedProcess(args, 1, stdout=FAKE_SECRET.encode(), stderr=FAKE_SECRET.encode())

        with mock.patch.object(cloud.subprocess, "run", side_effect=fake_run):
            with self.assertRaises(cloud.SetupError) as failure:
                cloud.enroll(self.base, self.env, self.environ, FAKE_SECRET, "tag:conductor-yamkit")
        self.assertNotIn(FAKE_SECRET, str(failure.exception))
        self.assertEqual(len(observed), 1)
        self.assertFalse(observed[0].exists())

    def test_oauth_file_is_deleted_on_timeout(self):
        with mock.patch.object(cloud.subprocess, "run", side_effect=subprocess.TimeoutExpired("tailscale", 55)):
            with self.assertRaisesRegex(cloud.SetupError, "timed out"):
                cloud.enroll(self.base, self.env, self.environ, FAKE_SECRET, "tag:conductor-yamkit")
        self.assertEqual(list((self.base / "tmp").iterdir()), [])

    def test_live_foreign_socket_is_not_unlinked_or_killed(self):
        server = socket.socket(socket.AF_UNIX)
        self.addCleanup(server.close)
        server.bind(str(self.base / "tailscaled.sock"))
        server.listen(1)
        with mock.patch.object(cloud.os, "kill") as kill:
            with self.assertRaisesRegex(cloud.SetupError, "unrecognized daemon"):
                cloud.clear_stale_daemon(self.base)
        kill.assert_not_called()
        self.assertTrue((self.base / "tailscaled.sock").exists())

    def test_recycled_pid_is_never_killed(self):
        (self.base / "daemon.pid").write_text(json.dumps({"pid": os.getpid(), "start": "old-start-time"}))
        with mock.patch.object(cloud.os, "kill") as kill:
            cloud.clear_stale_daemon(self.base)
        kill.assert_not_called()

    def package(self, symlink=False):
        content = io.BytesIO()
        with tarfile.open(fileobj=content, mode="w:gz") as package:
            for name in ("tailscale", "tailscaled"):
                member = tarfile.TarInfo(f"tailscale_{cloud.VERSION}_amd64/{name}")
                data = b"test executable\n"
                if symlink and name == "tailscaled":
                    member.type = tarfile.SYMTYPE
                    member.linkname = "/outside-the-repository"
                    package.addfile(member)
                else:
                    member.size = len(data)
                    package.addfile(member, io.BytesIO(data))
        return content.getvalue()

    def install_package(self, content, checksum=None):
        checksum = checksum or hashlib.sha256(content).hexdigest()
        responses = [io.BytesIO(checksum.encode()), io.BytesIO(content)]
        with mock.patch.object(cloud, "architecture", return_value="amd64"), \
                mock.patch.object(cloud.urllib.request, "urlopen", side_effect=responses):
            cloud.install(self.base)

    def test_bad_checksum_installs_nothing(self):
        with self.assertRaisesRegex(cloud.SetupError, "checksum mismatch"):
            self.install_package(self.package(), "0" * 64)
        self.assertFalse((self.base / "tailscale").exists())
        self.assertFalse((self.base / "tailscaled").exists())

    def test_archive_symlink_installs_nothing(self):
        with self.assertRaisesRegex(cloud.SetupError, "unexpected binary"):
            self.install_package(self.package(symlink=True))
        self.assertFalse((self.base / "tailscale").exists())
        self.assertFalse((self.base / "tailscaled").exists())

    def test_validated_install_is_reused_and_tampering_detected(self):
        self.install_package(self.package())
        self.assertTrue(cloud.installed(self.base, "amd64"))
        with mock.patch.object(cloud.urllib.request, "urlopen") as download, \
                mock.patch.object(cloud, "architecture", return_value="amd64"):
            cloud.install(self.base)
        download.assert_not_called()
        (self.base / "tailscale").write_bytes(b"changed")
        self.assertFalse(cloud.installed(self.base, "amd64"))


if __name__ == "__main__":
    unittest.main()
