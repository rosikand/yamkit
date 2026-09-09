"""Pre-launch external runtime provenance survives portable bundling without credentials."""

import json

import pytest

from tests.test_rollout_artifacts import (
    recording as recording,  # noqa: PLC0414 — re-export shared pytest fixture
)
from yamkit.rollout_artifacts import package_rollout, sanitize, validate_bundle


def test_bundle_preserves_original_runtime_snapshot_and_sanitizes_nested_private_fields(recording):
    run, trace = recording
    runtime = {
        "external_service": {"provider": "lambda", "service_id": "lambda-georgia", "host_id": "a" * 64,
                             "region": "Georgia", "region_source": "operator_declared"},
        "runtime_provenance": {"python": "3.12.14", "torch_cuda": "12.8",
                               "packages": {"torch": "2.11.0+cu128", "tokenizers": "0.22.2"},
                               "gpu": {"device": "cuda:0", "name": "NVIDIA H100 80GB HBM3"},
                               "api_token": "private-runtime-token"},
        "instance_id": "captured-before-launch-instance", "inference_build_id": "b" * 64,
        "http_ingress": "ssh", "http_session_expires_at": 1788964922.4488869,
        "http_endpoint": "http://127.0.0.1:8765", "token_file": "/private/credential/path",
        "authorization": "Bearer private-authorization",
        "nested": {"url": "https://private.example.test/secret", "password": "private-password"},
    }
    original = {"provenance": {"kind": "before_managed_child_launch", "captured_at": 123},
                "remote_runtime": runtime, "unreviewed_extra": {"private": "not-allowlisted"}}
    source = run / "run_metadata.json"
    source.write_text(json.dumps(original))
    original_bytes = source.read_bytes()
    bundle = package_rollout(run, trace_dir=trace)
    validate_bundle(bundle)
    value = json.loads((bundle / "run_metadata.json").read_text())
    assert value["provenance"] == original["provenance"]
    captured = value["remote_runtime"]
    assert captured["external_service"] == runtime["external_service"]
    assert captured["instance_id"] == runtime["instance_id"]
    assert captured["inference_build_id"] == runtime["inference_build_id"]
    assert captured["runtime_provenance"]["packages"] == runtime["runtime_provenance"]["packages"]
    assert captured["runtime_provenance"]["gpu"] == runtime["runtime_provenance"]["gpu"]
    assert captured["http_session_expires_at"] == runtime["http_session_expires_at"]
    assert captured["http_ingress"] == "ssh"
    assert source.read_bytes() == original_bytes
    for private in ("private-runtime-token", "private-authorization", "private-password", "/private/credential/path",
                    "http://127.0.0.1:8765", "private.example.test", "unreviewed_extra"):
        assert private not in json.dumps(value)


@pytest.mark.parametrize("version", ["hf_ABCDEFGH12345678", "secret", "0.22.2-private-secret", {"token": "private"},
                                     "123456789.0", "0.22.2\n"])
def test_tokenizers_exception_never_preserves_nonversion_credentials(version):
    value = {"runtime_provenance": {"packages": {"tokenizers": version}}}
    assert sanitize(value) == {"runtime_provenance": {"packages": {}}}


def test_tokenizers_exception_is_confined_to_runtime_package_versions():
    assert sanitize({"tokenizers": "0.22.2", "packages": {"tokenizers": "0.22.2"}}) == {"packages": {}}
    value = {"remote_runtime": {"runtime_provenance": {"packages": {
        "tokenizers": "0.22.2", "access_token": "credential", "secret_package": "credential"}}}}
    assert sanitize(value) == {"remote_runtime": {"runtime_provenance": {"packages": {"tokenizers": "0.22.2"}}}}
