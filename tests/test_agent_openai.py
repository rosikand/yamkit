"""All provider tests are deterministic; no real API, credentials, or hardware."""
import base64
import copy
import json
import logging
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from yamkit.agent_openai import (
    MAX_CONTEXT_CHARS,
    MAX_HISTORY_TURNS,
    MAX_IMAGE_BYTES,
    MAX_TEXT_CONTEXT_CHARS,
    TOOLS,
    MockProvider,
    OpenAIProvider,
    ProviderError,
    _without_image_payload,
    credential_status,
)


def observation(sequence=1, images=None):
    return SimpleNamespace(
        state={**{f"joint_{i}.pos": i / 10 for i in range(1, 7)}, "gripper.pos": 0.4},
        images=images if images is not None else {"fixture_top": np.full((640, 960, 3), [255, 0, 0], np.uint8)},
        captured_at=10.0 + sequence, sequence=sequence, source="fixture: synthetic test image",
    )


def response(sequence=1, *, calls=None, extra=None, status="completed"):
    if calls is None:
        calls = [(f"call-{sequence}", "observe", "{}")]
    output = list(extra or []) + [
        {"type": "function_call", "id": f"fc-{sequence}-{index}", "call_id": call_id,
         "name": name, "arguments": args, "status": "completed"}
        for index, (call_id, name, args) in enumerate(calls)
    ]
    return SimpleNamespace(
        id=f"response-{sequence}", output=output, status=status,
        usage={"input_tokens": 123, "output_tokens": 10, "total_tokens": 133,
               "output_tokens_details": {"reasoning_tokens": 7}},
    )


def client_with(*responses):
    return SimpleNamespace(responses=SimpleNamespace(create=Mock(side_effect=responses)))


@pytest.mark.parametrize("namespaced,fallback,expected", [
    ("dummy-yam-key", "dummy-fallback-key", "dummy-yam-key"),
    (None, "dummy-fallback-key", "dummy-fallback-key"),
])
def test_explicit_key_precedence_and_no_ambient_endpoint(monkeypatch, namespaced, fallback, expected):
    if namespaced is None:
        monkeypatch.delenv("YAMKIT_OPENAI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("YAMKIT_OPENAI_API_KEY", namespaced)
    monkeypatch.setenv("OPENAI_API_KEY", fallback)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://invalid.example")
    factory = Mock()
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=factory))
    provider = OpenAIProvider("test-model", "Test fixture task")
    assert credential_status() == "SET"
    factory.assert_called_once_with(api_key=expected, max_retries=0, base_url="https://api.openai.com/v1")
    assert expected not in repr(provider)


@pytest.mark.parametrize("namespaced", ["", "   "])
def test_present_empty_namespaced_key_never_falls_back(monkeypatch, namespaced):
    monkeypatch.setenv("YAMKIT_OPENAI_API_KEY", namespaced)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-fallback-key")
    factory = Mock()
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=factory))
    assert credential_status() == "MISSING"
    with pytest.raises(ProviderError, match="credential status: MISSING"):
        OpenAIProvider("test-model", "Test fixture task")
    factory.assert_not_called()


def test_missing_credentials_and_optional_sdk(monkeypatch):
    monkeypatch.delenv("YAMKIT_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "openai", None)
    assert credential_status() == "MISSING"
    # Mock and injected clients work with no optional dependency and no key.
    assert MockProvider().decide(observation(), timeout_s=1).calls[0].name == "observe"
    OpenAIProvider("test-model", "fixture", client=client_with(response()))
    monkeypatch.setenv("YAMKIT_OPENAI_API_KEY", "dummy-key")
    with pytest.raises(ProviderError, match="uv sync --extra agent"):
        OpenAIProvider("test-model", "fixture")


def test_custom_headers_cannot_override_explicit_key(monkeypatch):
    monkeypatch.setenv("YAMKIT_OPENAI_API_KEY", "dummy-key")
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "Authorization: dummy-override")
    factory = Mock()
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=factory))
    with pytest.raises(ProviderError, match="custom headers are unsupported") as exc:
        OpenAIProvider("test-model", "fixture")
    assert "dummy" not in str(exc.value)
    factory.assert_not_called()


def test_verbose_logging_cannot_dump_requests(caplog):
    with caplog.at_level(logging.DEBUG):
        for name in ("openai", "openai._base_client", "httpx", "httpcore"):
            logging.getLogger(name).setLevel(logging.DEBUG)
        OpenAIProvider("test-model", "fixture", client=client_with(response()))
        for name in ("openai._base_client", "httpx", "httpcore.connection"):
            logging.getLogger(name).debug("dummy-sensitive-request-body")
    assert "dummy-sensitive-request-body" not in caplog.text


def test_actual_sdk_request_uses_mock_transport_only(monkeypatch):
    sdk = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    monkeypatch.setenv("YAMKIT_OPENAI_API_KEY", "dummy-explicit-key")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-other-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://invalid.example")
    monkeypatch.delenv("OPENAI_CUSTOM_HEADERS", raising=False)
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={
            "id": "resp-sdk-test", "object": "response", "created_at": 123,
            "status": "completed", "model": "test-model", "output": response().output,
            "parallel_tool_calls": False, "tool_choice": "required", "tools": TOOLS,
            "usage": response().usage,
        })

    actual_factory = sdk.OpenAI

    def factory(**kwargs):
        return actual_factory(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handle)))

    monkeypatch.setattr(sdk, "OpenAI", factory)
    provider = OpenAIProvider("test-model", "fixture")
    try:
        decision = provider.decide(observation(), timeout_s=1)
        provider.record_result(decision, {"ok": True})
        provider.decide(observation(2), timeout_s=1)
    finally:
        provider.close()
    assert len(requests) == 2
    assert all(str(request.url) == "https://api.openai.com/v1/responses" for request in requests)
    assert all(request.headers["Authorization"] == "Bearer dummy-explicit-key" for request in requests)
    body = json.loads(requests[-1].content)
    assert body["input"][-2]["type"] == "function_call_output"
    assert body["input"][-2]["call_id"] == "call-1"
    assert body["input"][-1]["content"][-1]["type"] == "input_image"


def test_only_strict_tools_and_multimodal_response_request():
    client = client_with(response())
    provider = OpenAIProvider("intended-model", "Inspect the synthetic scene", client=client)
    decision = provider.decide(observation(), timeout_s=2.25)
    req = client.responses.create.call_args.kwargs
    assert req["model"] == "intended-model"
    assert req["timeout"] == 2.25
    assert req["parallel_tool_calls"] is False
    assert req["store"] is False
    assert req["max_output_tokens"] == 512
    assert req["reasoning"] == {"effort": "low"}
    assert req["include"] == ["reasoning.encrypted_content"]
    assert "previous_response_id" not in req
    assert {tool["name"] for tool in req["tools"]} == {
        "observe", "move_joints", "open_gripper", "close_gripper", "finish",
    }
    for tool in req["tools"]:
        assert tool["strict"] is True
        schema = tool["parameters"]
        assert schema["additionalProperties"] is False
        assert schema["required"] == list(schema["properties"])
    delta = TOOLS[1]["parameters"]["properties"]["delta"]
    assert delta["minItems"] == delta["maxItems"] == 6
    assert TOOLS[-1]["parameters"]["properties"]["reason"]["maxLength"] == 512
    content = req["input"][-1]["content"]
    metadata = json.loads(content[0]["text"])
    assert metadata["state"]["gripper.pos"] == 0.4
    assert metadata["source"].startswith("fixture")
    assert "fixture_top" in content[1]["text"]
    assert content[2]["type"] == "input_image"
    assert content[2]["detail"] == "low"
    data = base64.b64decode(content[2]["image_url"].split(",", 1)[1])
    assert len(data) <= MAX_IMAGE_BYTES
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert max(bgr.shape[:2]) <= 512
    assert bgr[0, 0, 2] > 250 and bgr[0, 0, 0] < 5  # genuine RGB, not red/blue swapped
    assert decision.calls[0].call_id == "call-1"
    assert decision.usage["output_tokens_details"]["reasoning_tokens"] == 7


def test_preserves_opaque_reasoning_ids_outputs_and_only_current_images():
    opaque = {"type": "reasoning", "id": "rs-opaque", "summary": [], "encrypted_content": "opaque-ciphertext"}
    message = {"type": "message", "id": "msg-1", "role": "assistant", "status": "completed",
               "content": [{"type": "output_text", "text": "Inspecting.", "annotations": []}]}
    first = response(extra=[opaque, message])
    client = client_with(first, response(2))
    provider = OpenAIProvider("test-model", "fixture", client=client)
    decision = provider.decide(observation(), timeout_s=1)
    result = {"ok": True, "measured": {"joint_1.pos": 0.1}}
    provider.record_result(decision, result)
    provider.decide(observation(2), timeout_s=1)
    items = client.responses.create.call_args.kwargs["input"]
    assert opaque in items and message in items
    assert first.output[-1] in items
    output = next(item for item in items if item.get("type") == "function_call_output")
    assert output["call_id"] == "call-1" and json.loads(output["output"]) == result
    assert json.loads(items[1]["content"][0]["text"])["observation"] == "historical; images omitted"
    images = [part for item in items for part in item.get("content", []) if part.get("type") == "input_image"]
    assert len(images) == 1
    assert "opaque-ciphertext" not in repr(decision)


def test_history_prunes_whole_turns_with_matching_outputs():
    client = client_with(*(response(i) for i in range(1, 10)))
    provider = OpenAIProvider("test-model", "fixture", client=client)
    for i in range(1, 10):
        decision = provider.decide(observation(i), timeout_s=1)
        provider.record_result(decision, {"ok": True})
    items = client.responses.create.call_args.kwargs["input"]
    calls = [item["call_id"] for item in items if item.get("type") == "function_call"]
    results = [item["call_id"] for item in items if item.get("type") == "function_call_output"]
    assert calls == results == ["call-6", "call-7", "call-8"]
    assert len(provider._turns) <= MAX_HISTORY_TURNS


def test_context_prunes_complete_turns_to_character_caps():
    client = client_with(*(response(i, extra=[{
        "type": "reasoning", "id": f"rs-{i}", "summary": [], "encrypted_content": "x" * 15_000,
    }]) for i in range(1, 5)))
    provider = OpenAIProvider("test-model", "fixture", client=client)
    for i in range(1, 5):
        decision = provider.decide(observation(i), timeout_s=1)
        provider.record_result(decision, {"ok": True, "detail": "x" * 1000})
    req = client.responses.create.call_args.kwargs
    assert len(json.dumps(_without_image_payload(req))) < MAX_TEXT_CONTEXT_CHARS
    assert len(json.dumps(req)) < MAX_CONTEXT_CHARS
    items = req["input"]
    assert [x["call_id"] for x in items if x.get("type") == "function_call"] == ["call-3"]
    assert [x["call_id"] for x in items if x.get("type") == "function_call_output"] == ["call-3"]


def test_cannot_decide_before_recording_and_cannot_record_twice():
    client = client_with(response())
    provider = OpenAIProvider("test-model", "fixture", client=client)
    decision = provider.decide(observation(), timeout_s=1)
    with pytest.raises(ProviderError, match="previous tool result"):
        provider.decide(observation(2), timeout_s=1)
    provider.record_result(decision, {"ok": True})
    with pytest.raises(ProviderError, match="pending decision"):
        provider.record_result(decision, {"ok": True})
    assert client.responses.create.call_count == 1


def test_multiple_and_empty_calls_get_matching_rejection_outputs():
    client = client_with(response(calls=[("a", "observe", "{}"), ("b", "open_gripper", "{}")]),
                         response(2, calls=[]), response(3))
    provider = OpenAIProvider("test-model", "fixture", client=client)
    first = provider.decide(observation(), timeout_s=1)
    assert len(first.calls) == 2  # Controller must reject the entire batch without action.
    provider.record_result(first, {"ok": False, "error": "multiple calls rejected"})
    second = provider.decide(observation(2), timeout_s=1)
    items = client.responses.create.call_args.kwargs["input"]
    assert [x["call_id"] for x in items if x.get("type") == "function_call_output"] == ["a", "b"]
    assert second.calls == []
    provider.record_result(second, {"ok": False, "error": "empty decision"})
    provider.decide(observation(3), timeout_s=1)
    assert "empty decision" in json.dumps(client.responses.create.call_args.kwargs["input"])


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True, "1"])
def test_invalid_timeouts_never_call_api(timeout):
    client = client_with(response())
    with pytest.raises(ProviderError, match="timeout"):
        OpenAIProvider("test-model", "fixture", client=client).decide(observation(), timeout_s=timeout)
    client.responses.create.assert_not_called()


@pytest.mark.parametrize("images", [{}, {"camera": np.zeros((10, 10), np.uint8)},
                                    {"camera": np.zeros((10, 10, 3), float)},
                                    {str(i): np.zeros((10, 10, 3), np.uint8) for i in range(5)}])
def test_invalid_images_never_call_api(images):
    client = client_with(response())
    with pytest.raises(ProviderError):
        OpenAIProvider("test-model", "fixture", client=client).decide(observation(images=images), timeout_s=1)
    client.responses.create.assert_not_called()


def test_sdk_error_never_exposes_body_and_never_retries():
    exc = RuntimeError("dummy-sensitive-key and server body")
    exc.status_code = 401
    client = client_with(exc)
    provider = OpenAIProvider("test-model", "fixture", client=client)
    with pytest.raises(ProviderError) as failure:
        provider.decide(observation(), timeout_s=1)
    assert str(failure.value) == "OpenAI request failed (HTTP 401); no retry was attempted."
    assert failure.value.__suppress_context__
    assert client.responses.create.call_count == 1


@pytest.mark.parametrize("status", ["incomplete", "failed", "in_progress"])
def test_incomplete_response_cannot_execute_a_partial_call(status):
    provider = OpenAIProvider("test-model", "fixture", client=client_with(response(status=status)))
    with pytest.raises(ProviderError, match="no tool can execute"):
        provider.decide(observation(), timeout_s=1)


def test_output_and_result_size_limits():
    oversized = response(extra=[{"type": "reasoning", "encrypted_content": "x" * 70_000}])
    provider = OpenAIProvider("test-model", "fixture", client=client_with(oversized))
    with pytest.raises(ProviderError, match="Response output exceeds"):
        provider.decide(observation(), timeout_s=1)
    provider = OpenAIProvider("test-model", "fixture", client=client_with(response()))
    decision = provider.decide(observation(), timeout_s=1)
    with pytest.raises(ProviderError, match="Tool result exceeds"):
        provider.record_result(decision, {"too_large": "x" * 20_000})


def test_model_dump_output_preserved_and_detached():
    first = response()
    raw = first.output[0]
    first.output = [SimpleNamespace(model_dump=lambda **kwargs: copy.deepcopy(raw))]
    provider = OpenAIProvider("test-model", "fixture", client=client_with(first))
    decision = provider.decide(observation(), timeout_s=1)
    assert decision.output == [raw]
    raw["name"] = "changed"
    assert decision.calls[0].name == "observe"


def test_offline_script_is_short_and_does_not_claim_success():
    provider = MockProvider()
    decisions = [provider.decide(observation(), timeout_s=1) for _ in range(3)]
    assert [decision.calls[0].name for decision in decisions] == ["observe", "move_joints", "finish"]
    assert json.loads(decisions[-1].calls[0].arguments)["success"] is False
