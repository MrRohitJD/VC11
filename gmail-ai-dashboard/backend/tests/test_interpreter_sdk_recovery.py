"""Exercise real SDK parsing with local transports; never contact providers."""

import json

import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI

from app.rag.request_interpreter import (
    InterpretationResponseError,
    _validated_interpretation,
    interpret_request,
)
from app.rag.service_errors import error_code


def test_real_nvidia_sdk_recovers_when_structured_parser_discards_response(monkeypatch):
    calls = []
    monkeypatch.setattr(ChatNVIDIA, "available_models", property(lambda self: []))

    def generate(self, messages, **kwargs):
        calls.append(kwargs)
        structured = any(key in kwargs for key in ("response_format", "guided_json", "nvext"))
        content = "Hello!" if structured else '{"intent":"GENERAL_CHAT"}'
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    monkeypatch.setattr(ChatNVIDIA, "_generate", generate)
    model = ChatNVIDIA(model="openai/gpt-oss-20b", api_key="test-only")
    assert interpret_request(model, "Hello").intent == "GENERAL_CHAT"
    # SDK attempts three structured formats; the app performs one plain recovery.
    assert len(calls) == 4
    assert not any(key in calls[-1] for key in ("response_format", "guided_json", "nvext"))


def test_real_chat_sdk_uses_plain_json_for_bounded_recovery():
    payloads = []

    def completion(request):
        payload = json.loads(request.content)
        payloads.append(payload)
        message = {"role": "assistant", "content": (
            "Hello!" if "tools" in payload else '{"intent":"GENERAL_CHAT"}'
        )}
        return httpx.Response(200, json={"id": "local", "object": "chat.completion",
            "created": 1, "model": "test", "choices": [
                {"index": 0, "finish_reason": "stop", "message": message}]})

    with httpx.Client(transport=httpx.MockTransport(completion)) as client:
        model = ChatOpenAI(model="test", api_key="test-only", base_url="https://example.test/v1",
                           http_client=client, max_retries=0)
        assert interpret_request(model, "Hello").intent == "GENERAL_CHAT"
    assert len(payloads) == 2
    assert "tools" in payloads[0] and "tools" not in payloads[1]
    assert "JSON schema" in payloads[1]["messages"][0]["content"]


@pytest.mark.parametrize("status,expected", [(401, "authentication"), (403, "permission"), (429, "rate_limit")])
def test_transport_and_nvidia_statuses_are_classified_consistently(status, expected):
    response = httpx.Response(status, request=httpx.Request("POST", "https://example.test"))
    error = httpx.HTTPStatusError("private response", request=response.request, response=response)
    assert error_code(error) == expected
    assert error_code(Exception(f"[{status}] private response")) == expected


@pytest.mark.parametrize("status", [401, 403])
def test_auth_and_permission_failures_never_trigger_json_recovery(status):
    calls = []

    def completion(request):
        calls.append(request)
        return httpx.Response(status, json={"error": {"message": "private", "type": "access_error"}})

    with httpx.Client(transport=httpx.MockTransport(completion)) as client:
        model = ChatOpenAI(model="test", api_key="test-only", base_url="https://example.test/v1",
                           http_client=client, max_retries=0)
        with pytest.raises(Exception) as failure:
            interpret_request(model, "Hello")
        assert error_code(failure.value) == ("authentication" if status == 401 else "permission")
    assert len(calls) == 1


def test_recovery_preserves_action_validation_and_rejects_unknown_tools():
    for payload in (
        {"intent": "ACTION_SEND", "draft": {"recipient": "a@example.test", "subject": "Hi", "body": "Hi"}},
        {"intent": "ACTION_DRAFT"},
    ):
        with pytest.raises(ValueError):
            _validated_interpretation(AIMessage(content=json.dumps(payload)))
    with pytest.raises(InterpretationResponseError):
        _validated_interpretation(AIMessage(content="", tool_calls=[
            {"id": "one", "name": "send_email", "args": {}}]))
