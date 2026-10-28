"""Inbox assistant provider-order and empty-reply regressions."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app.rag import chains
from app.rag.agent import GmailAIAssistant, parse_draft_request


def test_configured_order_advances_on_error_and_empty_answer(monkeypatch):
    calls = []

    def provider(name, response):
        def invoke(_messages):
            calls.append(name)
            if isinstance(response, Exception):
                raise response
            return AIMessage(content=response)
        return RunnableLambda(invoke)

    monkeypatch.setattr(chains.settings, "azure_openai_api_key", "test-key")
    monkeypatch.setattr(chains.settings, "azure_openai_endpoint", "https://example.test")
    monkeypatch.setattr(chains.settings, "nvidia_api_key", "test-key")
    monkeypatch.setattr(chains.settings, "explabs_api_key", "test-key")
    monkeypatch.setattr(chains, "AzureChatOpenAI", lambda **_: provider("azure", TimeoutError()))
    monkeypatch.setattr(chains, "ChatOpenAI", lambda **_: provider("lab", "Lab answer"))
    nvidia_module = ModuleType("langchain_nvidia_ai_endpoints")
    nvidia_module.ChatNVIDIA = lambda **_: provider("nvidia", "  ")
    monkeypatch.setitem(sys.modules, "langchain_nvidia_ai_endpoints", nvidia_module)

    chains.get_chat_model.cache_clear()
    try:
        model = chains.get_chat_model()
        assert chains.answer_runnable(model).invoke("question").content == "Lab answer"
        assert calls == ["azure", "nvidia", "lab"]
    finally:
        chains.get_chat_model.cache_clear()


def test_primary_answer_does_not_call_fallbacks():
    calls = []
    providers = [
        RunnableLambda(lambda _: calls.append("azure") or AIMessage(content="Azure answer")),
        RunnableLambda(lambda _: calls.append("nvidia") or AIMessage(content="Nvidia answer")),
    ]
    model = providers[0].with_fallbacks(providers[1:])
    assert chains.answer_runnable(model).invoke("question").content == "Azure answer"
    assert calls == ["azure"]


def test_tool_call_without_text_is_a_valid_provider_response():
    calls = []

    class Provider:
        def __init__(self, name, response):
            self.name, self.response = name, response

        def bind_tools(self, _tools):
            return RunnableLambda(lambda _: calls.append(self.name) or self.response)

    model = SimpleNamespace(
        runnable=Provider("azure", AIMessage(content="")),
        fallbacks=[
            Provider("nvidia", AIMessage(content="", tool_calls=[
                {"name": "search", "args": {}, "id": "call-1"},
            ])),
            Provider("lab", AIMessage(content="Lab answer")),
        ],
    )
    response = chains.answer_runnable(model, tools=[]).invoke("question")
    assert response.tool_calls[0]["name"] == "search"
    assert calls == ["azure", "nvidia"]


@pytest.mark.parametrize("primary_result", [
    TimeoutError(),
    AIMessage(content='{"subject":"","body":"Missing subject"}'),
])
def test_draft_generation_uses_fallback_after_primary_failure(primary_result):
    calls = []
    def invoke_primary(_messages):
        calls.append("azure")
        if isinstance(primary_result, Exception):
            raise primary_result
        return primary_result
    primary = RunnableLambda(invoke_primary)
    fallback = RunnableLambda(lambda _: calls.append("nvidia") or AIMessage(content=(
        '{"subject":"Assignment submission","body":"Please confirm the deadline."}'
    )))
    assistant = GmailAIAssistant.__new__(GmailAIAssistant)
    assistant.model = primary.with_fallbacks([fallback])
    draft = assistant.generate_email_draft(parse_draft_request(
        "Draft an email to teacher@example.com about assignment submission"
    ))
    assert draft["subject"] == "Assignment submission"
    assert calls == ["azure", "nvidia"]


def test_grounded_answer_uses_fallback_after_empty_primary():
    calls = []
    primary = RunnableLambda(lambda _: calls.append("azure") or AIMessage(content=""))
    fallback = RunnableLambda(lambda _: calls.append("nvidia") or AIMessage(content="Grounded answer"))
    chain = chains.GroundedRAGChain(model=primary.with_fallbacks([fallback]))
    assert chain.chain.invoke({"context": "Evidence", "question": "What happened?"}) == "Grounded answer"
    assert calls == ["azure", "nvidia"]
