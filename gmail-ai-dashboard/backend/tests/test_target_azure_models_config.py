"""Tests for Azure AI Foundry model configurations:
- GPT-5 Nano (gpt-5-nano)
- Microsoft-Decision-1 (MCRO)
- Text Embedding 3 Small (text-embedding-3-small, 1536 dims)
"""

from app.core.config import Settings, settings
from app.rag import chains, embeddings
from app.services import llm_provider, embedding_provider


def test_target_azure_environment_variable_loading(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_NANO_API_KEY", "nano-test-key")
    monkeypatch.setenv("AZURE_DECISION_API_KEY", "decision-test-key")
    monkeypatch.setenv("AZURE_EMBEDDING_SMALL_API_KEY", "embedding-small-key")
    monkeypatch.setenv("GPT_MODEL_DEPLOYMENT", "gpt-5-nano")
    monkeypatch.setenv("CLASSIFICATION_MODEL_DEPLOYMENT", "MCRO")
    monkeypatch.setenv("EMBEDDING_MODEL_DEPLOYMENT", "text-embedding-3-small")

    s = Settings(_env_file=None)
    assert s.azure_openai_nano_api_key == "nano-test-key"
    assert s.azure_decision_api_key == "decision-test-key"
    assert s.azure_embedding_small_api_key == "embedding-small-key"
    assert s.gpt_model_deployment == "gpt-5-nano"
    assert s.classification_model_deployment == "MCRO"
    assert s.embedding_model_deployment == "text-embedding-3-small"
    assert s.embedding_dimensions == 1536
    assert s.vector_collection == "postgres_halfvec_1536"


def test_gpt_5_nano_chat_model_initialization(monkeypatch):
    monkeypatch.setattr(settings, "azure_openai_endpoint", "https://rg-openai-sea.cognitiveservices.azure.com/")
    monkeypatch.setattr(settings, "azure_openai_nano_api_key", "test-nano-key")
    monkeypatch.setattr(settings, "azure_openai_api_key", "test-nano-key")
    monkeypatch.setattr(settings, "gpt_model_deployment", "gpt-5-nano")
    monkeypatch.setattr(settings, "azure_openai_deployment", "gpt-5-nano")
    monkeypatch.setattr(settings, "azure_openai_api_version", "2024-02-01")

    captured = {}

    class FakeChat:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def with_fallbacks(self, fallbacks):
            return self

    monkeypatch.setattr(chains, "AzureChatOpenAI", FakeChat)
    chains.get_chat_model.cache_clear()
    try:
        chains.get_chat_model()
    finally:
        chains.get_chat_model.cache_clear()

    assert captured["azure_endpoint"] == "https://rg-openai-sea.cognitiveservices.azure.com/"
    assert captured["api_key"] == "test-nano-key"
    assert captured["azure_deployment"] == "gpt-5-nano"
    assert captured["temperature"] == 1.0
    assert "max_completion_tokens" in captured
    assert "max_tokens" not in captured


def test_text_embedding_3_small_initialization(monkeypatch):
    monkeypatch.setattr(settings, "azure_openai_endpoint", "https://rg-openai-sea.cognitiveservices.azure.com/")
    monkeypatch.setattr(settings, "azure_embedding_small_api_key", "test-emb-key")
    monkeypatch.setattr(settings, "azure_openai_embedding_api_key", "test-emb-key")
    monkeypatch.setattr(settings, "embedding_model_deployment", "text-embedding-3-small")
    monkeypatch.setattr(settings, "azure_openai_embedding_deployment", "text-embedding-3-small")

    captured = {}

    def fake_embeddings(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(embeddings, "AzureOpenAIEmbeddings", fake_embeddings)
    embeddings.get_embedding_model()

    assert captured["azure_endpoint"] == "https://rg-openai-sea.cognitiveservices.azure.com/"
    assert captured["api_key"] == "test-emb-key"
    assert captured["azure_deployment"] == "text-embedding-3-small"


def test_microsoft_decision_1_system_one_payload(monkeypatch):
    monkeypatch.setattr(settings, "azure_openai_endpoint", "https://rg-openai-sea.cognitiveservices.azure.com/")
    monkeypatch.setattr(settings, "azure_decision_api_key", "test-decision-key")
    monkeypatch.setattr(settings, "classification_model_deployment", "MCRO")

    recorded_requests = []

    class FakeResponse:
        def read(self):
            import json
            return json.dumps({
                "answers": {
                    "category": {
                        "choice": "WORK",
                        "confidence": 0.98,
                    }
                }
            }).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    import urllib.request
    def fake_urlopen(req, timeout=None):
        import json
        recorded_requests.append({
            "url": req.full_url,
            "headers": req.headers,
            "payload": json.loads(req.data.decode("utf-8")),
        })
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    test_emails = [{
        "email_id": "test-email-123",
        "subject": "Sprint planning roadmap review",
        "snippet": "Let's review the sprint roadmap for Q4.",
        "body": "Hi team, please find the roadmap attached.",
        "from_email": "pm@company.com",
    }]

    res, model_name = llm_provider._try_azure_decision(test_emails)
    assert model_name == "azure/MCRO"
    assert len(res) == 1
    assert res[0].email_id == "test-email-123"
    assert res[0].label == "WORK"
    assert res[0].confidence == 0.98

    assert len(recorded_requests) == 1
    rec = recorded_requests[0]
    assert rec["url"] == "https://rg-openai-sea.cognitiveservices.azure.com/providers/microsoft/v1/systemone"
    assert rec["headers"]["Api-key"] == "test-decision-key"
    assert rec["payload"]["model"] == "MCRO"
    assert "WORK" in rec["payload"]["questions"]["category"]["criteria"]
