"""Chat and embedding requests can use distinct Azure OpenAI resources."""

from app.core.config import Settings, settings
from app.rag import embeddings
from app.services import embedding_provider


def test_separate_embedding_settings_load_from_environment(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_ENDPOINT", "https://embedding-resource.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_API_KEY", "embedding-resource-key")
    configured = Settings(_env_file=None)
    assert configured.azure_openai_embedding_endpoint == "https://embedding-resource.openai.azure.com/"
    assert configured.azure_openai_embedding_api_key == "embedding-resource-key"


def test_query_embeddings_use_separate_resource_then_shared_fallback(monkeypatch):
    monkeypatch.setattr(settings, "azure_openai_endpoint", "https://chat-resource.openai.azure.com/")
    monkeypatch.setattr(settings, "azure_openai_api_key", "chat-resource-key")
    monkeypatch.setattr(settings, "azure_openai_embedding_endpoint", "https://embedding-resource.openai.azure.com/")
    monkeypatch.setattr(settings, "azure_openai_embedding_api_key", "embedding-resource-key")
    monkeypatch.setattr(embeddings, "AzureOpenAIEmbeddings", lambda **kwargs: kwargs)

    separate = embeddings.get_embedding_model()
    assert separate["azure_endpoint"] == "https://embedding-resource.openai.azure.com/"
    assert separate["api_key"] == "embedding-resource-key"

    monkeypatch.setattr(settings, "azure_openai_embedding_endpoint", "")
    monkeypatch.setattr(settings, "azure_openai_embedding_api_key", "")
    shared = embeddings.get_embedding_model()
    assert shared["azure_endpoint"] == "https://chat-resource.openai.azure.com/"
    assert shared["api_key"] == "chat-resource-key"


def test_background_indexer_embeddings_use_separate_resource(monkeypatch):
    monkeypatch.setattr(settings, "azure_openai_endpoint", "https://chat-resource.openai.azure.com/")
    monkeypatch.setattr(settings, "azure_openai_api_key", "chat-resource-key")
    monkeypatch.setattr(settings, "azure_openai_embedding_endpoint", "https://embedding-resource.openai.azure.com/")
    monkeypatch.setattr(settings, "azure_openai_embedding_api_key", "embedding-resource-key")
    captured = {}

    def fake_client(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(embedding_provider, "AzureOpenAIEmbeddings", fake_client)
    provider = embedding_provider.EmbeddingProvider()
    assert provider.is_mock is False
    assert captured["azure_endpoint"] == "https://embedding-resource.openai.azure.com/"
    assert captured["api_key"] == "embedding-resource-key"
