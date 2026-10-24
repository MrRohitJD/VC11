from langchain_core.documents import Document

from app.rag.relevance import local_relevance_fallback


def test_local_relevance_fallback_keeps_only_retrieved_evidence_and_ranks_terms():
    documents = [
        Document(
            page_content="A password reset was requested for the account.",
            metadata={"email_id": "reset", "subject": "Password reset"},
        ),
        Document(
            page_content="Your monthly newsletter is available.",
            metadata={"email_id": "newsletter", "subject": "Newsletter"},
        ),
    ]

    result = local_relevance_fallback(documents, "Find password reset emails")

    assert [doc.metadata["email_id"] for doc in result] == ["reset"]
    assert result[0].metadata["relevance_validated"] is True
    assert result[0].metadata["relevance_source"] == "local_retrieval_fallback"


def test_local_relevance_fallback_returns_no_sources_for_no_candidates():
    assert local_relevance_fallback([], "Find security emails") == []
