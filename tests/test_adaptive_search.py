from langchain_core.documents import Document

from app.rag.activity import activity_context, emit_activity
from app.rag.adaptive import bm25_rank, fuse_rankings, plan_retrieval


def test_planner_uses_exhaustive_lookup_for_all_without_losing_date_scope():
    plan = plan_retrieval("Find all interview emails from last 3 months and summarize upcoming interviews", use_multi_query=True)
    assert plan.strategy == "exhaustive_mailbox"
    assert plan.date_start and plan.date_end
    assert not plan.multi_query


def test_planner_selects_diversity_complexity_and_exact_id_paths():
    assert plan_retrieval("Give an overview of different topics").strategy == "mmr"
    assert plan_retrieval("Compare interviews and meetings").multi_query
    assert not plan_retrieval("Compare interviews and meetings", use_multi_query=False).multi_query
    assert plan_retrieval("Read this message", {"gmail_message_id": "known-id"}).strategy == "structured"


def test_keyword_ranking_finds_synonym_without_embedding_and_boosts_subject():
    docs = [Document(page_content="Please complete the assessment.", metadata={"email_id": "assessment", "subject": "Technical screening"}),
            Document(page_content="Your monthly utility bill.", metadata={"email_id": "bill", "subject": "Invoice"})]
    result = bm25_rank(docs, "Find my interview emails")
    assert [d.metadata["email_id"] for d in result] == ["assessment"]


def test_rank_fusion_counts_repeated_chunks_once_per_retriever():
    a = Document(page_content="a", metadata={"email_id": "a"})
    b = Document(page_content="b", metadata={"email_id": "b"})
    result = fuse_rankings([a, a, a, b], [b])
    assert result[0].metadata["email_id"] == "b"
    assert len(result) == 2


def test_activity_context_is_request_scoped_and_nested_callbacks_are_preserved():
    received = []
    with activity_context(received.append) as recorder:
        with activity_context():
            emit_activity("retrieve", "Found 3 candidate emails", count=3)
        assert len(recorder.steps) == 1
    with activity_context() as other:
        assert other.steps == []
    assert received[0]["count"] == 3
    assert "thought" not in received[0]
