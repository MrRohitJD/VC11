"""Regression tests for topic-aware outbound email drafting."""

import pytest
from langchain_core.messages import AIMessage

from app.rag.agent import (
    GmailAIAssistant,
    _parse_generated_draft,
    build_deterministic_draft_request,
    parse_draft_request,
)
from app.services.gmail_service import normalize_recipient_header


EXAMPLE_REQUEST = (
    'Draft a professional reply to adijadhav651@gmail.com related to '
    '"send university assignment in college portal in 2 days only " '
    'draft a long mail around 800 words long'
)


def test_parser_separates_topic_from_length_instruction():
    parsed = parse_draft_request(EXAMPLE_REQUEST)

    assert parsed is not None
    assert parsed["recipient"] == "adijadhav651@gmail.com"
    assert parsed["topic"] == "send university assignment in college portal in 2 days only"
    assert parsed["word_count"] == 800
    assert "draft a long mail" not in parsed["topic"].lower()


def test_parser_extracts_cc_and_bcc_addresses():
    parsed = parse_draft_request(
        'Draft a professional email to owner@example.com cc advisor@example.com, '
        'mentor@example.com bcc archive@example.com related to "assignment submission"'
    )

    assert parsed is not None
    assert parsed["cc"] == "advisor@example.com, mentor@example.com"
    assert parsed["bcc"] == "archive@example.com"


def test_parser_accepts_mailto_typo_in_explicit_draft_request():
    parsed = parse_draft_request(
        "Draft a professional mailto adijadhav651@gmail.com friend related to "
        "study hard for end term semester exam."
    )

    assert parsed is not None
    assert parsed["recipient"] == "adijadhav651@gmail.com"
    assert parsed["topic"] == "study hard for end term semester exam"


def test_fallback_subject_is_specific_to_assignment_portal_topic():
    draft = build_deterministic_draft_request(EXAMPLE_REQUEST)

    assert draft is not None
    assert draft["subject"] == "University Assignment Submission Through the College Portal"
    assert "university assignment" in draft["body"].lower()
    assert "requested target" not in draft["body"].lower()
    assert "800 words" not in draft["body"].lower()


def test_model_generated_draft_returns_subject_and_body():
    class FakeModel:
        def __init__(self):
            self.messages = None

        def invoke(self, messages):
            self.messages = messages
            return AIMessage(content=(
                '{"subject":"University Assignment Portal Submission Deadline",'
                '"body":"Hello,\\n\\nPlease confirm the assignment submission details.\\n\\nBest regards,\\n[Your Name]"}'
            ))

    assistant = GmailAIAssistant.__new__(GmailAIAssistant)
    assistant.model = FakeModel()
    parsed = parse_draft_request(EXAMPLE_REQUEST)
    generated = assistant.generate_email_draft(parsed)

    assert generated["subject"] == "University Assignment Portal Submission Deadline"
    assert generated["body"].startswith("Hello,")
    assert "send university assignment in college portal" in assistant.model.messages[1].content
    assert "approximately 800 words" in assistant.model.messages[1].content
    assert "Never mention the requested word count" in assistant.model.messages[0].content


def test_model_output_with_drafting_metadata_is_rejected():
    with pytest.raises(ValueError, match="drafting metadata"):
        _parse_generated_draft(
            '{"subject":"Assignment submission",'
            '"body":"This email is approximately 800 words and ready to send."}'
        )


def test_recipient_header_normalization_supports_cc_and_bcc_lists():
    assert normalize_recipient_header("advisor@example.com; mentor@example.com", "Cc") == (
        "advisor@example.com, mentor@example.com"
    )
    with pytest.raises(ValueError, match="Bcc"):
        normalize_recipient_header("not-an-email", "Bcc")
