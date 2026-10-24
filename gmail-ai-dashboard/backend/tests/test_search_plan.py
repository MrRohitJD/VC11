from app.rag.search_plan import date_only_request, gmail_search_query, requested_sender, sender_only_request


def test_sender_name_and_angle_bracket_address_are_normalized():
    query = "Give mails that we received from Aadi Jadhav <adijadhav651@gmail.com>"

    assert requested_sender(query) == "adijadhav651@gmail.com"
    assert sender_only_request(query) is True
    gmail_query = gmail_search_query(query)
    assert 'from:"adijadhav651@gmail.com"' in gmail_query
    assert "category:primary" not in gmail_query
    assert "category:social" not in gmail_query
    assert "-in:spam" in gmail_query
    assert "-in:trash" in gmail_query
    assert "-in:drafts" in gmail_query


def test_date_scoped_summaries_use_the_exhaustive_local_path():
    assert date_only_request("Summarize this week's mails") is True
    assert date_only_request("Summarize this week’s mails") is True


def test_plain_sender_requests_are_deterministic_even_with_filler_words():
    assert sender_only_request("A few mails that we received from chess.com") is True
