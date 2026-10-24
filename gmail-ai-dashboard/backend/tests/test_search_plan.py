from app.rag.search_plan import gmail_search_query, requested_sender, sender_only_request


def test_sender_name_and_angle_bracket_address_are_normalized():
    query = "Give mails that we received from Aadi Jadhav <adijadhav651@gmail.com>"

    assert requested_sender(query) == "adijadhav651@gmail.com"
    assert sender_only_request(query) is True
    assert 'from:"adijadhav651@gmail.com"' in gmail_search_query(query)
