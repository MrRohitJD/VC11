from app.rag.retrieval import normalize_key_point, normalize_key_points


def test_key_point_is_one_sentence_and_at_most_100_characters():
    point = normalize_key_point(
        "Your interview is on Friday at 10 AM. https://example.test/tracking "
        "This second sentence must not appear."
    )
    assert point == "Your interview is on Friday at 10 AM."
    assert len(point) <= 100
    assert "http" not in point


def test_key_points_are_deduplicated_and_bounded():
    points = normalize_key_points(["• A short fact", "A short fact.", ""])
    assert points == ["A short fact."]
