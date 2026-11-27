import json

from app.rag.chains import parse_grounded_answer_output


def test_parses_llm_designed_ui_without_replacing_grounded_answer():
    raw = json.dumps({
        "answer": "The bill is due on the 28th. [1]",
        "presentation": {
            "html": '<article><button data-source="1">Electricity bill</button></article>',
            "css": "article { display: grid; }",
            "javascript": "document.querySelector('article').classList.add('ready');",
        },
    })

    answer, presentation = parse_grounded_answer_output(raw, 1)

    assert answer == "The bill is due on the 28th. [1]"
    assert presentation["html"].startswith("<article>")
    assert presentation["css"] == "article { display: grid; }"
    assert presentation["javascript"].startswith("document.querySelector")


def test_rejects_oversized_generated_ui_but_keeps_answer():
    raw = json.dumps({"answer": "A grounded answer.", "presentation": {
        "html": "<div>" + "x" * 16000 + "</div>", "css": "", "javascript": "",
    }})

    assert parse_grounded_answer_output(raw, 1) == ("A grounded answer.", None)
