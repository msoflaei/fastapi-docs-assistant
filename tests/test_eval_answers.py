"""Unit tests for the answer-grading helpers (no API calls)."""
import pytest

from docs_assistant.eval_answers import cited_numbers, is_refusal, parse_judgment


def test_refusal_detected_with_straight_or_curly_apostrophe():
    assert is_refusal("I couldn't find this in the FastAPI documentation.")
    assert is_refusal("I couldn\u2019t find this in the FastAPI documentation.")
    assert not is_refusal("Use fastapi dev [1].")


def test_citations_ignore_code():
    answer = "Use a list [1].\n```python\nx = items[0]\n```\nAlso `data[2]` and see [3]."
    assert cited_numbers(answer) == {1, 3}


def test_judgment_parsed_from_json_in_code_fence():
    text = '```json\n{"points": [true, false], "faithfulness": "yes", "explanation": "ok"}\n```'
    assert parse_judgment(text, n_points=2)["points"] == [True, False]


def test_judgment_with_wrong_point_count_is_rejected():
    text = '{"points": [true], "faithfulness": "yes", "explanation": "ok"}'
    with pytest.raises(ValueError):          # the test passes only if a ValueError is raised
        parse_judgment(text, n_points=2)