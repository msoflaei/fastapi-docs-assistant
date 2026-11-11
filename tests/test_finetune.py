"""Unit test for building fine-tuning pairs (no training)."""
from docs_assistant.finetune import build_pairs
from docs_assistant.search import QUERY_PREFIX


def test_pairs_use_breadcrumbs_and_skip_short_sections_and_excluded_pages():
    body = "word " * 30
    docs = [
        {"doc_id": "tutorial/x", "title": "X",
         "text": f"# X {{ #x }}\n\n{body}\n\n## Tiny\n\nshort\n\n## Sub {{ #sub }}\n\n{body}"},
        {"doc_id": "release-notes", "title": "Release Notes", "text": f"# Release Notes\n\n{body}"},
    ]
    pairs = build_pairs(docs)
    assert [p["anchor"] for p in pairs] == [QUERY_PREFIX + "X", QUERY_PREFIX + "X > Sub"]
    assert all(len(p["positive"].split()) == 30 for p in pairs)