"""Unit tests for the retrieval metrics."""
from docs_assistant.eval_retrieval import score_question


def test_relevant_page_at_rank_one():
    scores = score_question(["a", "b", "c"], ["a"])
    assert scores["hit@1"] == 1.0
    assert scores["mrr"] == 1.0


def test_relevant_page_at_rank_three():
    scores = score_question(["x", "y", "a", "z"], ["a"])
    assert scores["hit@1"] == 0.0
    assert scores["hit@3"] == 1.0
    assert scores["mrr"] == 1 / 3


def test_recall_counts_distinct_relevant_pages():
    # Two relevant pages, but only one is retrieved (twice): recall is 1/2, not 2/2.
    scores = score_question(["a", "a", "x"], ["a", "b"])
    assert scores["recall@3"] == 0.5


def test_nothing_relevant_found():
    scores = score_question(["x", "y"], ["a"])
    assert scores["hit@10"] == 0.0
    assert scores["mrr"] == 0.0