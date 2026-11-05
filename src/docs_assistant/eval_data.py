"""Load and validate the evaluation question set."""
import json
from difflib import get_close_matches     # finds similar strings, to suggest fixes for typos
from pathlib import Path

QUESTIONS_FILE = Path("evals/questions.jsonl")
DOCS_FILE = Path("data/processed/docs.jsonl")
CATEGORIES = {"in_scope", "out_of_scope"}


def load_questions(path: Path = QUESTIONS_FILE) -> list[dict]:
    """Read the questions, skipping any blank lines."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_doc_titles() -> dict[str, str]:
    """Map every doc_id in our processed data to its page title."""
    with DOCS_FILE.open(encoding="utf-8") as f:
        return {doc["doc_id"]: doc["title"] for doc in map(json.loads, f)}


def validate(questions: list[dict], titles: dict[str, str]) -> list[str]:
    """Return a list of problems found in the questions (empty if all is well)."""
    problems = []
    seen_ids = set()
    for q in questions:
        qid = q.get("id", "?")
        if qid in seen_ids:
            problems.append(f"{qid}: duplicate id")
        seen_ids.add(qid)

        category = q.get("category")
        docs = q.get("relevant_docs", [])
        if category not in CATEGORIES:
            problems.append(f"{qid}: category must be one of {sorted(CATEGORIES)}")
        if category == "in_scope" and not docs:
            problems.append(f"{qid}: in-scope questions need at least one relevant doc")
        if category == "out_of_scope" and docs:
            problems.append(f"{qid}: out-of-scope questions should have no relevant docs")
        if not q.get("answer_points"):
            problems.append(f"{qid}: missing answer_points")

        for doc_id in docs:
            if doc_id not in titles:
                suggestions = get_close_matches(doc_id, titles, n=3)
                hint = f" (did you mean: {', '.join(suggestions)}?)" if suggestions else ""
                problems.append(f"{qid}: unknown doc '{doc_id}'{hint}")
    return problems


def main() -> None:
    questions = load_questions()
    titles = load_doc_titles()

    # Print every question with the titles of its relevant pages, for review by eye.
    for q in questions:
        print(f"{q['id']} [{q['category']}] {q['question']}")
        for doc_id in q.get("relevant_docs", []):
            print(f"      -> {doc_id}: {titles.get(doc_id, '??? UNKNOWN')}")

    in_scope = sum(q.get("category") == "in_scope" for q in questions)
    print(f"\n{len(questions)} questions: {in_scope} in scope, {len(questions) - in_scope} out of scope")

    problems = validate(questions, titles)
    if problems:
        print(f"\n{len(problems)} problem(s) found:")
        for p in problems:
            print("  -", p)
    else:
        print("\nNo problems found.")


if __name__ == "__main__":
    main()