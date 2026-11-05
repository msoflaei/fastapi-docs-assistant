"""Evaluate answer quality: generate an answer for every eval question, then grade it.

Grading combines cheap code checks (refusals, citations) with an LLM judge
(answer points covered, faithfulness to the retrieved documents).

Usage:
    python -m docs_assistant.eval_answers --label smoke --limit 3     # quick test
    python -m docs_assistant.eval_answers --label baseline            # full run
"""
import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

from anthropic import Anthropic

from docs_assistant.db import connect
from docs_assistant.embed import load_model
from docs_assistant.eval_data import load_questions
from docs_assistant.generate import MODEL, build_prompt, generate_answer, response_text  # also loads .env
from docs_assistant.retrieve import retrieve

RESULTS_DIR = Path("evals/results")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-sonnet-5-5")

REFUSAL_TEXT = "couldn't find this in the fastapi documentation"   # lowercase, from the system prompt
CITATION_RE = re.compile(r"\[(\d+)\]")                  # matches [1], [12], ...
CODE_RE = re.compile(r"```.*?```|`[^`]*`", re.DOTALL)   # code blocks and `inline code`

JUDGE_SYSTEM = """You are a strict evaluator of answers written by a documentation assistant for the FastAPI web framework.

You will receive the documents the assistant was given, the user's question, the assistant's answer,
and a numbered list of key points that a correct answer should convey.

Evaluate:
1. For each key point: does the answer convey it? Judge meaning, not exact wording.
2. Faithfulness: is every factual claim in the answer supported by the documents?
   - "yes": all claims are supported by the documents.
   - "partial": mostly supported, but at least one claim is not in the documents.
   - "no": important claims are not supported by the documents.

Respond with ONLY a JSON object and no other text, in exactly this format:
{"points": [true, false], "faithfulness": "yes", "explanation": "one or two sentences"}
The "points" list must contain exactly one true/false value per key point, in order."""


# ---------------------------------------------------------------------------
# Code-based checks: cheap, fast, and perfectly consistent
# ---------------------------------------------------------------------------

def is_refusal(answer: str) -> bool:
    """Did the assistant say it couldn't find the answer? (Handles curly apostrophes too.)"""
    return REFUSAL_TEXT in answer.replace("\u2019", "'").lower()


def cited_numbers(answer: str) -> set[int]:
    """The document numbers cited in the answer, ignoring anything inside code
    (where something like items[0] isn't a citation)."""
    return {int(n) for n in CITATION_RE.findall(CODE_RE.sub("", answer))}


# ---------------------------------------------------------------------------
# The LLM judge
# ---------------------------------------------------------------------------

def build_judge_prompt(question: str, chunks: list[dict], answer: str, points: list[str]) -> str:
    """Documents + question (reusing the assistant's prompt format), then the answer and key points."""
    numbered_points = "\n".join(f"{i}. {p}" for i, p in enumerate(points, start=1))
    return (f"{build_prompt(question, chunks)}\n\n"
            f"<answer>\n{answer}\n</answer>\n\n"
            f"<key_points>\n{numbered_points}\n</key_points>")


def parse_judgment(text: str, n_points: int) -> dict:
    """Extract and check the judge's JSON. Raises ValueError if it's malformed."""
    match = re.search(r"\{.*\}", text, re.DOTALL)       # find the JSON object, even inside ```json fences
    if not match:
        raise ValueError("no JSON object in judge reply")
    data = json.loads(match.group(0))                    # raises ValueError if it isn't valid JSON
    if len(data.get("points", [])) != n_points:
        raise ValueError("wrong number of points")
    if data.get("faithfulness") not in {"yes", "partial", "no"}:
        raise ValueError("invalid faithfulness value")
    return data


def judge(client: Anthropic, question: str, chunks: list[dict], answer: str, points: list[str]) -> dict:
    """Ask the judge model to grade one answer, retrying if its output can't be parsed."""
    for _ in range(3):
        response = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=2048,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": build_judge_prompt(question, chunks, answer, points)}],
        )
        try:
            return parse_judgment(response_text(response), len(points))
        except ValueError:
            continue                                     # try again
    return {"points": [False] * len(points), "faithfulness": "error",
            "explanation": "Judge output could not be parsed after 3 attempts."}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate answer quality with an LLM judge.")
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--k", type=int, default=5, help="chunks retrieved per question")
    parser.add_argument("--limit", type=int, help="only evaluate the first N questions (for quick tests)")
    args = parser.parse_args()

    questions = load_questions()[: args.limit]          # [:None] keeps all of them
    model = load_model()
    client = Anthropic()
    print(f"Answering with {MODEL}, judging with {JUDGE_MODEL}, {len(questions)} questions\n")

    results = []
    with connect() as conn:
        for q in questions:
            chunks = retrieve(q["question"], model, conn, k=args.k)
            start = time.perf_counter()
            answer = generate_answer(q["question"], chunks, client)
            seconds = time.perf_counter() - start

            cited = cited_numbers(answer)
            verdict = judge(client, q["question"], chunks, answer, q["answer_points"])
            result = {
                "id": q["id"],
                "category": q["category"],
                "question": q["question"],
                "answer": answer,
                "sources": [c["chunk_id"] for c in chunks],
                "answer_points": q["answer_points"],
                "points_covered": verdict["points"],
                "faithfulness": verdict["faithfulness"],
                "judge_explanation": verdict["explanation"],
                "refused": is_refusal(answer),
                "cited": sorted(cited),
                "invalid_citations": sorted(n for n in cited if not 1 <= n <= len(chunks)),
                "seconds": round(seconds, 2),
            }
            results.append(result)
            print(f"{q['id']}  points {sum(result['points_covered'])}/{len(result['points_covered'])}  "
                  f"faithful: {result['faithfulness']:<7}  refused: {result['refused']}")

    # ---- Summary ----
    ins = [r for r in results if r["category"] == "in_scope"]
    oos = [r for r in results if r["category"] == "out_of_scope"]
    summary = {}
    if ins:
        summary["in_scope"] = {
            "n": len(ins),
            "point_coverage": round(mean(sum(r["points_covered"]) / len(r["points_covered"]) for r in ins), 3),
            "fully_correct": round(mean(all(r["points_covered"]) for r in ins), 3),
            "faithful": round(mean(r["faithfulness"] == "yes" for r in ins), 3),
            "partially_faithful": sum(r["faithfulness"] == "partial" for r in ins),
            "unfaithful": sum(r["faithfulness"] == "no" for r in ins),
            "false_refusals": sum(r["refused"] for r in ins),
            "answers_without_citations": sum(not r["refused"] and not r["cited"] for r in ins),
            "answers_with_invalid_citations": sum(bool(r["invalid_citations"]) for r in ins),
        }
    if oos:
        summary["out_of_scope"] = {
            "n": len(oos),
            "correct_refusals": round(mean(r["refused"] for r in oos), 3),
            "judge_all_points": round(mean(all(r["points_covered"]) for r in oos), 3),
        }
    summary["avg_answer_seconds"] = round(mean(r["seconds"] for r in results), 2)
    summary["judge_errors"] = sum(r["faithfulness"] == "error" for r in results)

    print("\n" + json.dumps(summary, indent=2))

    # ---- Error analysis: every answer that isn't fully right ----
    problems = [r for r in results if
                (r["category"] == "in_scope" and
                 (not all(r["points_covered"]) or r["faithfulness"] != "yes" or r["refused"]))
                or (r["category"] == "out_of_scope" and not r["refused"])]
    print(f"\nAnswers needing review: {len(problems)}")
    for r in problems:
        missing = [p for p, ok in zip(r["answer_points"], r["points_covered"]) if not ok]
        print(f"\n  {r['id']}: {r['question']}")
        print(f"     faithfulness: {r['faithfulness']}   refused: {r['refused']}")
        if missing:
            print(f"     missing points: {missing}")
        print(f"     judge: {r['judge_explanation']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / f"answers_{args.label}.json"
    out_file.write_text(json.dumps({
        "label": args.label,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "answer_model": MODEL,
        "judge_model": JUDGE_MODEL,
        "k": args.k,
        "summary": summary,
        "per_question": results,
    }, indent=2, ensure_ascii=False))
    print(f"\nSaved -> {out_file}")


if __name__ == "__main__":
    main()