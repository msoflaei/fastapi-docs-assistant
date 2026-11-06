"""Estimate plain RAG's input tokens per question, without calling the API.

Builds the exact prompt RAG would send (system prompt + 8 retrieved chunks + question)
for every eval question, and estimates tokens as characters / 4 (a rough rule of thumb).
"""
from statistics import mean

from docs_assistant.db import connect
from docs_assistant.embed import load_model
from docs_assistant.eval_data import load_questions
from docs_assistant.generate import SYSTEM_PROMPT, build_prompt
from docs_assistant.retrieve import DEFAULT_K, retrieve

CHARS_PER_TOKEN = 4

model = load_model()
estimates = []
with connect() as conn:
    for q in load_questions():
        chunks = retrieve(q["question"], model, conn, k=DEFAULT_K)
        prompt = SYSTEM_PROMPT + build_prompt(q["question"], chunks)
        estimates.append(len(prompt) / CHARS_PER_TOKEN)

print(f"Estimated RAG input tokens per question (k={DEFAULT_K}, {len(estimates)} questions):")
print(f"  mean {mean(estimates):.0f}, min {min(estimates):.0f}, max {max(estimates):.0f}")
print("Agent measured: ~7,000 input tokens per question (runs r1, r2)")