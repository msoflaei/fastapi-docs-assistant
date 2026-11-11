"""Fine-tune the embedding model on FastAPI docs, using section headings as training queries.

Training pairs: (section breadcrumb, section text). No evaluation questions are used,
so the retrieval evaluation measures genuine improvement, not memorization.

Usage: python -m docs_assistant.finetune
"""
import json
import random
from pathlib import Path

from docs_assistant.chunk import split_sections
from docs_assistant.search import QUERY_PREFIX

DOCS_FILE = Path("data/processed/docs.jsonl")
BASE_MODEL = "BAAI/bge-small-en-v1.5"
OUTPUT_DIR = Path("models/bge-small-fastapi")


def build_pairs(docs: list[dict], min_words: int = 20, max_words: int = 300) -> list[dict]:
    """One (query, passage) pair per documentation section.

    The query is the section's breadcrumb (with the same prefix used at search time);
    the passage is the section's text. Very short sections and excluded pages are skipped.
    """
    pairs = []
    for doc in docs:
        if (doc["doc_id"] == "release-notes" or doc["doc_id"].startswith("reference/")
                or doc["doc_id"] == "_llm-test"):
            continue
        for section in split_sections(doc["text"]):
            words = "\n".join(section["lines"]).split()
            if len(words) < min_words:
                continue
            query = section["breadcrumb"] or doc["title"]
            pairs.append({"anchor": QUERY_PREFIX + query, "positive": " ".join(words[:max_words])})
    return pairs


def main() -> None:
    # Heavy imports go inside main(), so tests can import build_pairs without
    # installing the training libraries (CI doesn't install them).
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.sentence_transformer import losses
    from sentence_transformers.sentence_transformer.training_args import BatchSamplers

    with DOCS_FILE.open(encoding="utf-8") as f:
        docs = [json.loads(line) for line in f]
    pairs = build_pairs(docs)
    random.seed(42)                  # fixed seed: the same shuffle every run, for reproducibility
    random.shuffle(pairs)
    print(f"{len(pairs)} training pairs. Example:\n  query:   {pairs[0]['anchor']}\n"
          f"  passage: {pairs[0]['positive'][:150]}...\n")

    model = SentenceTransformer(BASE_MODEL, model_kwargs={"attn_implementation": "eager"})
    model.max_seq_length = 256       # shorter texts during training, to fit in memory
    train_dataset = Dataset.from_list(pairs)
    # Same loss as before (in-batch negatives), but it processes the batch of 32 in pieces of 8
    # and combines the results ("gradient caching"): as good as a batch of 32, with the memory of 8.
    loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=8)

    args = SentenceTransformerTrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=1,          # one pass over the data: enough to adapt without forgetting
        per_device_train_batch_size=32,
        learning_rate=2e-5,          # small steps, so general knowledge isn't overwritten
        warmup_steps=0.1,            # warm up over the first 10% of training (a float means a ratio)
        batch_sampler=BatchSamplers.NO_DUPLICATES,   # no repeated texts within a batch
        logging_steps=10,
        save_strategy="no",
        report_to="none",            # don't send training logs to external services
        seed=42,
    )
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train_dataset, loss=loss)
    trainer.train()

    model.save(str(OUTPUT_DIR))
    print(f"\nSaved fine-tuned model -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()