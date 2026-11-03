"""Compute an embedding vector for every chunk and save them to disk."""
import json                                         # to read the chunks file
from pathlib import Path                            # file paths

import numpy as np                                  # fast arrays of numbers; our vectors live in one
from sentence_transformers import SentenceTransformer   # loads and runs embedding models

CHUNKS_FILE = Path("data/processed/chunks.jsonl")   # output of the chunking step
EMB_FILE = Path("data/processed/embeddings.npy")    # where the vectors will be saved
MODEL_NAME = "BAAI/bge-small-en-v1.5"               # open-source embedding model on Hugging Face


def load_chunks() -> list[dict]:
    """Read all chunks from the JSONL file, in order."""
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_model() -> SentenceTransformer:
    """Load the embedding model (downloaded automatically the first time)."""
    return SentenceTransformer(MODEL_NAME)


def main() -> None:
    chunks = load_chunks()
    texts = [c["text"] for c in chunks]             # just the text of each chunk
    model = load_model()

    # Measure how many chunks are longer than the model can read.
    limit = model.max_seq_length                    # 512 tokens for this model
    lengths = [len(model.tokenizer(t)["input_ids"]) for t in texts]
    too_long = sum(n > limit for n in lengths)
    print(f"Model reads up to {limit} tokens; {too_long} of {len(texts)} chunks are longer (will be truncated)")

    # Turn every text into a vector. batch_size=32 processes 32 texts at a time.
    # normalize_embeddings=True scales every vector to length 1, which makes
    # comparing them simple (see search.py).
    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    np.save(EMB_FILE, vectors.astype(np.float32))   # save as a compact binary file
    # Row i of the array is the vector for line i of chunks.jsonl, so order matters.
    print(f"Saved embeddings with shape {vectors.shape} -> {EMB_FILE}")


if __name__ == "__main__":
    main()