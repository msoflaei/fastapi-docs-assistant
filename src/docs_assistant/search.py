"""Find the chunks most similar in meaning to a question."""
import sys                                          # to read the question from the command line

import numpy as np

from docs_assistant.embed import EMB_FILE, load_chunks, load_model   # reuse what embed.py defined

# bge models work best when questions (but not documents) start with this instruction.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def search(query: str, model, vectors: np.ndarray, chunks: list[dict], k: int = 5) -> list[tuple[float, dict]]:
    """Return the k chunks most similar to the query, with their similarity scores."""
    q = model.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]   # the question's vector
    # Cosine similarity measures how closely two vectors point the same way (1 = identical meaning).
    # Because all vectors have length 1, it's simply the dot product, and "vectors @ q"
    # computes it for all 1059 chunks in one go.
    scores = vectors @ q
    top = np.argsort(-scores)[:k]                   # positions of the k highest scores
    return [(float(scores[i]), chunks[i]) for i in top]


def main() -> None:
    query = " ".join(sys.argv[1:]) or "How do I run my app in development mode?"
    model = load_model()
    vectors = np.load(EMB_FILE)
    chunks = load_chunks()

    print(f"Question: {query}\n")
    for score, chunk in search(query, model, vectors, chunks):
        print(f"{score:.3f}  {chunk['section']}")
        print(f"       {chunk['url']}")


if __name__ == "__main__":
    main()