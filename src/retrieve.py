"""
retrieve.py - Query-time retrieval against the FAISS index.

This module loads a pre-built FAISS index and its paired chunks, then exposes
a retrieve() function that takes a user question and returns the most relevant
chunks. This is the "Querying phase" half of the RAG pipeline.

Usage as a script (for testing):
    python src/retrieve.py "What product uses MSM for joint flexibility?"

Usage as a module (what generation.py will do):
    from retrieve import Retriever
    retriever = Retriever()
    results = retriever.retrieve("your question", top_k=5)
"""

import json
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ============================================================
# Configuration
# ============================================================
INDEX_FILE = Path("data/processed/faiss.index")
CHUNKS_FILE = Path("data/processed/chunks.jsonl")
MODEL_NAME = "intfloat/multilingual-e5-small"


class Retriever:
    """
    Holds the embedding model, the FAISS index, and the chunks in memory.

    We load everything once when the Retriever is created (this is the slow
    part, done at startup), then each retrieve() call is fast.
    """

    def __init__(self):
        print("Initializing retriever...")

        # Load the embedding model (same one used to build the index)
        print(f"  Loading model: {MODEL_NAME}")
        self.model = SentenceTransformer(MODEL_NAME)

        # Load the FAISS index from disk
        print(f"  Loading index: {INDEX_FILE}")
        if not INDEX_FILE.exists():
            raise FileNotFoundError(
                f"Index file not found at {INDEX_FILE}. "
                f"Run build_index.py first to create it."
            )
        self.index = faiss.read_index(str(INDEX_FILE))

        # Load the chunks (so we can map an index position back to its text)
        print(f"  Loading chunks: {CHUNKS_FILE}")
        self.chunks = []
        with CHUNKS_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.chunks.append(json.loads(line))

        # Safety check: the index and chunks must be the same length, or the
        # positional mapping between them is broken.
        if self.index.ntotal != len(self.chunks):
            raise RuntimeError(
                f"Mismatch: index has {self.index.ntotal} vectors but "
                f"chunks file has {len(self.chunks)} entries. "
                f"Rebuild the index so they stay in sync."
            )

        print(f"  Ready. {self.index.ntotal} chunks indexed.\n")

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Find the top_k chunks most relevant to the query.

        Returns a list of dicts, each containing the chunk and its distance:
            [{"chunk": {...}, "distance": 0.26, "rank": 1}, ...]
        """
        # e5 convention: prefix queries with "query: "
        query_vec = self.model.encode(
            [f"query: {query}"],
            convert_to_numpy=True,
        ).astype(np.float32)

        # Search the index. Returns distances and positions of the nearest k.
        distances, positions = self.index.search(query_vec, top_k)

        # Build the result list, mapping each position back to its chunk
        results = []
        for rank, (pos, dist) in enumerate(zip(positions[0], distances[0]), start=1):
            # FAISS returns -1 for positions if fewer than top_k vectors exist
            if pos == -1:
                continue
            results.append({
                "rank": rank,
                "distance": float(dist),
                "chunk": self.chunks[pos],
            })

        return results


# ============================================================
# Command-line testing
# ============================================================
def main():
    if len(sys.argv) < 2:
        print('Usage: python src/retrieve.py "your question here"')
        sys.exit(1)

    query = sys.argv[1]

    retriever = Retriever()

    print(f"Query: '{query}'\n")
    results = retriever.retrieve(query, top_k=3)

    print(f"Top {len(results)} results:")
    for r in results:
        chunk = r["chunk"]
        print(f"\n  Rank {r['rank']}: distance={r['distance']:.4f}")
        print(f"    chunk_id: {chunk['chunk_id']}")
        print(f"    product:  {chunk['metadata'].get('product_name', 'N/A')}")
        print(f"    text:     {chunk['text'][:100]}...")


if __name__ == "__main__":
    main()
