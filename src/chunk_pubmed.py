"""
chunk_pubmed.py - Chunk the PubMed documents and merge them into chunks.jsonl.

PubMed abstracts are prose (no ingredient structure), so we chunk them with the
'fixed' strategy (reusing chunk_fixed from chunk_documents.py). The resulting
chunks are merged into chunks.jsonl alongside the IncIDecoder chunks, and the
index is then rebuilt from the combined file.

Idempotent: removes any existing pubmed chunks before adding fresh ones, so it
is safe to re-run.

Workflow:
    python src/chunk_documents.py structural   # IncIDecoder chunks -> chunks.jsonl
    python src/chunk_pubmed.py                  # merge PubMed chunks in
    python src/build_index.py                   # rebuild combined index
"""

import json
from pathlib import Path

from chunk_documents import chunk_fixed   # reuse the exact same fixed-chunking logic

PUBMED_FILE = Path("data/processed/pubmed_documents.jsonl")
CHUNKS_FILE = Path("data/processed/chunks.jsonl")


def load_jsonl(path):
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def main():
    # 1. Load existing chunks, drop any previous pubmed chunks (idempotent re-run)
    existing = load_jsonl(CHUNKS_FILE) if CHUNKS_FILE.exists() else []
    existing = [c for c in existing if c["metadata"].get("source") != "pubmed"]
    print(f"Existing non-pubmed chunks: {len(existing)}")

    # 2. Chunk the PubMed documents with the fixed strategy
    pubmed_docs = load_jsonl(PUBMED_FILE)
    print(f"PubMed documents: {len(pubmed_docs)}")

    pubmed_chunks = []
    for doc in pubmed_docs:
        texts = chunk_fixed(doc["text"])
        for i, chunk_text in enumerate(texts):
            pubmed_chunks.append({
                "chunk_id": f"{doc['doc_id']}_fixed_{i}",
                "text": chunk_text,
                "metadata": {
                    "doc_id": doc["doc_id"],
                    "source": "pubmed",
                    # use the article title as 'product_name' so citations read sensibly
                    "product_name": doc["metadata"].get("title", doc["doc_id"]),
                    "chunk_index": i,
                    "strategy": "fixed",
                },
            })
    print(f"PubMed chunks created: {len(pubmed_chunks)}")

    # 3. Merge and write back
    combined = existing + pubmed_chunks
    with CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for c in combined:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"\nTotal chunks now: {len(combined)} ({len(existing)} original + {len(pubmed_chunks)} pubmed)")
    print("Next step: python src/build_index.py")


if __name__ == "__main__":
    main()
