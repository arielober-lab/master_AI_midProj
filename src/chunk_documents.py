"""
chunk_documents.py - Split documents into chunks using one of two strategies.

Reads documents.jsonl and produces chunks.jsonl. Supports two chunking
strategies, selectable via a command-line argument:

  fixed       - fixed-size chunks (600 chars) with overlap (100 chars),
                cutting at word boundaries
  structural  - one chunk per ingredient block (uses the natural structure
                of the IncIDecoder documents)

Run with:
    python src/chunk_documents.py fixed
    python src/chunk_documents.py structural
"""

import json
import sys
from pathlib import Path

DOCUMENTS_FILE = Path("data/processed/documents.jsonl")
OUTPUT_FILE = Path("data/processed/chunks.jsonl")

# Fixed-size strategy parameters
CHUNK_SIZE = 600      # target characters per chunk
CHUNK_OVERLAP = 100   # characters of overlap between consecutive chunks


def load_documents():
    docs = []
    with DOCUMENTS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def chunk_fixed(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split text into fixed-size chunks with overlap, breaking at word
    boundaries so we never cut a word in half.
    """
    words = text.split()
    chunks = []
    current = []
    current_len = 0

    for word in words:
        current.append(word)
        current_len += len(word) + 1  # +1 for the space

        if current_len >= chunk_size:
            chunks.append(" ".join(current))
            # Start the next chunk with an overlap: keep the last few words
            overlap_words = []
            overlap_len = 0
            for w in reversed(current):
                overlap_len += len(w) + 1
                overlap_words.insert(0, w)
                if overlap_len >= overlap:
                    break
            current = overlap_words
            current_len = overlap_len

    # Add the final chunk if anything remains
    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_structural(text):
    """
    Split text by ingredient blocks. In our documents, the section after
    'Ingredient details:' contains one ingredient per block, separated by
    double newlines. We keep the product header with the first chunk.
    """
    # Split off the header (Product + ingredient list) from the details
    if "Ingredient details:" in text:
        header, details = text.split("Ingredient details:", 1)
    else:
        header, details = "", text

    # Each ingredient block is separated by a blank line
    blocks = [b.strip() for b in details.split("\n\n") if b.strip()]

    chunks = []
    # Keep the product header as its own first chunk (useful context)
    if header.strip():
        chunks.append(header.strip())
    chunks.extend(blocks)
    return chunks


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("fixed", "structural"):
        print('Usage: python src/chunk_documents.py [fixed|structural]')
        sys.exit(1)

    strategy = sys.argv[1]
    documents = load_documents()
    print(f"Loaded {len(documents)} documents")
    print(f"Chunking strategy: {strategy}\n")

    all_chunks = []
    for doc in documents:
        if strategy == "fixed":
            texts = chunk_fixed(doc["text"])
        else:
            texts = chunk_structural(doc["text"])

        for i, chunk_text in enumerate(texts):
            all_chunks.append({
                "chunk_id": f"{doc['doc_id']}_{strategy}_{i}",
                "text": chunk_text,
                "metadata": {
                    "doc_id": doc["doc_id"],
                    "source": doc["metadata"]["source"],
                    "product_name": doc["metadata"]["product_name"],
                    "chunk_index": i,
                    "strategy": strategy,
                },
            })
        print(f"  {doc['metadata']['product_name']}: {len(texts)} chunks")

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # Stats
    sizes = [len(c["text"]) for c in all_chunks]
    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Chunk size (chars): min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)//len(sizes)}")
    print(f"Wrote chunks to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
