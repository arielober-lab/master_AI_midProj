"""
build_index.py - Build a FAISS index from chunks.jsonl.

Reads chunks from data/processed/chunks.jsonl (produced by chunk_documents.py),
embeds them with the e5 model, builds a FAISS IndexFlatL2, and saves it to
data/processed/faiss.index.

Run with:
    python src/build_index.py
"""

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ============================================================
# Configuration
# ============================================================
CHUNKS_FILE = Path("data/processed/chunks.jsonl")   # input: chunker output
INDEX_FILE = Path("data/processed/faiss.index")     # output: the vector index
MODEL_NAME = "intfloat/multilingual-e5-small"

# ============================================================
# Step 1: Load the embedding model
# ============================================================
print(f"Loading embedding model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)
print("Model loaded successfully.\n")

# ============================================================
# Step 2: Read all chunks
# ============================================================
print(f"Reading chunks from: {CHUNKS_FILE}")
chunks = []
with CHUNKS_FILE.open("r", encoding="utf-8") as f:
    for line_number, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            chunks.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"  Skipping malformed line {line_number}: {e}")

if not chunks:
    raise RuntimeError(f"No chunks loaded from {CHUNKS_FILE}. Run chunk_documents.py first.")

print(f"Loaded {len(chunks)} chunks.\n")

# ============================================================
# Step 3: Embed all chunks (e5 convention: prefix with "passage: ")
# ============================================================
print("Embedding all chunks...")
prefixed_texts = [f"passage: {chunk['text']}" for chunk in chunks]
embeddings = model.encode(
    prefixed_texts,
    show_progress_bar=True,
    convert_to_numpy=True,
).astype(np.float32)
print(f"Embeddings shape: {embeddings.shape}\n")

# ============================================================
# Step 4: Build the FAISS index
# ============================================================
embedding_dim = embeddings.shape[1]
print(f"Building FAISS IndexFlatL2 with dimension {embedding_dim}")
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings)
print(f"Index now contains {index.ntotal} vectors.\n")

# ============================================================
# Step 5: Save the index to disk
# ============================================================
INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
print(f"Saving FAISS index to: {INDEX_FILE}")
faiss.write_index(index, str(INDEX_FILE))

print("\n" + "=" * 60)
print(f"Build complete. {index.ntotal} chunks indexed and saved.")
print("=" * 60)
