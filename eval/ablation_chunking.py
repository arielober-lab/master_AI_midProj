"""
ablation_chunking.py - Content-based retrieval evaluation for comparing
chunking strategies (fixed vs structural).

Because chunk IDs differ between chunking strategies, this script uses a
CONTENT-based relevance criterion instead of matching chunk IDs: a question
is a "hit" if any of the top-k retrieved chunks contains the target ingredient
name in its text. This criterion is valid across any chunking strategy.

Evaluates only ingredient_function questions (where the target ingredient is
unambiguous and present in the question text).

Workflow to run the ablation:
    # 1. Structural (current default)
    python src/chunk_documents.py structural
    python src/build_index.py
    python eval/ablation_chunking.py        # -> note the Hit@5

    # 2. Fixed
    python src/chunk_documents.py fixed
    python src/build_index.py
    python eval/ablation_chunking.py        # -> note the Hit@5

    # 3. Restore structural for the rest of the project
    python src/chunk_documents.py structural
    python src/build_index.py
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from retrieve import Retriever

GOLD_FILE = Path("eval/gold_set.jsonl")
TOP_K = 5


def load_gold():
    items = []
    with GOLD_FILE.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def target_ingredient(question):
    """Extract the ingredient name from 'What does {X} do?'."""
    m = re.match(r"What does (.+?) do\?", question)
    return m.group(1).strip().lower() if m else None


def main():
    gold = load_gold()
    retriever = Retriever()

    # Only ingredient_function questions have an unambiguous content target
    questions = [q for q in gold if q.get("type") == "ingredient_function"]

    hits = 0
    evaluated = 0
    misses = []
    for q in questions:
        target = target_ingredient(q["question"])
        if not target:
            continue
        evaluated += 1
        results = retriever.retrieve(q["question"], top_k=TOP_K)
        texts = [r["chunk"]["text"].lower() for r in results]
        if any(target in t for t in texts):
            hits += 1
        else:
            misses.append(q["question"])

    print("=" * 60)
    print(f"CONTENT-BASED RETRIEVAL  (top_k={TOP_K})")
    print(f"Index currently has {retriever.index.ntotal} vectors")
    print("=" * 60)
    print(f"Questions evaluated: {evaluated}")
    print(f"Content Hit@{TOP_K}:     {hits}/{evaluated} = {hits/evaluated*100:.1f}%")
    if misses:
        print(f"\nMisses ({len(misses)}):")
        for m in misses[:15]:
            print(f"  - {m}")
    print("=" * 60)


if __name__ == "__main__":
    main()
