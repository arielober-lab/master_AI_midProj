"""
run_eval.py - Evaluate the RAG system's retrieval quality against the gold set.

Computes two standard retrieval metrics over all gold questions that have
known relevant chunks:
  - Hit@k  : fraction of questions where at least one relevant chunk is in top-k
  - MRR    : Mean Reciprocal Rank (1 / rank of the first relevant chunk)

This evaluation uses ONLY retrieval (no LLM calls), so it is fast and free.
Answer-level evaluation (including refusal accuracy) is handled separately.

Run from the project root with:
    python eval/run_eval.py
"""

import json
import sys
from pathlib import Path

# Make src/ importable so we can use the Retriever
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


def evaluate_retrieval(gold, retriever, top_k=TOP_K):
    hits = 0
    reciprocal_ranks = []
    evaluated = 0
    per_type = {}  # track hit rate by question type

    for q in gold:
        relevant = set(q.get("relevant_chunk_ids", []))
        if not relevant:
            continue  # refusal questions have no relevant chunks; skip for retrieval
        evaluated += 1
        qtype = q.get("type", "unknown")
        per_type.setdefault(qtype, {"n": 0, "hits": 0})
        per_type[qtype]["n"] += 1

        results = retriever.retrieve(q["question"], top_k=top_k)
        retrieved_ids = [r["chunk"]["chunk_id"] for r in results]

        # Hit@k
        if any(cid in relevant for cid in retrieved_ids):
            hits += 1
            per_type[qtype]["hits"] += 1

        # MRR
        rr = 0.0
        for rank, cid in enumerate(retrieved_ids, start=1):
            if cid in relevant:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    return {
        "evaluated": evaluated,
        "hit_at_k": hits / evaluated if evaluated else 0.0,
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "per_type": per_type,
    }


def main():
    gold = load_gold()
    print(f"Loaded {len(gold)} gold questions")

    retriever = Retriever()

    print(f"\nEvaluating retrieval (top_k={TOP_K})...\n")
    metrics = evaluate_retrieval(gold, retriever)

    print("=" * 60)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Questions evaluated:  {metrics['evaluated']}")
    print(f"Hit@{TOP_K}:              {metrics['hit_at_k']:.3f}  ({metrics['hit_at_k']*100:.1f}%)")
    print(f"MRR:                  {metrics['mrr']:.3f}")
    print("\nBy question type:")
    for qtype, stats in metrics["per_type"].items():
        rate = stats["hits"] / stats["n"] if stats["n"] else 0
        print(f"  {qtype}: {stats['hits']}/{stats['n']} hits ({rate*100:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
