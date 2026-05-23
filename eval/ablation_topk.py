"""
ablation_topk.py - Ablation over the retrieval depth k.

Re-runs the standard chunk-ID-based retrieval evaluation for several values of
k (1, 3, 5, 10) on the current (structural) index and gold set, reporting
Hit@k and MRR for each. Shows the trade-off between retrieving more chunks
(higher recall) and ranking quality.

Run from the project root with:
    python eval/ablation_topk.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from retrieve import Retriever

GOLD_FILE = Path("eval/gold_set.jsonl")
K_VALUES = [1, 3, 5, 10]


def load_gold():
    items = []
    with GOLD_FILE.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def evaluate_at_k(gold, retriever, k):
    hits = 0
    reciprocal_ranks = []
    evaluated = 0
    for q in gold:
        relevant = set(q.get("relevant_chunk_ids", []))
        if not relevant:
            continue
        evaluated += 1
        results = retriever.retrieve(q["question"], top_k=k)
        retrieved_ids = [r["chunk"]["chunk_id"] for r in results]
        if any(cid in relevant for cid in retrieved_ids):
            hits += 1
        rr = 0.0
        for rank, cid in enumerate(retrieved_ids, start=1):
            if cid in relevant:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)
    return {
        "k": k,
        "hit": hits / evaluated if evaluated else 0.0,
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0,
    }


def main():
    gold = load_gold()
    retriever = Retriever()

    print("=" * 60)
    print("ABLATION: retrieval depth (top-k)")
    print("=" * 60)
    print(f"{'k':>4} | {'Hit@k':>8} | {'MRR':>8}")
    print("-" * 28)
    for k in K_VALUES:
        m = evaluate_at_k(gold, retriever, k)
        print(f"{m['k']:>4} | {m['hit']*100:>7.1f}% | {m['mrr']:>8.3f}")
    print("=" * 60)
    print("Note: MRR is independent of k once the first relevant chunk is found;")
    print("Hit@k can only increase (or stay equal) as k grows.")


if __name__ == "__main__":
    main()
