"""
run_eval.py - Evaluate the RAG system's retrieval quality against the gold set.

Computes Hit@k and MRR over all gold questions that have known relevant chunks,
broken down by question type, and reports any misses for error analysis.

Run from the project root with:
    python eval/run_eval.py
"""

import json
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


def evaluate_retrieval(gold, retriever, top_k=TOP_K):
    hits = 0
    reciprocal_ranks = []
    evaluated = 0
    per_type = {}
    misses = []  # questions where no relevant chunk was in top-k

    for q in gold:
        relevant = set(q.get("relevant_chunk_ids", []))
        if not relevant:
            continue
        evaluated += 1
        qtype = q.get("type", "unknown")
        per_type.setdefault(qtype, {"n": 0, "hits": 0})
        per_type[qtype]["n"] += 1

        results = retriever.retrieve(q["question"], top_k=top_k)
        retrieved_ids = [r["chunk"]["chunk_id"] for r in results]

        if any(cid in relevant for cid in retrieved_ids):
            hits += 1
            per_type[qtype]["hits"] += 1
        else:
            # Record the miss with what was retrieved instead
            misses.append({
                "question": q["question"],
                "type": qtype,
                "expected": sorted(relevant),
                "got_top1": results[0]["chunk"]["metadata"].get("product_name") if results else None,
                "got_top1_text": results[0]["chunk"]["text"][:90] if results else None,
            })

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
        "misses": misses,
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

    if metrics["misses"]:
        print("\n" + "=" * 60)
        print(f"MISSES ({len(metrics['misses'])}) - for error analysis")
        print("=" * 60)
        for m in metrics["misses"]:
            print(f"\n  Question: {m['question']}")
            print(f"  Type:     {m['type']}")
            print(f"  Expected chunks: {m['expected']}")
            print(f"  Got (rank 1):    [{m['got_top1']}] {m['got_top1_text']}...")
    else:
        print("\nNo misses.")
    print("=" * 60)


if __name__ == "__main__":
    main()
