"""
eval_answers.py - Answer-level evaluation of the RAG system.

Two parts:
  1. Refusal accuracy: runs every refusal_expected question and checks that the
     system actually declines (the anti-hallucination test).
  2. Answer quality sample: runs a small, diverse sample of factual questions
     and prints question + answer + sources for manual faithfulness review.

Uses the LLM (one API call per question). The Gemini free tier allows only
5 requests/minute, so we space calls ~14s apart and retry on rate-limit errors.

Run from the project root with:
    python eval/eval_answers.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from rag_system import RAGSystem

GOLD_FILE = Path("eval/gold_set.jsonl")
DELAY_SECONDS = 14         # spacing between calls; keeps us under 5 requests/min
SAMPLE_PER_TYPE = 3        # factual questions to sample per type
MAX_RETRIES = 2            # retries if a rate-limit error still occurs

REFUSAL_INDICATORS = [
    "don't have enough information", "do not have enough information",
    "does not have", "doesn't have", "no information", "cannot answer",
    "not contain", "isn't enough information", "is not enough information",
    "not provide", "no mention", "not mention", "does not specify",
    "not specified", "not available in the",
]


def load_gold():
    items = []
    with GOLD_FILE.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def is_refusal(answer):
    a = answer.lower()
    return any(ind in a for ind in REFUSAL_INDICATORS)


def has_citation(answer):
    return "[" in answer and "]" in answer


def answer_with_retry(rag, question):
    """Call the RAG system, retrying with a pause if we hit the rate limit."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            return rag.answer(question)
        except Exception as e:
            msg = str(e).lower()
            is_rate_limit = "resource_exhausted" in msg or "429" in msg or "quota" in msg
            if is_rate_limit and attempt < MAX_RETRIES:
                print("   (rate limit hit - waiting 60s before retry...)")
                time.sleep(60)
                continue
            raise


def main():
    gold = load_gold()
    rag = RAGSystem()

    refusal_qs = [q for q in gold if q.get("type") == "refusal_expected"]

    sample = []
    for qtype in ("ingredient_function", "product_ingredients", "paraphrased_hard"):
        of_type = [q for q in gold if q.get("type") == qtype]
        sample.extend(of_type[:SAMPLE_PER_TYPE])

    # ---- Part 1: Refusal accuracy ----
    print("=" * 60)
    print("PART 1: REFUSAL EVALUATION (anti-hallucination)")
    print("=" * 60)
    correct = 0
    for q in refusal_qs:
        result = answer_with_retry(rag, q["question"])
        refused = is_refusal(result["answer"])
        correct += 1 if refused else 0
        print(f"\nQ: {q['question']}")
        print(f"   Refused correctly: {refused}")
        print(f"   Answer: {result['answer'][:160]}")
        time.sleep(DELAY_SECONDS)
    if refusal_qs:
        print(f"\n>>> Refusal accuracy: {correct}/{len(refusal_qs)} "
              f"({correct/len(refusal_qs)*100:.0f}%)")

    # ---- Part 2: Answer quality sample ----
    print("\n" + "=" * 60)
    print("PART 2: ANSWER QUALITY SAMPLE (review manually)")
    print("=" * 60)
    no_citation = 0
    for q in sample:
        result = answer_with_retry(rag, q["question"])
        cited = has_citation(result["answer"])
        if not cited:
            no_citation += 1
        print(f"\n[{q.get('type')}] Q: {q['question']}")
        print(f"  A: {result['answer']}")
        print(f"  Sources: {', '.join(result['sources'])}")
        print(f"  Has citation: {cited}")
        time.sleep(DELAY_SECONDS)
    print(f"\n>>> Answers with citations: {len(sample)-no_citation}/{len(sample)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
