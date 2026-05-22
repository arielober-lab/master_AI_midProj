"""
dedupe_gold_set.py - Clean up the auto-generated gold set draft.

Merges near-duplicate ingredient questions (same ingredient written with
different formatting across products) into one question, combining their
relevant_chunk_ids. Produces a cleaner gold_set.jsonl.

Run with:
    python src/dedupe_gold_set.py
"""

import json
import re
from pathlib import Path

DRAFT_FILE = Path("eval/gold_set_draft.jsonl")
OUTPUT_FILE = Path("eval/gold_set.jsonl")


def normalize_ingredient(question):
    """Strip parentheticals and formatting so the same ingredient maps together."""
    # Extract the ingredient name from "What does {ingredient} do?"
    m = re.match(r"What does (.+) do\?", question)
    if not m:
        return question.lower()
    name = m.group(1)
    # Remove anything in parentheses
    name = re.sub(r"\s*\([^)]*\)", "", name)
    # Normalize apostrophes and whitespace, lowercase
    name = name.replace("\u2019", "'").strip().lower()
    name = re.sub(r"\s+", " ", name)
    return name


def main():
    items = []
    with DRAFT_FILE.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    print(f"Loaded {len(items)} draft questions")

    # Separate by type
    ingredient_qs = [q for q in items if q["type"] == "ingredient_function"]
    product_qs = [q for q in items if q["type"] == "product_ingredients"]

    # Merge ingredient questions by normalized name
    merged = {}
    for q in ingredient_qs:
        key = normalize_ingredient(q["question"])
        if key not in merged:
            # Use a clean question form (capitalize the normalized name)
            clean_name = key.title()
            merged[key] = {
                "question": f"What does {clean_name} do?",
                "relevant_chunk_ids": list(q["relevant_chunk_ids"]),
                "relevant_product": "multiple",
                "type": "ingredient_function",
            }
        else:
            # Combine the relevant chunk ids
            merged[key]["relevant_chunk_ids"].extend(q["relevant_chunk_ids"])

    # Deduplicate chunk id lists
    for q in merged.values():
        q["relevant_chunk_ids"] = sorted(set(q["relevant_chunk_ids"]))

    cleaned = list(merged.values()) + product_qs

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for q in cleaned:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"\nAfter dedup:")
    print(f"  unique ingredient questions: {len(merged)}")
    print(f"  product questions: {len(product_qs)}")
    print(f"  total: {len(cleaned)}")
    print(f"\nWrote cleaned gold set to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
