"""
generate_gold_set.py - Auto-generate a draft gold set from the chunks.

For each chunk we know the source, so we can generate questions whose
relevant chunk is known by construction. This produces a draft that the
human reviews and supplements with realistic manual questions.

Generates two question types:
  - ingredient_function:  "What does {ingredient} do?"  (one per unique ingredient)
  - product_ingredients:  "What are the ingredients in {product}?"

Run with:
    python src/generate_gold_set.py
"""

import json
from collections import defaultdict
from pathlib import Path

CHUNKS_FILE = Path("data/processed/chunks.jsonl")
OUTPUT_FILE = Path("eval/gold_set_draft.jsonl")


def load_chunks():
    chunks = []
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def extract_ingredient_name(text):
    """Ingredient chunks start with the ingredient name. Cut at the earliest marker."""
    delimiters = (" - goodie", " - superstar", " - icky", " Also-called",
                  " What-it-does", " Irritancy", " Comedogenicity")
    positions = [text.find(d) for d in delimiters if text.find(d) != -1]
    if positions:
        return text[:min(positions)].strip()
    # Fallback: first few words
    return " ".join(text.split()[:3]).strip()


def main():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    # Map each unique ingredient to all chunk_ids that describe it
    ingredient_to_chunks = defaultdict(list)
    product_header_chunks = {}  # product_name -> chunk_id of its header

    for chunk in chunks:
        text = chunk["text"]
        product = chunk["metadata"]["product_name"]
        chunk_id = chunk["chunk_id"]

        if text.startswith("Product:"):
            product_header_chunks[product] = chunk_id
        else:
            ingredient = extract_ingredient_name(text)
            if ingredient and len(ingredient) > 2:
                ingredient_to_chunks[ingredient].append(chunk_id)

    gold = []

    # Type 1: one question per unique ingredient (relevant = all its chunks)
    for ingredient, chunk_ids in sorted(ingredient_to_chunks.items()):
        gold.append({
            "question": f"What does {ingredient} do?",
            "relevant_chunk_ids": chunk_ids,
            "relevant_product": "multiple" if len(chunk_ids) > 1 else None,
            "type": "ingredient_function",
        })

    # Type 2: one question per product (relevant = its header chunk)
    for product, chunk_id in sorted(product_header_chunks.items()):
        gold.append({
            "question": f"What are the ingredients in {product}?",
            "relevant_chunk_ids": [chunk_id],
            "relevant_product": product,
            "type": "product_ingredients",
        })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for item in gold:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nGenerated {len(gold)} draft questions:")
    print(f"  ingredient_function: {sum(1 for g in gold if g['type'] == 'ingredient_function')}")
    print(f"  product_ingredients: {sum(1 for g in gold if g['type'] == 'product_ingredients')}")
    print(f"\nWrote draft to {OUTPUT_FILE}")
    print("\nSample questions:")
    for g in gold[:8]:
        print(f"  - {g['question']}")


if __name__ == "__main__":
    main()
