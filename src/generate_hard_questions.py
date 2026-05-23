"""
generate_hard_questions.py - Generate paraphrased ("hard") gold questions.

Unlike the auto-generated questions (which contain the ingredient name and are
therefore easy to retrieve), these questions ask about an ingredient's FUNCTION
without naming it. This tests whether the system can retrieve semantically.

Example:
  Easy:  "What does Allantoin do?"          (names the ingredient)
  Hard:  "Which ingredient soothes skin?"   (only describes the function)

The relevant chunks for each question are all chunks whose "What-it-does" field
mentions that function.

Run with:
    python src/generate_hard_questions.py
"""

import json
import re
from collections import defaultdict
from pathlib import Path

CHUNKS_FILE = Path("data/processed/chunks.jsonl")
OUTPUT_FILE = Path("eval/hard_questions.jsonl")

# Known cosmetic functions -> a natural question that does NOT name any ingredient
FUNCTION_QUESTIONS = {
    "soothing": "Which ingredient helps soothe irritated skin?",
    "moisturizer": "Which ingredient works as a moisturizer?",
    "humectant": "Which ingredient draws moisture into the skin?",
    "emollient": "Which ingredient softens the skin as an emollient?",
    "antioxidant": "Which ingredient acts as an antioxidant?",
    "preservative": "Which ingredient is used as a preservative?",
    "solvent": "Which ingredient acts as a solvent in the formula?",
    "buffering": "Which ingredient is used to adjust the pH?",
    "chelating": "Which ingredient works as a chelating agent?",
    "perfuming": "Which ingredient is used for fragrance?",
    "skin brightening": "Which ingredient helps brighten the skin?",
    "viscosity controlling": "Which ingredient controls the product's viscosity?",
    "sunscreen": "Which ingredient protects the skin from the sun?",
    "emulsion stabilising": "Which ingredient stabilizes the emulsion?",
}


def main():
    chunks = []
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    # Map each function to the chunks whose "What-it-does" mentions it
    func_to_chunks = defaultdict(list)
    for c in chunks:
        m = re.search(r"What-it-does:\s*(.{0,60})", c["text"])
        if not m:
            continue
        whatitdoes = m.group(1).lower()
        for func in FUNCTION_QUESTIONS:
            if func in whatitdoes:
                func_to_chunks[func].append(c["chunk_id"])

    hard = []
    for func, question in FUNCTION_QUESTIONS.items():
        chunk_ids = func_to_chunks.get(func, [])
        if not chunk_ids:
            continue  # skip functions with no matching chunk in this corpus
        hard.append({
            "question": question,
            "relevant_chunk_ids": sorted(set(chunk_ids)),
            "relevant_product": "multiple",
            "type": "paraphrased_hard",
        })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for q in hard:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"Generated {len(hard)} hard (paraphrased) questions:")
    for q in hard:
        print(f"  - {q['question']}  ({len(q['relevant_chunk_ids'])} relevant chunks)")
    print(f"\nWrote to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
