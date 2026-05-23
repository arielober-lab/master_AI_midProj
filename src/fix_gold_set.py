"""
fix_gold_set.py - Repair a gold_set.jsonl whose JSON objects got concatenated
onto the same line (e.g. by PowerShell Add-Content line-ending mismatches).

Reads ALL JSON objects regardless of how they are split across lines, then
rewrites them cleanly, one object per line with proper newlines.

Run with:
    python src/fix_gold_set.py
"""

import json
from pathlib import Path

GOLD_FILE = Path("eval/gold_set.jsonl")


def main():
    raw = GOLD_FILE.read_text(encoding="utf-8-sig").strip()

    decoder = json.JSONDecoder()
    objects = []
    idx = 0
    while idx < len(raw):
        # skip any whitespace (spaces, tabs, newlines) between objects
        while idx < len(raw) and raw[idx] in " \t\r\n":
            idx += 1
        if idx >= len(raw):
            break
        obj, end = decoder.raw_decode(raw, idx)
        objects.append(obj)
        idx = end

    # Rewrite cleanly: one object per line, UTF-8, no BOM
    with GOLD_FILE.open("w", encoding="utf-8") as f:
        for obj in objects:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Repaired gold set: {len(objects)} questions, one per line.")
    # Show type breakdown so we can confirm everything survived
    from collections import Counter
    types = Counter(o.get("type", "unknown") for o in objects)
    for t, n in types.items():
        print(f"  {t}: {n}")


if __name__ == "__main__":
    main()
