"""
parse_incidecoder.py - Extract clean structured text from IncIDecoder HTML files.

Reads raw HTML files from data/raw/incidecoder/, extracts the product name and
each ingredient's name, function, and description, then writes one clean
document per product to data/processed/documents.jsonl.

Run with:
    python src/parse_incidecoder.py
"""

import json
from pathlib import Path

from bs4 import BeautifulSoup

RAW_DIR = Path("data/raw/incidecoder")
OUTPUT_FILE = Path("data/processed/documents.jsonl")


def parse_html_file(html_path):
    """Parse a single IncIDecoder HTML file into a clean document dict."""
    with html_path.open("r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "lxml")

    # --- Product name (from the h1) ---
    h1 = soup.find("h1")
    product_name = h1.get_text(separator=" ", strip=True) if h1 else html_path.stem

    # --- Short ingredient list ---
    ingred_links = soup.find_all("a", class_="ingred-link")
    seen = set()
    ingredient_list = []
    for a in ingred_links:
        name = a.get_text(strip=True)
        if name and name not in seen:
            seen.add(name)
            ingredient_list.append(name)

    # --- Detailed ingredient blocks ---
    detail_parts = []
    for block in soup.find_all("div", class_="ingred-long"):
        title = block.find("a", class_="product-long-ingred-link")
        if not title:
            continue
        name = title.get_text(strip=True)
        funcs = [a.get_text(strip=True) for a in block.find_all("a", class_="product-long-ingredfunc-link")]
        full_text = block.get_text(separator=" ", strip=True)
        # Clean up multiple spaces
        full_text = " ".join(full_text.split())
        detail_parts.append(full_text)

    # --- Build the clean document text ---
    text_sections = [f"Product: {product_name}"]
    if ingredient_list:
        text_sections.append("Full ingredient list: " + ", ".join(ingredient_list))
    text_sections.append("Ingredient details:")
    text_sections.extend(detail_parts)
    clean_text = "\n\n".join(text_sections)

    doc_id = f"incidecoder_{html_path.stem}"
    return {
        "doc_id": doc_id,
        "text": clean_text,
        "metadata": {
            "source": "incidecoder.com",
            "product_name": product_name,
            "source_file": html_path.name,
            "num_ingredients": len(ingredient_list),
        },
    }


def main():
    html_files = sorted(RAW_DIR.glob("*.html"))
    print(f"Found {len(html_files)} HTML files in {RAW_DIR}")

    documents = []
    for html_path in html_files:
        # Skip brand index pages (they list products, not a single product)
        if html_path.stem in ("forever", "forever-living-products"):
            print(f"  Skipping brand index page: {html_path.name}")
            continue
        doc = parse_html_file(html_path)
        documents.append(doc)
        print(f"  Parsed {html_path.name}: {doc['metadata']['num_ingredients']} ingredients, {len(doc['text'])} chars")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(documents)} documents to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
