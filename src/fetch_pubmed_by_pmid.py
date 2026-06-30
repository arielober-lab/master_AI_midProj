"""
fetch_pubmed_by_pmid.py - Fetch SPECIFIC PubMed articles by exact PMID and
append them to pubmed_documents.jsonl (de-duplicated).

A topic query sometimes misses the most relevant article because PubMed ranks
popular articles first. When you know exactly which article you want, fetch it
by PMID. This script does that for a hand-picked list of absorption /
bioavailability studies.

Run from the project root:
    python src/fetch_pubmed_by_pmid.py

Then continue:
    python src/chunk_pubmed.py
    python src/build_index.py
"""

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
OUTPUT_FILE = Path("data/processed/pubmed_documents.jsonl")
HEADERS = {"User-Agent": "aloe-rag-student-project/1.0"}

# Hand-picked PMIDs of articles directly about aloe vera and nutrient
# absorption / bioavailability (found via targeted search).
TARGET_PMIDS = [
    "22435613",  # Aloe vera on bioavailability of vitamins C and B12, blood glucose, lipid profile
    "16323295",  # Aloe vera preparations on human bioavailability of vitamins C and E
]


def fetch_abstracts(pmids: list) -> list:
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    }
    r = requests.get(EFETCH, params=params, headers=HEADERS, timeout=60)
    r.raise_for_status()

    root = ET.fromstring(r.text)
    docs = []
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        pmid = pmid_el.text if pmid_el is not None else "unknown"

        title_el = article.find(".//ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else ""

        abstract_parts = []
        for ab in article.findall(".//Abstract/AbstractText"):
            label = ab.get("Label")
            text = "".join(ab.itertext()).strip()
            if text:
                abstract_parts.append(f"{label}: {text}" if label else text)
        abstract = "\n".join(abstract_parts)

        if not abstract:
            print(f"  WARN: PMID {pmid} has no abstract, skipping")
            continue

        journal_el = article.find(".//Journal/Title")
        journal = journal_el.text if journal_el is not None else ""
        year_el = article.find(".//PubDate/Year")
        year = year_el.text if year_el is not None else ""

        docs.append({
            "doc_id": f"pubmed_{pmid}",
            "text": f"Title: {title}\n\nAbstract:\n{abstract}",
            "metadata": {
                "source": "pubmed",
                "pmid": pmid,
                "title": title,
                "journal": journal,
                "year": year,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            },
        })
        print(f"  OK: PMID {pmid} - {title[:70]}")
    return docs


def load_existing(path: Path) -> dict:
    by_id = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    by_id[d["doc_id"]] = d
    return by_id


def main():
    existing = load_existing(OUTPUT_FILE)
    print(f"Existing PubMed documents: {len(existing)}")

    print(f"\nFetching {len(TARGET_PMIDS)} target articles by PMID...")
    docs = fetch_abstracts(TARGET_PMIDS)

    added = 0
    for d in docs:
        if d["doc_id"] not in existing:
            existing[d["doc_id"]] = d
            added += 1
        else:
            print(f"  (already present: {d['doc_id']})")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for d in existing.values():
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print("\n=== DONE ===")
    print(f"  New articles added: {added}")
    print(f"  Total PubMed documents now: {len(existing)}")
    print("\nNext steps:")
    print("  python src/chunk_pubmed.py")
    print("  python src/build_index.py")


if __name__ == "__main__":
    main()
