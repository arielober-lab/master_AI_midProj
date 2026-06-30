"""
fetch_pubmed_append.py - Fetch PubMed articles on several topics and APPEND them
to the existing pubmed_documents.jsonl, WITHOUT overwriting what's already there.

Unlike fetch_pubmed.py (which writes with "w" and replaces the file), this script:
  1. Loads the existing pubmed_documents.jsonl (if any)
  2. Fetches articles for each query in QUERIES
  3. Merges new articles in, de-duplicating by doc_id (pubmed_<PMID>)
  4. Writes the combined set back

After running this, continue the normal pipeline:
    python src/chunk_pubmed.py     # rebuild all pubmed chunks from the merged file
    python src/build_index.py      # rebuild the combined FAISS index

Run from the project root:
    python src/fetch_pubmed_append.py
"""

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
OUTPUT_FILE = Path("data/processed/pubmed_documents.jsonl")
HEADERS = {"User-Agent": "aloe-rag-student-project/1.0"}

# Topics to cover, with how many articles to fetch per topic.
QUERIES = [
    ("aloe vera iron absorption", 5),
    ("aloe vera nutrient absorption bioavailability", 5),
    ("aloe vera digestion gut health", 5),
    ("aloe vera blood sugar glucose", 5),
    ("aloe vera immune system", 5),
    ("aloe vera cholesterol lipid", 5),
]


def search_pubmed(query: str, max_results: int) -> list:
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": str(max_results),
        "retmode": "json",
    }
    r = requests.get(ESEARCH, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()["esearchresult"]["idlist"]


def fetch_abstracts(pmids: list) -> list:
    if not pmids:
        return []
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
    return docs


def load_existing(path: Path) -> dict:
    """Load existing docs into a dict keyed by doc_id (for de-duplication)."""
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

    added = 0
    for query, n in QUERIES:
        print(f"\nSearching: {query!r} (max {n})")
        try:
            pmids = search_pubmed(query, n)
        except Exception as e:
            print(f"  search error: {e}")
            continue
        print(f"  found {len(pmids)} PMIDs")
        time.sleep(0.5)  # polite to NCBI

        try:
            docs = fetch_abstracts(pmids)
        except Exception as e:
            print(f"  fetch error: {e}")
            continue

        for d in docs:
            if d["doc_id"] not in existing:
                existing[d["doc_id"]] = d
                added += 1
        print(f"  retrieved {len(docs)} abstracts; new so far this run: {added}")
        time.sleep(0.5)

    # Write the merged set back
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
