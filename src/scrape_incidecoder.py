"""
IncIDecoder Scraper for Forever Living Aloe Topical Products RAG Project

Downloads product pages from IncIDecoder.com for offline processing.
Respects robots.txt, rate-limits requests, and saves raw HTML for later parsing.

Usage:
    # Test mode: download just one page to verify everything works
    python scrape_incidecoder.py --test

    # Full mode: download all product pages
    python scrape_incidecoder.py --full

    # Single URL mode: download just one specific URL
    python scrape_incidecoder.py --url https://incidecoder.com/products/forever-living-aloe-vera-gelly

Output:
    - HTML files saved to data/raw/incidecoder/
    - Log file at data/raw/incidecoder/scrape_log.txt
    - Metadata file at data/raw/incidecoder/metadata.jsonl

Authors: [שמות שלכם]
Course: BGU AI Engineering Certificate Program - Mid-Course Assignment
"""

import argparse
import json
import logging
import time
import urllib.robotparser
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests


# ============================================================
# Configuration
# ============================================================

# Project structure - all paths are relative to project root
RAW_DATA_DIR = Path("data/raw/incidecoder")
LOG_FILE = RAW_DATA_DIR / "scrape_log.txt"
METADATA_FILE = RAW_DATA_DIR / "metadata.jsonl"

# Be a polite scraper: identify yourselves honestly so the site owner can
# block you specifically if needed, and so you appear in their logs as
# a known agent rather than as suspicious traffic.
USER_AGENT = (
    "AI-Engineering-Course-Project/1.0 "
    "(Academic research; mid-course RAG assignment; "
    "Course at Ben-Gurion University, Israel)"
)

# Wait this many seconds between requests to be polite to the server.
# 2 seconds means at most 30 requests per minute - well under what any
# normal website would consider abusive.
DELAY_BETWEEN_REQUESTS_SECONDS = 2.5

# Maximum number of retries on transient failures (network errors, 5xx)
MAX_RETRIES = 3

# How long to wait for a response from the server before giving up
REQUEST_TIMEOUT_SECONDS = 30

# Target URLs - the product pages to download
# These are the 9 pages we confirmed exist on IncIDecoder, plus the
# brand index pages which list the remaining products.
PRODUCT_URLS = [
    "https://incidecoder.com/products/forever-living-aloe-vera-gelly",
    "https://incidecoder.com/products/forever-living-products-aloe-activator",
    "https://incidecoder.com/products/forever-living-products-aloe-msm-gel",
    "https://incidecoder.com/products/forever-living-aloe-lips",
    "https://incidecoder.com/products/forever-living-products-aloe-ever-shield",
    "https://incidecoder.com/products/forever-living-aloe-first-r-spray",
    "https://incidecoder.com/products/forever-living-products-aloe-liquid-soap",
    "https://incidecoder.com/products/forever-living-aloe-sunscreen",
    "https://incidecoder.com/products/forever-living-aloe-propolis-creme",
]

# Brand index pages - download these to discover additional product URLs
BRAND_INDEX_URLS = [
    "https://incidecoder.com/brands/forever-living-products",
    "https://incidecoder.com/brands/forever",
]


# ============================================================
# Logging setup
# ============================================================

def setup_logging():
    """Set up logging to both file and console with a clear format."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


# ============================================================
# Robots.txt compliance
# ============================================================

def check_robots_txt(url: str) -> bool:
    """
    Check whether the IncIDecoder robots.txt allows our User-Agent to fetch
    the given URL. Returns True if allowed, False if disallowed.

    This is a courtesy check. We are required to honor robots.txt for any
    polite scraping.
    """
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    try:
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        allowed = rp.can_fetch(USER_AGENT, url)

        if allowed:
            logging.info(f"robots.txt allows fetching {url}")
        else:
            logging.warning(f"robots.txt DISALLOWS {url} - will NOT fetch")

        return allowed

    except Exception as e:
        # If we cannot read robots.txt, be conservative and assume disallowed.
        # This is the polite default for ambiguous cases.
        logging.error(f"Could not read robots.txt at {robots_url}: {e}")
        logging.error("Being conservative and assuming NOT allowed")
        return False


# ============================================================
# Fetching with retries and rate limiting
# ============================================================

def fetch_url(url: str, session: requests.Session) -> str | None:
    """
    Fetch a single URL with retries on transient failures.
    Returns the HTML text on success, or None on permanent failure.

    Rate limiting (the sleep between requests) is the caller's responsibility,
    not this function's. This function only handles a single request.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logging.info(f"Fetching (attempt {attempt}/{MAX_RETRIES}): {url}")

            response = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)

            # 200 OK is what we want
            if response.status_code == 200:
                logging.info(f"  Success: {len(response.text)} characters received")
                return response.text

            # 404 Not Found - the page does not exist. No point retrying.
            if response.status_code == 404:
                logging.warning(f"  Page does not exist (404): {url}")
                return None

            # 403 Forbidden - we are being blocked. No point retrying.
            if response.status_code == 403:
                logging.error(f"  Forbidden (403): {url}")
                logging.error("  This may indicate the site is blocking our User-Agent")
                return None

            # 5xx server errors - retry
            if 500 <= response.status_code < 600:
                logging.warning(f"  Server error {response.status_code}, will retry")
                time.sleep(2 ** attempt)  # exponential backoff: 2, 4, 8 seconds
                continue

            # Other error codes - log and give up
            logging.error(f"  Unexpected status code {response.status_code}")
            return None

        except requests.exceptions.Timeout:
            logging.warning(f"  Timeout on attempt {attempt}, will retry")
            time.sleep(2 ** attempt)
        except requests.exceptions.RequestException as e:
            logging.error(f"  Request error: {e}")
            time.sleep(2 ** attempt)

    logging.error(f"Failed to fetch after {MAX_RETRIES} attempts: {url}")
    return None


# ============================================================
# Saving the downloaded content
# ============================================================

def url_to_filename(url: str) -> str:
    """
    Convert a URL into a safe filename.

    Example:
        https://incidecoder.com/products/forever-living-aloe-vera-gelly
        becomes
        forever-living-aloe-vera-gelly.html
    """
    parsed = urlparse(url)
    # Get the last segment of the path - that is the product slug
    slug = parsed.path.rstrip("/").split("/")[-1]
    return f"{slug}.html"


def save_html(url: str, html: str) -> Path:
    """Save HTML content to disk and return the path it was saved to."""
    filename = url_to_filename(url)
    filepath = RAW_DATA_DIR / filename

    filepath.write_text(html, encoding="utf-8")
    logging.info(f"  Saved to {filepath}")
    return filepath


def append_metadata(url: str, filepath: Path, status: str):
    """Append a record to the metadata file describing this fetch."""
    record = {
        "url": url,
        "filepath": str(filepath) if filepath else None,
        "status": status,
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
    }
    with METADATA_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ============================================================
# Main scraping loop
# ============================================================

def scrape_urls(urls: list[str]) -> dict:
    """
    Download a list of URLs, with politeness checks and rate limiting.
    Returns summary statistics for the run.
    """
    # Verify robots.txt before doing anything
    if not check_robots_txt(urls[0]):
        logging.error("Aborting: robots.txt does not allow scraping these URLs")
        return {"total": len(urls), "succeeded": 0, "failed": len(urls)}

    # Create a session - this reuses the TCP connection across requests,
    # which is more efficient and polite than creating a new connection each time.
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    stats = {"total": len(urls), "succeeded": 0, "failed": 0}

    for i, url in enumerate(urls, start=1):
        logging.info(f"\n--- {i}/{len(urls)} ---")

        html = fetch_url(url, session)

        if html is not None:
            filepath = save_html(url, html)
            append_metadata(url, filepath, "success")
            stats["succeeded"] += 1
        else:
            append_metadata(url, None, "failed")
            stats["failed"] += 1

        # Rate limiting: wait between requests (except after the last one)
        if i < len(urls):
            logging.info(f"  Sleeping {DELAY_BETWEEN_REQUESTS_SECONDS}s before next request...")
            time.sleep(DELAY_BETWEEN_REQUESTS_SECONDS)

    return stats


# ============================================================
# Command-line interface
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--test", action="store_true",
                            help="Download just one page to verify the setup works")
    mode_group.add_argument("--full", action="store_true",
                            help="Download all product pages plus brand index")
    mode_group.add_argument("--url", type=str, metavar="URL",
                            help="Download a single specific URL")

    args = parser.parse_args()

    setup_logging()

    logging.info("=" * 60)
    logging.info("IncIDecoder Scraper - Forever Living Aloe Topical RAG")
    logging.info("=" * 60)
    logging.info(f"User-Agent: {USER_AGENT}")
    logging.info(f"Delay between requests: {DELAY_BETWEEN_REQUESTS_SECONDS}s")
    logging.info(f"Output directory: {RAW_DATA_DIR}")

    # Determine which URLs to fetch based on mode
    if args.test:
        urls_to_fetch = [PRODUCT_URLS[0]]  # just the first product
        logging.info("MODE: Test (single product)")
    elif args.full:
        urls_to_fetch = PRODUCT_URLS + BRAND_INDEX_URLS
        logging.info(f"MODE: Full ({len(urls_to_fetch)} URLs)")
    else:  # --url
        urls_to_fetch = [args.url]
        logging.info(f"MODE: Single URL ({args.url})")

    # Execute the scrape
    start_time = time.time()
    stats = scrape_urls(urls_to_fetch)
    elapsed = time.time() - start_time

    # Summary
    logging.info("\n" + "=" * 60)
    logging.info("Summary")
    logging.info("=" * 60)
    logging.info(f"Total URLs:    {stats['total']}")
    logging.info(f"Succeeded:     {stats['succeeded']}")
    logging.info(f"Failed:        {stats['failed']}")
    logging.info(f"Elapsed time:  {elapsed:.1f} seconds")
    logging.info(f"Output:        {RAW_DATA_DIR}")

    if stats["failed"] > 0:
        logging.warning(f"\n{stats['failed']} URLs failed. Check the log for details.")


if __name__ == "__main__":
    main()
