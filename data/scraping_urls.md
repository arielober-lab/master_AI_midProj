# URL Collection - Forever Living Aloe Topical RAG

Master list of sources to scrape and download for the corpus. Each entry includes:
the URL, content type, expected content, scraping notes, and priority.

Priority levels:
- **P0**: Must-have. Without these, the corpus is insufficient.
- **P1**: Should-have. Adds significant depth and corroboration.
- **P2**: Nice-to-have. Scrape only if time allows after P0 and P1.

---

## Source 1: IncIDecoder (P0 - Primary Source)

Independent ingredient analyzer. Each product page has: full ingredient list, functional
classification of each ingredient (soothing, emollient, etc.), and scientific explanation.

### Brand index pages (start here)
- https://incidecoder.com/brands/forever-living-products
- https://incidecoder.com/brands/forever

### Individual product pages (P0 - core targets)

| # | Product | URL |
|---|---|---|
| 1 | Aloe Vera Gelly | https://incidecoder.com/products/forever-living-aloe-vera-gelly |
| 2 | Aloe Activator | https://incidecoder.com/products/forever-living-products-aloe-activator |
| 3 | Aloe MSM Gel | https://incidecoder.com/products/forever-living-products-aloe-msm-gel |
| 4 | Aloe Lips | https://incidecoder.com/products/forever-living-aloe-lips |
| 5 | Aloe Ever-shield Deodorant | https://incidecoder.com/products/forever-living-products-aloe-ever-shield |
| 6 | Aloe First Spray | https://incidecoder.com/products/forever-living-aloe-first-r-spray |
| 7 | Aloe Liquid Soap | https://incidecoder.com/products/forever-living-products-aloe-liquid-soap |
| 8 | Aloe Sunscreen | https://incidecoder.com/products/forever-living-aloe-sunscreen |
| 9 | Aloe Propolis Creme | https://incidecoder.com/products/forever-living-aloe-propolis-creme |
| 10 | Aloe Heat Lotion | (find via brand page) |
| 11 | Aloe Body Lotion | (find via brand page) |
| 12 | Aloe Moisturizing Lotion | (find via brand page) |
| 13 | Aloe Scrub | (find via brand page) |
| 14 | Aloe Jojoba Shampoo | (find via brand page) |
| 15 | Aloe Jojoba Conditioner | (find via brand page) |
| 16 | Aloe Body Wash | (find via brand page) |
| 17 | Aloe Balancing Cream | (find via brand page) |
| 18 | Avocado Face And Body Bar Soap | (find via brand page) |

**Scraping notes for IncIDecoder:**
- Robots.txt: check before scraping at https://incidecoder.com/robots.txt
- Rate limiting: keep requests at no more than 1 per 2 seconds to be polite
- User-Agent: identify yourselves honestly in the User-Agent string
- Each page has a stable structure: ingredient list with hover popups containing the explanation
- Use `requests` + `BeautifulSoup` for scraping
- Save raw HTML in `data/raw/incidecoder/` before extraction

---

## Source 2: Forever Living Official (P1 - Authoritative Source)

Official manufacturer content. Used for product identifiers, intended use, suggested
application, and any usage warnings.

### Catalogs (P1)
- https://cdn.foreverliving.com/global-assets/shop-link/KEN/ECatalog.pdf
  - Kenya edition global e-catalog, comprehensive product catalog in English
- https://resources.finalsite.net/images/v1667945330/tomballisdnet/x3wtzji4qd1kdpexuroq/ForeverLivingProducts-Tomball.pdf
  - Tomball ISD distributor catalog
- https://www.miamidade.gov/enet_discount/library/forever-living.pdf
  - Miami-Dade distributor catalog

### Product downloads page (P1)
- https://foreveraloe.in/downloads
  - Indian distributor with downloadable PDFs of individual products
  - Browse and download specific product PDFs

**Scraping notes for Forever Living:**
- The official site (foreverliving.com) may have geographic restrictions and bot detection
- Prefer the catalog PDFs over scraping product pages
- Use `pypdf` or `pdfplumber` to extract text from PDFs
- Save raw PDFs in `data/raw/forever-official/`

---

## Source 3: Scientific Evidence (P1 - Adds Depth)

Selected peer-reviewed studies on aloe vera and key ingredients used in Forever
products. Used for "what does research say about X" type questions.

### Search strategies (use these queries on PubMed)
- "Aloe barbadensis" topical skin
- "Aloe vera" wound healing
- "Aloe vera" anti-inflammatory skin
- "Bee propolis" skin
- "Allantoin" wound healing
- "MSM" methylsulfonylmethane topical

### Target studies (P1 - select 3 to 5)
- Open access studies only (look for Creative Commons or open access tags on PubMed)
- Prefer review articles over single studies for breadth
- Limit to studies published 2010 or later for relevance
- Save abstracts and conclusions, full text if open access

**Saving notes:**
- Save PDFs in `data/raw/studies/` with filename pattern `study_<topic>_<year>.pdf`
- Record DOI and citation in metadata

---

## Source 4: Forever Living Israel (P2 - Optional, Hebrew Materials)

For future Hebrew version. NOT for initial English version.
- https://www.foreverliving.co.il (if accessible)
- Save any Hebrew product info pages for Phase 2

**Note:** Do not include these in the initial submission. Phase 2 only.

---

## Collection Checklist

Use this to track progress as you collect:

### P0 (must complete before submission)
- [ ] All 18 IncIDecoder product pages saved as HTML to `data/raw/incidecoder/`
- [ ] Text extracted to clean `.md` files in `data/processed/incidecoder/`
- [ ] All product metadata captured (URL, product name, scraped date)

### P1 (should complete before submission)
- [ ] Kenya e-catalog PDF saved
- [ ] Text extracted from PDF, split by product
- [ ] 3-5 scientific studies selected and saved
- [ ] All metadata captured (URL, title, DOI, scraped/downloaded date)

### P2 (only if time allows)
- [ ] Indian distributor PDFs (select 5 most relevant)
- [ ] Additional Forever Living official sources

---

## Output Structure

After collection, the `data/` directory should look like:

```
data/
├── raw/
│   ├── incidecoder/
│   │   ├── aloe-vera-gelly.html
│   │   ├── aloe-activator.html
│   │   └── ... (one HTML per product)
│   ├── forever-official/
│   │   ├── KEN-ECatalog.pdf
│   │   └── ... (other PDFs)
│   └── studies/
│       ├── study_aloe_wound_healing_2018.pdf
│       └── ... (3-5 studies)
├── processed/
│   ├── documents.jsonl          # one document per line, cleaned text
│   └── chunks.jsonl             # one chunk per line, after chunking
└── MANIFEST.md
```

Each document entry in `documents.jsonl` should have the structure described
in the assignment:

```json
{
  "doc_id": "incidecoder_aloe_vera_gelly",
  "text": "...cleaned full text...",
  "metadata": {
    "source": "incidecoder.com",
    "url": "https://incidecoder.com/products/forever-living-aloe-vera-gelly",
    "product_name": "Forever Living Aloe Vera Gelly",
    "scraped_date": "2026-05-18"
  }
}
```
