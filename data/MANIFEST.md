# Dataset MANIFEST

## Corpus Information

**Corpus name:** Forever Living Aloe Topical Products Knowledge Base

**Domain:** Cosmetic and personal care products with aloe vera as primary or major ingredient. Focus on topical use (skin care, hair care, oral care), excluding ingestible drinks.

**Source of documents:**

The corpus is composed of three complementary source types:

1. **Independent ingredient analysis** (primary content source)
   - Source: IncIDecoder.com (third-party cosmetics ingredient analysis database)
   - Provides: ingredient lists, functional classification of each ingredient, scientific explanation of effects, allergy warnings
   - Notes: Independent perspective, not Forever Living marketing material

2. **Official Forever Living product information** (authoritative product info)
   - Source: Forever Living official PDFs and product pages
   - Provides: official product names, intended use, suggested usage instructions, marketing positioning
   - Notes: Used for product identifiers and official usage guidelines

3. **Peer-reviewed scientific studies** (depth and evidence)
   - Source: Selected open-access studies on aloe vera from PubMed and similar
   - Provides: scientific evidence on properties of aloe vera and key ingredients
   - Notes: Used for "what does research show" type questions

**Number of documents:** [TO COMPLETE AFTER COLLECTION]
Expected approximate counts:
- 15-18 IncIDecoder product pages
- 10-15 Forever Living official product information pages
- 3-5 selected scientific studies
- Estimated total: 30-40 source documents, expanding to 100-200 chunks after chunking

**Approximate number of pages / tokens:** [TO COMPLETE AFTER COLLECTION]
Initial estimate: 80-150 pages of substantive text, approximately 60,000-100,000 tokens

**File types:**
- `.html` (scraped from IncIDecoder, parsed)
- `.pdf` (Forever Living official catalogs and product sheets)
- `.txt` / `.md` (cleaned text extracted from above)
- `.jsonl` (final chunked corpus with metadata)
- `.json` (PubMed study abstracts, where applicable)

**License / permission:**
- IncIDecoder content: Used under fair use for educational and academic research purposes. Original source attributed in every chunk metadata. Not redistributed.
- Forever Living official content: Used under fair use for educational and academic research. Trademarks acknowledged. Not used for commercial purposes.
- PubMed studies: Open access content used in accordance with each paper's specific license.
- All content cited with source identifiers (URLs, DOIs) in retrieved chunk metadata.

**Why this corpus is suitable for RAG:**

1. **Non-trivial domain for baseline LLMs.** General-purpose language models have broad knowledge of aloe vera, but they do not know the specific ingredient compositions of individual Forever Living products. A question like "What preservative system does Forever Aloe MSM Gel use?" has a specific answer that exists only in the ingredient list of that product.

2. **Factual and verifiable.** Most questions about ingredients, suggested use, and product properties have objective answers that can be verified against the source documents.

3. **Multiple-source corroboration.** Each product is documented from at least two independent perspectives (the manufacturer and an independent ingredient analyzer), which is realistic for production RAG systems and enables interesting retrieval comparisons.

4. **Bounded scope, sufficient depth.** Fifteen to twenty products with rich content per product provide enough material for meaningful retrieval experiments without becoming unmanageable.

5. **Real-world relevance.** This is a domain where a working RAG system has practical value for distributors, consumers, and customer support, not just an academic exercise.

**What kind of questions should the system answer:**

The system is designed to answer factual questions about product composition, properties, suggested use, and the scientific role of ingredients. The intended question taxonomy includes:

- **Ingredient queries:** "What are all the ingredients in Forever Aloe Vera Gelly?" / "Which preservatives are used in Forever Aloe MSM Gel?"
- **Comparative ingredient queries:** "What are the differences in ingredients between Forever Aloe Propolis Creme and Forever Aloe Lips?"
- **Functional queries:** "Which ingredient in Forever Aloe Sunscreen provides UV protection?" / "What is the role of allantoin in Forever Aloe Activator?"
- **Usage queries:** "What is the suggested use for Forever Aloe Lips?" / "How is Forever Aloe First Spray intended to be applied?"
- **Property queries:** "Which Forever Living products contain bee propolis?" / "Which products are suitable for sensitive skin according to their ingredients?"
- **Negation/absence queries:** "Does Forever Aloe Liquid Soap contain parabens?"
- **Evidence queries:** "What does research say about the moisturizing properties of aloe barbadensis leaf juice?"

**What the system explicitly does not answer:**

- Medical or therapeutic claims about products ("Will this cure my eczema?")
- Personal recommendations ("Which product should I buy for my skin type?")
- Comparative superlatives ("Is Forever better than brand X?")

These are filtered at the prompt level. The system will respond that it cannot make such claims and refer the user to a medical professional or product specialist.

## Privacy and Sensitive Content

No private or sensitive information is included in this corpus. All sources are publicly available. No personal data, customer information, or distributor records appear in the corpus.

## Update Schedule

The corpus is collected and frozen at submission time. No live updates are part of this project.

## Authors and Acknowledgments

[TO COMPLETE]
- Project authors: [שמות שלך ושל השותף]
- Course: BGU AI Engineering Certificate Program
- Assignment: Mid-Course RAG Pipeline
- Submission date: 2026-05-26
