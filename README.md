# Forever Living Q&A — Image-Aware Agentic RAG

A research-grade Retrieval-Augmented Generation system for Forever Living
aloe-based products, built as a mid-course project at Ben-Gurion University and
extended into a multimodal agentic system.

The project demonstrates the full progression from a baseline RAG to an
**agent loop** (Loop 1 in [LangChain's loop-engineering framework](https://www.langchain.com/blog/the-art-of-loop-engineering))
that decides which of six tools to call (image identification is switchable between CLIP and Gemini Vision) to answer a user's question.

---

## Two modes, one system

| Mode         | What it does                                                                                                | When to use it                              |
|--------------|-------------------------------------------------------------------------------------------------------------|---------------------------------------------|
| **Simple**   | Direct RAG. Upload a photo, confirm the product, ask a question, get a grounded answer.                     | Predictable, fast, deterministic flow.      |
| **Agent**    | The LLM (Gemini 2.5-flash) decides which of six tools to call (image identification is switchable between CLIP and Gemini Vision), in what order, until it can answer.         | Flexible. Multi-source, multi-step queries. |

Both modes share the same retrieval backend; the agent simply adds an LLM-driven
control layer on top of it. The agent's trajectory is streamed live to the UI,
making the model's reasoning visible.

---

## What it can do

Examples of questions the system answers end-to-end:

- **Ingredient-level**: *"What does Octocrylene do?"*
  → `retrieve_product_info` returns the INCIDecoder chunk; cited as
  `[Forever Living Aloe Sunscreen]`.
- **Product-level**: *"What is Aloe Sunscreen used for and how do I apply it?"*
  → `get_product_description` returns the curated description; cited as
  `[Description: Aloe Sunscreen]`.
- **Comparison**: *"Compare Aloe Sunscreen and Aloe Lips for moisturizing ingredients."*
  → `compare_products` returns parallel chunks; the LLM synthesises the
  comparison with separate citations per product.
- **Scientific research**: *"Is there research on aloe vera for wound healing?"*
  → `search_pubmed` returns excerpts from PubMed; cited as
  `[PubMed: <article title>]`.
- **Image-driven**: *"What can you tell me about this product?"* (with photo)
  → `identify_product_from_image` → `retrieve_product_info` (possibly several
  times with refined queries).
- **Similar-product (non-Forever photo)**: *"What similar product do you have to
  this one?"* (photo of another brand) → `identify_product_from_image` returns
  "not in catalog" → `find_similar_product` recommends the closest Forever
  product by function (e.g. a competitor's muscle-rub photo → Aloe Heat Lotion).

All answers cite their sources. Medical/therapeutic claims and comparative
superlatives are blocked by the system prompt.

---

## Quick start

### 1. Install

```bash
git clone https://github.com/mrivd10/master_AI_midProj.git
cd master_AI_midProj
pip install -r requirements.txt
```

Required dependencies include `faiss-cpu`, `sentence-transformers`,
`transformers`, `torch`, `google-generativeai`, `streamlit`, `pillow`,
`python-dotenv`.

### 2. Set the API key

Create a `.env` file at the project root:

```
GEMINI_API_KEY=your-key-here
```

Get a free key at [https://aistudio.google.com/](https://aistudio.google.com/).

### 3. Build the indices

```bash
# Text index (INCIDecoder + PubMed corpus)
python src/build_index.py

# Image index (CLIP reference embeddings from product photos)
python src/build_image_index.py
```

### 4. Run the Streamlit app

```bash
# Local only (open http://localhost:8501)
streamlit run streamlit_app.py

# Accessible from a phone on the same network
streamlit run streamlit_app.py --server.address 0.0.0.0
# Then open http://<your-LAN-ip>:8501 on the phone
```

The first run downloads two embedding models (~600 MB CLIP +
~470 MB multilingual-e5-small). Subsequent runs use the local cache.

---

## The agent (Loop 1)

In Sydney Runkle's [Loop Engineering framework](https://www.langchain.com/blog/the-art-of-loop-engineering),
the agent loop is the foundation: a model that calls tools in a loop until a
task is complete. This project implements that loop with six tools:

| Tool                              | Purpose                                                                          | Returns                            |
|-----------------------------------|----------------------------------------------------------------------------------|------------------------------------|
| `identify_product_from_image()`   | Identify a product from the uploaded photo. Backend is switchable (USE_VISION): CLIP nearest-neighbour, or Gemini Vision which reads the label text (Hebrew/English) to tell look-alike products apart. | Product slug + confidence (+ reasoning for Vision). |
| `retrieve_product_info(q, p?)`    | Retrieve INCIDecoder ingredient chunks, optionally filtered to one product.      | Up to 5 chunks.                    |
| `search_pubmed(q)`                | Retrieve PubMed scientific literature chunks.                                    | Up to 5 chunks.                    |
| `compare_products(a, b, aspect)`  | Wrapper that retrieves info for two products in parallel.                        | Two parallel chunk lists.          |
| `get_product_description(p)`      | Return a curated, product-level description (use case + application).            | A single description string.       |
| `find_similar_product(desc)`      | Given a NON-Forever product, recommend the closest Forever product by function.  | Recommended slug + match quality.  |

The LLM's system prompt encodes **decision rules** that direct which tool to use
for which question type. The trajectory is bounded by `max_turns=5` and guarded
against same-tool-with-same-args repetition.

### Example trajectory

```
User: What is Aloe Sunscreen used for, and which of its ingredients act as moisturizers?

Turn 1: get_product_description(product='forever-living-aloe-sunscreen')
   → "A topical sunscreen with SPF 30, providing UVA and UVB protection..."

Turn 2: retrieve_product_info(query='moisturizing ingredients',
                              product='forever-living-aloe-sunscreen')
   → [Caprylyl Glycol, Aloe Barbadensis Leaf Juice, Sodium Gluconate, Propanediol, ...]

Final answer:
   Aloe Sunscreen is a topical sunscreen with SPF 30 ... [Description: Aloe Sunscreen]
   Ingredients in Aloe Sunscreen that act as moisturizers include:
     - Caprylyl Glycol     [Aloe Sunscreen]
     - Aloe Vera Gel       [Aloe Sunscreen]
     - Sodium Gluconate    [Aloe Sunscreen]
     - Propanediol         [Aloe Sunscreen]
```

The agent picked the right tool for each part of the question, passed the slug
from one tool into the next, and synthesised the result with two distinct
citation formats.

---

## Architecture

```
┌───────────────────────────────────────────────────────────────┐
│  Streamlit UI                                                  │
│  ┌─────────────────┐         ┌──────────────────────────┐     │
│  │  Simple mode    │         │  Agent mode (trajectory) │     │
│  └─────────────────┘         └──────────────────────────┘     │
└───────────────────┬─────────────────┬─────────────────────────┘
                    │                 │
                    ▼                 ▼
       ┌──────────────────┐    ┌────────────────────────┐
       │  src/retrieve.py │    │  src/agent.py          │
       │  (Retriever)     │    │  (6 tools + Gemini     │
       │                  │    │   function calling     │
       │                  │    │   loop)                │
       └────────┬─────────┘    └────────┬───────────────┘
                │                       │
                ▼                       ▼
       ┌──────────────────────────────────────────────┐
       │  data/processed/                              │
       │    faiss.index           (text embeddings)    │
       │    chunks.jsonl          (corpus)             │
       │    image_index.faiss     (CLIP embeddings)    │
       │    image_labels.json     (image -> product)   │
       └──────────────────────────────────────────────┘
                │
                ▼
       ┌──────────────────────────────────────────────┐
       │  src/identify_product.py                      │
       │    CLIP-based nearest-neighbour identifier    │
       │                                                │
       │  src/generation.py                            │
       │    Gemini 2.5-flash with strict SYSTEM_RULES  │
       └──────────────────────────────────────────────┘
```

### The data layer

The corpus combines **two sources** of factual grounding plus one curated
side-channel:

| Source                                | Format                | Purpose                              |
|---------------------------------------|-----------------------|--------------------------------------|
| **INCIDecoder** (`source=incidecoder.com`) | Structural chunks    | Ingredient functions and composition. |
| **PubMed** (`source=pubmed`)              | Fixed-size chunks    | Scientific research and safety.       |
| **Curated descriptions** (JSON)           | One-paragraph each   | Product-level use case + application. |

INCIDecoder and PubMed share the same FAISS index; the agent filters them by
`metadata.source`. Curated descriptions live in a separate JSON file loaded at
agent initialisation.

### Retrieval

Embeddings are computed with `intfloat/multilingual-e5-small`
(384-dimensional). FAISS uses exact L2 search. In Simple mode, a
filter-first strategy ranks only chunks of the user-selected product, which
solves the aggregative-question limitation present in the baseline RAG.

### Image identification

Two interchangeable backends, selected by the `USE_VISION` flag in `agent.py`:

- **CLIP** (`clip-ViT-B-32`, 512-dim): reference product images are encoded into
  a separate FAISS index. New photos are matched by inner product (= cosine
  similarity on normalised vectors). A confidence threshold (0.70) blocks
  false-positive identifications. Free and offline, but matches on overall
  appearance only - it cannot tell apart products with near-identical packaging
  (e.g. Bee Propolis vs Bee Pollen, same jar, different label text).
- **Gemini Vision** (`gemini-2.5-flash`): the photo is sent to the multimodal
  model along with the catalog of known product slugs. It reads the label text
  (Hebrew and English) and returns the matching slug plus a short reasoning
  string. This distinguishes look-alike products and needs no reference image
  database. Trade-off: one paid API call per identification.

### Generation

Gemini 2.5-flash with `temperature=0` and a strict system prompt that requires:
- citations in `[Source]` brackets after each factual claim,
- "I don't know" instead of hallucinated answers,
- no medical/therapeutic claims, no comparative superlatives.

---

## Evaluation (baseline RAG)

The base RAG was evaluated on a hand-built gold set with 131 questions across
9 Forever Living products:

| Metric          | Score   |
|-----------------|---------|
| **Hit@5**       | 99.3%   |
| **MRR**         | 0.979   |
| **Refusal accuracy** | 3/3 (manual refusal-expected set) |

Ablation studies validated:
- **Structural chunking** outperforms fixed-size chunking for this corpus.
- **`top-k = 5`** is the optimal retrieval depth.

The extensions (image identification, agent loop, curated descriptions) were
not part of the baseline evaluation and are validated qualitatively against the
target scenarios.

---

## Project structure

```
master_AI_midProj/
├── data/
│   ├── raw/
│   │   ├── incidecoder/              Scraped HTML + parsed JSON per product
│   │   ├── pubmed/                   PubMed article texts
│   │   ├── product_images/           Reference photos per product (3-5 each)
│   │   └── product_descriptions.json Curated descriptions (5th-tool source)
│   └── processed/
│       ├── chunks.jsonl              Unified corpus (incidecoder + pubmed)
│       ├── faiss.index               Text embedding index
│       ├── image_index.faiss         CLIP embedding index
│       └── image_labels.json         vector_idx -> {product, image_path}
├── eval/
│   ├── gold_set.jsonl                131 evaluation questions
│   ├── hard_questions.jsonl          Paraphrased "hard" questions
│   └── run_eval.py                   Hit@k + MRR
├── src/
│   ├── build_index.py                Build the text FAISS index
│   ├── build_image_index.py          Build the CLIP reference index
│   ├── chunk_pubmed.py               Fixed-size chunking for PubMed
│   ├── fetch_pubmed.py               PubMed API fetcher
│   ├── generate_gold_set.py          Auto-generate ingredient questions
│   ├── generate_hard_questions.py    Paraphrase questions for harder eval
│   ├── generation.py                 Gemini 2.5-flash answer generator
│   ├── retrieve.py                   FAISS-backed retriever
│   ├── identify_product.py           CLIP-based product identifier
│   ├── identify_product_vision.py     Gemini Vision product identifier (reads labels)
│   ├── find_similar_product.py        Match a non-Forever product to the closest Forever one
│   └── agent.py                      Agent loop + 6 tools + function calling
├── streamlit_app.py                  UI with Simple mode + Agent mode toggle
├── requirements.txt
├── README.md                         (this file)
└── .env                              GEMINI_API_KEY=...   (gitignored)
```

---

## Extending the system

### Add a new product

1. Scrape its INCIDecoder page (or hand-craft a similar JSON) and place it under
   `data/raw/incidecoder/`.
2. Take 3-5 photos of the product in varied lighting/angles and put them in
   `data/raw/product_images/<product-slug>/`.
3. Add a description entry in `data/raw/product_descriptions.json`.
4. Rebuild the indices:
   ```bash
   python src/build_index.py
   python src/build_image_index.py
   ```

### Refine the curated descriptions

`data/raw/product_descriptions.json` ships with generic starter descriptions.
The system runs immediately, but the recommended next step is to refine each
description with first-hand product knowledge (factual, no medical claims, no
superlatives). The agent picks up changes on the next process start.

### Tune retrieval

- `top_k` and the confidence threshold live in `src/identify_product.py`
  (`DEFAULT_TOP_K`, `DEFAULT_MIN_CONFIDENCE`).
- The agent's `max_turns` and model name live in `src/agent.py`
  (`MAX_TURNS_DEFAULT`, `MODEL_NAME`).

---

## Known limitations

- **Aggregative cross-product questions** ("which Forever products contain
  Tocopherol?") are still partial: top-`k` retrieval can miss minor mentions.
  Filter-first retrieval mitigates this within a single product.
- **Image identification depends on reference set quality.** Products without
  reference photos cannot be identified; very low-light or partial photos may
  score below the 0.70 threshold and fall through to manual selection.
- **The INCIDecoder corpus is ingredient-focused**, not marketing copy. Use-case
  questions are handled by curated descriptions, which require manual upkeep.

---

## Credits

- **Authors**: Ariel Oberstein, Dvir Margalit
- **Course**: AI Engineering certificate, Ben-Gurion University (110-hour
  program: Foundations, RAG Systems, LLM Agents)
- **Data sources**:
  - [INCIDecoder](https://incidecoder.com/) — public cosmetic ingredient database
  - [PubMed](https://pubmed.ncbi.nlm.nih.gov/) — biomedical literature
  - Forever Living product photos (collected by the author and distributor
    colleagues)
- **Frameworks**: FAISS, sentence-transformers, transformers (CLIP), Google
  Generative AI SDK (Gemini 2.5-flash), Streamlit

This is an educational project. It is not affiliated with Forever Living
Products and makes no medical or therapeutic claims about any product.
