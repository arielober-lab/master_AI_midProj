"""
generation.py - Generate a grounded answer from retrieved context using Gemini.

This module takes a question plus the chunks retrieved by retrieve.py, builds a
strict context-only prompt, calls the Gemini LLM, and returns the answer text.

The prompt enforces several rules important for this project:
  - Answer ONLY from the provided context (no outside knowledge -> no hallucination)
  - Say "I don't know" when the context lacks the answer
  - Cite the source product for each claim
  - No medical/therapeutic claims (regulatory safety)
  - No comparative superlatives (brand rule)
"""

import os

import google.generativeai as genai
from dotenv import load_dotenv

# Load the API key from the .env file (kept out of git by .gitignore)
load_dotenv()
_api_key = os.environ.get("GEMINI_API_KEY")
if not _api_key:
    raise RuntimeError(
        "GEMINI_API_KEY not found. Create a .env file in the project root with:\n"
        "    GEMINI_API_KEY=your-key-here"
    )
genai.configure(api_key=_api_key)

MODEL_NAME = "gemini-2.5-flash"

SYSTEM_RULES = """You are a factual assistant that answers questions about Forever Living
aloe-based topical products (skin, hair, and oral care), using ONLY the provided context.

Follow these rules strictly:
1. Use ONLY information found in the CONTEXT below. Never use outside knowledge.
2. If the context does not contain the answer, reply exactly:
   "I don't have enough information in the provided sources to answer that."
3. After each factual claim, cite the source product in square brackets, e.g. [Aloe MSM Gel].
4. Do NOT make medical or therapeutic claims (curing, treating, or healing diseases).
   Describe ingredient functions factually, only as stated in the context.
5. Do NOT use comparative superlatives (best, strongest, most effective).
6. Be concise and factual."""


def build_prompt(question, retrieved_chunks):
    """Assemble the full prompt from the system rules, context, and question."""
    context_parts = []
    for r in retrieved_chunks:
        chunk = r["chunk"]
        product = chunk["metadata"].get("product_name", "Unknown product")
        context_parts.append(f"[Source: {product}]\n{chunk['text']}")
    context = "\n\n---\n\n".join(context_parts)

    return f"""{SYSTEM_RULES}

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def generate_answer(question, retrieved_chunks):
    """Call the LLM with the retrieved context and return the answer text."""
    model = genai.GenerativeModel(
        MODEL_NAME,
        generation_config=genai.types.GenerationConfig(temperature=0.0),
    )
    prompt = build_prompt(question, retrieved_chunks)
    response = model.generate_content(prompt)
    return response.text.strip()
