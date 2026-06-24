"""
find_similar_product.py - Recommend the closest Forever Living product to a
non-Forever product, by function.

Scenario: a user photographs a competitor's product (e.g. a moisturizing lotion
from another brand). The Vision identifier returns "not in catalog" but reads
what the product is. This module then asks Gemini to match that description to
the most functionally similar Forever Living product from our catalog.

It matches on FUNCTION / use case (cream -> cream, shampoo -> shampoo), using
the curated product descriptions as the catalog.

Usage as a module:
    from find_similar_product import SimilarProductFinder
    finder = SimilarProductFinder()
    result = finder.find("a moisturizing face cream for dry skin")
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import google.generativeai as genai
from dotenv import load_dotenv

MODEL_NAME = "gemini-2.5-flash"
DESCRIPTIONS_PATH = (
    Path(__file__).parent.parent / "data" / "raw" / "product_descriptions.json"
)

load_dotenv()
_api_key = os.environ.get("GEMINI_API_KEY")
if not _api_key:
    raise RuntimeError(
        "GEMINI_API_KEY not found. Create a .env file in the project root with:\n"
        "    GEMINI_API_KEY=your-key-here"
    )
genai.configure(api_key=_api_key)


def _display_name(slug: str) -> str:
    name = slug
    for prefix in ("forever-living-products-", "forever-living-", "forever-"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name.replace("-", " ").title()


def _load_descriptions() -> Dict[str, str]:
    if not DESCRIPTIONS_PATH.exists():
        return {}
    with DESCRIPTIONS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


class SimilarProductFinder:
    """Match a non-Forever product description to the closest Forever product."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.descriptions = _load_descriptions()
        self.model = genai.GenerativeModel(
            model_name,
            generation_config=genai.types.GenerationConfig(temperature=0.0),
        )

    def _build_prompt(self, foreign_description: str) -> str:
        catalog = "\n".join(
            f"  - {slug}  ({_display_name(slug)}): {desc}"
            for slug, desc in self.descriptions.items()
        )
        return f"""A user showed a product that is NOT a Forever Living product.
Here is what it is:

  "{foreign_description}"

Below is the Forever Living catalog. Recommend the SINGLE Forever Living product
whose function and use case are the closest match to the user's product. Match
on what the product is FOR (e.g. a face cream matches a face cream, a shampoo
matches a shampoo), not on brand or exact ingredients.

Catalog:
{catalog}

Respond with ONLY a JSON object, no markdown, no extra text:
{{
  "recommended_product": "<exact-slug-from-catalog-or-null>",
  "match_quality": "<close | partial | none>",
  "reasoning": "<one short sentence explaining the functional match>"
}}

Rules:
- Choose the slug EXACTLY as written in the catalog.
- If no Forever product serves a similar function, set recommended_product to
  null and match_quality to "none".
- Do NOT make medical or therapeutic claims.
- Do NOT use comparative superlatives (best, strongest, most effective)."""

    def find(self, foreign_description: str) -> Dict:
        """
        Recommend the closest Forever product to a described non-Forever product.

        Returns a dict with:
            recommended_product: slug (or None)
            match_quality:       "close" | "partial" | "none"
            reasoning:           short explanation
        """
        prompt = self._build_prompt(foreign_description)
        response = self.model.generate_content(prompt)
        text = (response.text or "").strip()
        text = text.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {
                "recommended_product": None,
                "match_quality":       "none",
                "reasoning":           f"Could not parse model response: {text[:120]}",
            }

        rec = data.get("recommended_product")
        if rec is not None and rec not in self.descriptions:
            return {
                "recommended_product": None,
                "match_quality":       "none",
                "reasoning":           f"Model returned unknown slug '{rec}'.",
            }
        return {
            "recommended_product": rec,
            "match_quality":       data.get("match_quality", "none"),
            "reasoning":           data.get("reasoning", ""),
        }


def _main():
    import sys
    if len(sys.argv) < 2:
        print('Usage: python src/find_similar_product.py "<product description>"')
        sys.exit(1)
    description = sys.argv[1]
    finder = SimilarProductFinder()
    result = finder.find(description)
    print()
    print("=" * 70)
    print(f"  Foreign product: {description}")
    print("=" * 70)
    if result["recommended_product"]:
        print(f"  Recommended : {result['recommended_product']}  "
              f"({_display_name(result['recommended_product'])})")
        print(f"  Match       : {result['match_quality']}")
    else:
        print(f"  Recommended : none")
    print(f"  Reasoning   : {result['reasoning']}")
    print()


if __name__ == "__main__":
    _main()
