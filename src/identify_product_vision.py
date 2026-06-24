"""
identify_product_vision.py - Identify a Forever Living product from a photo
using Gemini 2.5-flash vision instead of CLIP.

Why this exists alongside identify_product.py (CLIP):
  CLIP matches on overall visual appearance. For products with near-identical
  packaging that differ only by the text on the label (e.g. Bee Propolis vs
  Bee Pollen - same jar, same colours, only the Hebrew/English words differ),
  CLIP cannot tell them apart. Gemini Vision reads the label text, so it
  distinguishes them reliably. It also needs NO reference image database -
  it identifies directly against a provided product list.

Trade-off: each identification is a paid API call (fine for demos / personal
use). CLIP remains available in identify_product.py for the free, offline path.

Usage as a module:
    from identify_product_vision import VisionProductIdentifier
    identifier = VisionProductIdentifier(known_products=[...])
    result = identifier.identify("path/to/photo.jpg")

Usage from CLI (for testing):
    python src/identify_product_vision.py path/to/photo.jpg
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

# --- Config ---
MODEL_NAME = "gemini-2.5-flash"

load_dotenv()
_api_key = os.environ.get("GEMINI_API_KEY")
if not _api_key:
    raise RuntimeError(
        "GEMINI_API_KEY not found. Create a .env file in the project root with:\n"
        "    GEMINI_API_KEY=your-key-here"
    )
genai.configure(api_key=_api_key)


def _load_catalog_from_descriptions():
    """Load the known-product slugs from product_descriptions.json.

    Keeping a single source of truth (the descriptions file) means adding a
    product there automatically makes the Vision identifier aware of it - no
    second list to maintain.
    """
    path = (Path(__file__).parent.parent / "data" / "raw"
            / "product_descriptions.json")
    if not path.exists():
        return list(_FALLBACK_PRODUCTS)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    slugs = [k for k in data if not k.startswith("_")]
    return slugs or list(_FALLBACK_PRODUCTS)


# Fallback list used only if the descriptions file is missing.
_FALLBACK_PRODUCTS = [
    "forever-living-aloe-first-r-spray",
    "forever-living-aloe-lips",
    "forever-living-aloe-propolis-creme",
    "forever-living-aloe-sunscreen",
    "forever-living-aloe-vera-gelly",
    "forever-living-products-aloe-activator",
    "forever-living-products-aloe-ever-shield",
    "forever-living-products-aloe-liquid-soap",
    "forever-living-products-aloe-msm-gel",
]

DEFAULT_KNOWN_PRODUCTS = _load_catalog_from_descriptions()


def _display_name(slug: str) -> str:
    name = slug
    for prefix in ("forever-living-products-", "forever-living-"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name.replace("-", " ").title()


class VisionProductIdentifier:
    """Identify Forever Living products from photos using Gemini Vision."""

    def __init__(self, known_products: Optional[List[str]] = None,
                 model_name: str = MODEL_NAME):
        self.known_products = known_products or DEFAULT_KNOWN_PRODUCTS
        self.model = genai.GenerativeModel(
            model_name,
            generation_config=genai.types.GenerationConfig(temperature=0.0),
        )

    def _build_prompt(self) -> str:
        product_lines = "\n".join(
            f"  - {slug}  ({_display_name(slug)})" for slug in self.known_products
        )
        return f"""You are a product identification system for Forever Living products.

Look at the photo and identify which product it is. Read any text on the label
(it may be in Hebrew or English) to distinguish between products that look
similar but are different (for example, two jars with identical packaging but
different product names).

You must choose from this exact list of known products. Use the EXACT slug
string in your answer:

{product_lines}

Respond with ONLY a JSON object, no markdown, no extra text:
{{
  "product": "<exact-slug-or-null>",
  "confidence": <number between 0 and 1>,
  "is_confident": <true or false>,
  "reasoning": "<one short sentence: what on the label led to your choice>"
}}

Rules:
- If the photo clearly shows one of the known products, set product to its slug
  and is_confident to true.
- If the photo is NOT a Forever Living product from the list, or you cannot tell,
  set product to null and is_confident to false.
- confidence reflects how sure you are (1.0 = certain, 0.0 = no idea).
"""

    def identify(self, image: Union[str, Path, Image.Image]) -> Dict:
        """
        Identify the product in the given image.

        Returns a dict with:
            product:      best matching slug (or None)
            confidence:   0..1
            is_confident: bool
            reasoning:    short explanation of what on the label decided it
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise TypeError(f"image must be path or PIL.Image, got {type(image)}")

        prompt = self._build_prompt()
        response = self.model.generate_content([prompt, image])

        # Parse the JSON response (strip any stray markdown fences)
        text = (response.text or "").strip()
        text = text.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {
                "product":      None,
                "confidence":   0.0,
                "is_confident": False,
                "reasoning":    f"Could not parse model response: {text[:120]}",
            }

        # Validate the returned slug is actually in our known list
        product = data.get("product")
        if product is not None and product not in self.known_products:
            # Model returned something off-list; treat as uncertain
            return {
                "product":      None,
                "confidence":   float(data.get("confidence", 0.0)),
                "is_confident": False,
                "reasoning":    f"Model returned unknown slug '{product}'.",
            }

        return {
            "product":      product,
            "confidence":   float(data.get("confidence", 0.0)),
            "is_confident": bool(data.get("is_confident", False)),
            "reasoning":    data.get("reasoning", ""),
        }


def main():
    if len(sys.argv) != 2:
        print("Usage: python src/identify_product_vision.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    identifier = VisionProductIdentifier()
    result = identifier.identify(image_path)

    print()
    print("=" * 70)
    print(f"  Query image: {image_path}")
    print("=" * 70)
    if result["is_confident"]:
        print(f"  Identified : {result['product']}  ({_display_name(result['product'])})")
        print(f"  Confidence : {result['confidence']:.2f}")
    else:
        print(f"  Identified : UNCERTAIN")
        if result["product"]:
            print(f"  Best guess : {result['product']}")
        print(f"  Confidence : {result['confidence']:.2f}")
    print(f"  Reasoning  : {result['reasoning']}")
    print()


if __name__ == "__main__":
    main()
