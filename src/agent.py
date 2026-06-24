"""
src/agent.py - The agentic version of the Forever Living Q&A system.

Defines five tools and an agent loop that lets Gemini 2.5-flash decide which
of them to call, in what order, until it can answer the user's question.

Tools:
  - identify_product_from_image()                              CLIP or Gemini Vision
  - retrieve_product_info(query, product=None)                 INCIDecoder corpus
  - search_pubmed(query)                                       PubMed corpus
  - compare_products(product_a, product_b, aspect)             wraps retrieve x2
  - get_product_description(product)                           curated descriptions
  - find_similar_product(product_description)                  closest Forever match

The image identifier is switchable via the USE_VISION flag below.

The agent loop is a generator that yields trajectory events as they happen,
so the Streamlit UI can stream them and show the agent's reasoning live:
    {"type": "tool_call",   "turn": N, "name": "...", "args": {...}}
    {"type": "tool_result", "turn": N, "result": ...}
    {"type": "final",       "answer": "..."}
    {"type": "error",       "message": "..."}

Usage:
    for event in run_agent(user_message, image_path=...):
        ...   # display in UI
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import google.generativeai as genai
from dotenv import load_dotenv

# Make the rest of src/ importable
sys.path.insert(0, str(Path(__file__).parent))

from identify_product import ProductIdentifier              # noqa: E402
from identify_product_vision import VisionProductIdentifier  # noqa: E402
from find_similar_product import SimilarProductFinder        # noqa: E402
from retrieve import Retriever                              # noqa: E402

# ============================================================
# Config
# ============================================================
# Which image identifier to use:
#   True  -> Gemini Vision (reads the label text; distinguishes look-alike
#            products like Bee Propolis vs Bee Pollen; needs no reference images)
#   False -> CLIP (free, offline, but matches on overall appearance only)
USE_VISION                = True

MODEL_NAME                = "gemini-2.5-flash"
MAX_TURNS_DEFAULT         = 5
PRODUCT_DESCRIPTIONS_PATH = (
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


# ============================================================
# Static data loaded at import time
# ============================================================
def _load_product_descriptions():
    """Load the curated product descriptions from disk (or return empty dict)."""
    if not PRODUCT_DESCRIPTIONS_PATH.exists():
        return {}
    with PRODUCT_DESCRIPTIONS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Drop comment keys (any key starting with "_")
    return {k: v for k, v in data.items() if not k.startswith("_")}


_PRODUCT_DESCRIPTIONS = _load_product_descriptions()


# ============================================================
# Per-request context (set by run_agent before tools are called)
# ============================================================
_context = {
    "identifier":  None,   # ProductIdentifier instance
    "retriever":   None,   # Retriever instance
    "image_path":  None,   # str or None
}


def _set_context(identifier, retriever, image_path):
    _context["identifier"] = identifier
    _context["retriever"]  = retriever
    _context["image_path"] = image_path


# ============================================================
# Helpers
# ============================================================
def _chunk_matches_product(chunk, target_product):
    """Match a chunk to a folder-style product slug."""
    pname    = chunk["metadata"].get("product_name", "")
    chunk_id = chunk.get("chunk_id", "")
    if pname == target_product:
        return True

    def norm(s):
        return s.lower().replace(" ", "").replace("-", "").replace("_", "")

    if norm(pname) == norm(target_product):
        return True
    if norm(target_product) in norm(pname) or norm(pname) in norm(target_product):
        return True
    if target_product in chunk_id:
        return True
    return False


def _display_name(folder_name):
    """'forever-living-aloe-ever-shield' -> 'Aloe Ever Shield'."""
    name = folder_name
    for prefix in ("forever-living-products-", "forever-living-"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name.replace("-", " ").title()


def _shrink_chunk(chunk_record):
    """Trim a chunk down to what's useful to feed back to the LLM."""
    chunk = chunk_record["chunk"]
    return {
        "product":  chunk["metadata"].get("product_name", "Unknown"),
        "chunk_id": chunk["chunk_id"],
        "text":     chunk["text"][:500],
    }


# ============================================================
# THE FIVE TOOLS
# ============================================================
def identify_product_from_image() -> Dict:
    """Identify the Forever Living product in the photo the user uploaded.

    Only call this if (1) the user uploaded a photo, AND (2) the question
    refers to that specific product. No arguments needed - operates on the
    most recent upload.

    Returns:
        A dict with 'product' (slug like 'forever-living-aloe-sunscreen'),
        'confidence' (0..1), 'is_confident' (bool), and 'alternatives'
        (other possible products).
    """
    img_path = _context["image_path"]
    if not img_path:
        return {"error": "No image was uploaded by the user. Cannot identify."}
    identifier = _context["identifier"]
    if identifier is None:
        return {"error": "Identifier not initialized."}
    result = identifier.identify(img_path)

    # Normalize across the two identifier backends:
    #   CLIP    -> result has 'top_matches' (list of {product, score, ...})
    #   Vision  -> result has 'reasoning'   (a sentence explaining the choice)
    out = {
        "product":      result.get("product"),
        "confidence":   result.get("confidence"),
        "is_confident": result.get("is_confident"),
    }
    if "top_matches" in result:
        out["alternatives"] = [m["product"] for m in result["top_matches"][:3]]
    if "reasoning" in result:
        out["reasoning"] = result["reasoning"]
    return out


def retrieve_product_info(query: str, product: Optional[str] = None) -> List[Dict]:
    """Search the INCIDecoder ingredient knowledge base for product information.

    Use this for any factual question about a Forever Living product's
    INGREDIENTS, their cosmetic functions (moisturizer, preservative, etc.),
    or the product's chemical composition.

    Do NOT use this for general "what is this product for" questions about
    use case or application - use get_product_description for those.

    Args:
        query:   What information to search for, in natural language.
        product: Optional product slug (e.g. 'forever-living-aloe-sunscreen').
                 If provided, results are filtered to that product only.

    Returns:
        A list of up to 5 dicts, each with 'product', 'chunk_id', 'text'.
    """
    retriever = _context["retriever"]
    if retriever is None:
        return [{"error": "Retriever not initialized."}]
    raw = retriever.retrieve(query, top_k=20)
    filtered = [
        r for r in raw
        if r["chunk"]["metadata"].get("source") == "incidecoder.com"
    ]
    if product:
        filtered = [r for r in filtered if _chunk_matches_product(r["chunk"], product)]
    return [_shrink_chunk(r) for r in filtered[:5]]


def search_pubmed(query: str) -> List[Dict]:
    """Search PubMed scientific literature for medical, safety, or research-related info.

    Use this for questions about safety, contraindications, clinical evidence,
    or scientific research related to ingredients or aloe vera in general.

    Args:
        query: What to search for, in natural language.

    Returns:
        A list of up to 5 dicts, each with 'product' (article title),
        'chunk_id', and 'text' (excerpt from the article).
    """
    retriever = _context["retriever"]
    if retriever is None:
        return [{"error": "Retriever not initialized."}]
    raw = retriever.retrieve(query, top_k=20)
    filtered = [r for r in raw if r["chunk"]["metadata"].get("source") == "pubmed"]
    return [_shrink_chunk(r) for r in filtered[:5]]


def compare_products(product_a: str, product_b: str, aspect: str) -> Dict:
    """Compare two Forever Living products on a specific aspect.

    Returns parallel chunks from each product so you can compare them.
    The LLM should synthesize the actual comparison in the final answer.

    Args:
        product_a: First product slug (e.g. 'forever-living-aloe-lips').
        product_b: Second product slug.
        aspect:    What to compare them on (e.g. 'moisturizing ingredients').

    Returns:
        A dict with 'product_a' and 'product_b', each holding up to 5 chunks.
    """
    return {
        "product_a": retrieve_product_info(aspect, product=product_a),
        "product_b": retrieve_product_info(aspect, product=product_b),
    }


def get_product_description(product: str) -> Dict:
    """Get the curated general description of a Forever Living product.

    Use this for natural product-level questions like:
      - "What is this product for?"
      - "How do I use it?"
      - "What is Aloe Sunscreen used for?"
      - "Tell me about this product."

    Do NOT use this for ingredient-level questions - use retrieve_product_info
    for those.

    Args:
        product: Product slug (e.g. 'forever-living-aloe-sunscreen').

    Returns:
        A dict with the description, or an error if not found.
    """
    desc = _PRODUCT_DESCRIPTIONS.get(product)
    if desc is None:
        return {
            "error": (
                f"No curated description for '{product}'. "
                f"Try retrieve_product_info instead, or check that the product "
                f"slug is correct."
            )
        }
    return {"product": product, "description": desc}


# Lazily-created SimilarProductFinder (built on first use to avoid loading it
# when the agent never needs it).
_similar_finder = None


def find_similar_product(product_description: str) -> Dict:
    """Recommend the closest Forever Living product to a NON-Forever product.

    Use this when the user shows or describes a product that is NOT in the
    Forever catalog (identify_product_from_image returned null / not confident),
    and asks what similar product Forever offers.

    Args:
        product_description: A description of the non-Forever product - what it
            is and what it is for (e.g. "a moisturizing face cream for dry skin",
            or whatever the image identification read off the label).

    Returns:
        A dict with 'recommended_product' (slug or None), 'match_quality'
        (close/partial/none), and 'reasoning'.
    """
    global _similar_finder
    if _similar_finder is None:
        _similar_finder = SimilarProductFinder()
    return _similar_finder.find(product_description)


# Registry mapping tool names -> Python callables
TOOL_FUNCS = {
    "identify_product_from_image": identify_product_from_image,
    "retrieve_product_info":       retrieve_product_info,
    "search_pubmed":               search_pubmed,
    "compare_products":            compare_products,
    "get_product_description":     get_product_description,
    "find_similar_product":        find_similar_product,
}


# ============================================================
# System prompt
# ============================================================
def _build_system_prompt(known_products, has_image: bool):
    """Build the system instruction for the agent."""
    products_list = "\n".join(
        f"  - {slug}  ({_display_name(slug)})" for slug in known_products
    )
    image_note = (
        "The user HAS uploaded a photo. You may call identify_product_from_image "
        "if the question refers to that product."
        if has_image else
        "The user did NOT upload a photo. Do NOT call identify_product_from_image; "
        "it will return an error."
    )
    return f"""You are an agentic Q&A assistant for Forever Living aloe-based topical products.

You have five tools available. Use them to answer the user's question. Use ONLY
the information returned by your tools - never rely on your own knowledge for
factual claims about products, ingredients, or research.

Decision rules — pick the right tool for each kind of question:

  - PRODUCT-LEVEL questions ("What is this product for?", "How do I use it?",
    "Tell me about Aloe Lips") -> get_product_description.

  - INGREDIENT-LEVEL questions ("What does Octocrylene do?", "What ingredients
    are in X?", "Which ingredients act as moisturizers?") -> retrieve_product_info.

  - SAFETY / RESEARCH questions ("Is X safe during pregnancy?", "Is there
    research on aloe vera and wound healing?") -> search_pubmed.

  - COMPARISON questions ("Compare A and B on Z") -> compare_products.

  - IDENTIFY-FROM-PHOTO -> identify_product_from_image (only when the user
    uploaded a photo and the question references that product).

  - SIMILAR-PRODUCT REQUESTS: if the user shows or describes a product that is
    NOT a Forever Living product (for example, identify_product_from_image
    returned null / not confident) and asks what similar product Forever has,
    use find_similar_product with a short description of what the product is.

You may call multiple tools in sequence, building on previous results. When you
have enough information, write the final answer and stop calling tools.

Final answer rules (same as the base RAG, applied to the synthesis you write):
  - Cite sources in square brackets:
      [Aloe Sunscreen]                     for retrieve_product_info results
      [PubMed: <article title>]            for search_pubmed results
      [Description: Aloe Sunscreen]        for get_product_description results
  - Do NOT make medical or therapeutic claims (curing, treating, healing diseases).
  - Do NOT use comparative superlatives (best, strongest, most effective).
  - If your tools do not contain the answer, say so explicitly.

Image context: {image_note}

Known product slugs (use these EXACT strings when calling tools):
{products_list}
"""


# ============================================================
# The agent loop
# ============================================================
def run_agent(
    user_message: str,
    image_path: Optional[str] = None,
    identifier: Optional[ProductIdentifier] = None,
    retriever:  Optional[Retriever] = None,
    max_turns:  int = MAX_TURNS_DEFAULT,
) -> Iterator[Dict]:
    """
    Run the agent loop. Generator that yields trajectory events as they happen.

    Args:
        user_message: The user's question.
        image_path:   Path to an uploaded photo (or None).
        identifier:   Optional pre-loaded ProductIdentifier (Streamlit passes one).
        retriever:    Optional pre-loaded Retriever.
        max_turns:    Safety cap on tool-call iterations.

    Yields:
        Trajectory events (see module docstring).
    """
    if identifier is None:
        identifier = VisionProductIdentifier() if USE_VISION else ProductIdentifier()
    if retriever is None:
        retriever = Retriever()

    _set_context(identifier, retriever, image_path)

    # Derive the known-product list in a way that works for both backends:
    #   CLIP   has .labels (list of {product, ...})
    #   Vision has .known_products (list of slugs)
    if hasattr(identifier, "labels"):
        known_products = sorted(set(L["product"] for L in identifier.labels))
    else:
        known_products = sorted(set(identifier.known_products))
    system_prompt  = _build_system_prompt(known_products, has_image=image_path is not None)

    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        tools=list(TOOL_FUNCS.values()),
        system_instruction=system_prompt,
        generation_config=genai.types.GenerationConfig(temperature=0.0),
    )
    chat = model.start_chat(enable_automatic_function_calling=False)
    response = chat.send_message(user_message)

    recent_calls: List[str] = []

    for turn in range(1, max_turns + 1):
        fn_call = None
        for part in response.parts:
            if getattr(part, "function_call", None) and part.function_call.name:
                fn_call = part.function_call
                break

        if fn_call is None:
            text = (response.text or "").strip() or "(model returned empty answer)"
            yield {"type": "final", "answer": text}
            return

        args = {k: v for k, v in fn_call.args.items()} if fn_call.args else {}

        call_signature = f"{fn_call.name}({json.dumps(args, sort_keys=True, default=str)})"
        if recent_calls and recent_calls[-1] == call_signature:
            yield {
                "type": "error",
                "message": (
                    f"Agent repeated the same tool call ({fn_call.name}) with the same "
                    f"arguments. Stopping to prevent a loop."
                ),
            }
            return
        recent_calls.append(call_signature)

        yield {"type": "tool_call", "turn": turn, "name": fn_call.name, "args": args}

        tool_func = TOOL_FUNCS.get(fn_call.name)
        if tool_func is None:
            result = {"error": f"Unknown tool: {fn_call.name}"}
        else:
            try:
                result = tool_func(**args)
            except TypeError as e:
                result = {"error": f"Bad arguments for {fn_call.name}: {e}"}
            except Exception as e:
                result = {"error": f"Tool '{fn_call.name}' raised: {e}"}

        yield {"type": "tool_result", "turn": turn, "result": result}

        response = chat.send_message(
            genai.protos.Content(
                parts=[genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=fn_call.name,
                        response={"result": json.dumps(result, default=str)},
                    )
                )]
            )
        )

    yield {
        "type": "error",
        "message": (
            f"Agent reached the maximum number of steps ({max_turns}) without "
            f"completing the answer. Try a more specific question, or use "
            f"Simple mode for a direct retrieval."
        ),
    }


# ============================================================
# CLI testing
# ============================================================
def _main():
    if len(sys.argv) < 2:
        print('Usage: python src/agent.py "your question here" [path/to/image.jpg]')
        sys.exit(1)
    question   = sys.argv[1]
    image_path = sys.argv[2] if len(sys.argv) >= 3 else None
    print(f"\n🧑  {question}")
    if image_path:
        print(f"📷  (image: {image_path})")
    print("─" * 70)
    for event in run_agent(question, image_path=image_path):
        etype = event["type"]
        if etype == "tool_call":
            args_str = ", ".join(f"{k}={v!r}" for k, v in event["args"].items())
            print(f"   🔧 turn {event['turn']}: {event['name']}({args_str})")
        elif etype == "tool_result":
            preview = json.dumps(event["result"], default=str)[:200]
            print(f"      ↩  {preview}{'...' if len(preview) >= 200 else ''}")
        elif etype == "final":
            print(f"🤖  {event['answer']}")
        elif etype == "error":
            print(f"⚠️   {event['message']}")
    print("─" * 70)


if __name__ == "__main__":
    _main()
