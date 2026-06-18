"""
src/agent.py - The agentic version of the Forever Living Q&A system.

Defines four tools and an agent loop that lets Gemini 2.5-flash decide which
of them to call, in what order, until it can answer the user's question.

Tools:
  - identify_product_from_image()                              uses CLIP
  - retrieve_product_info(query, product=None)                 INCIDecoder corpus
  - search_pubmed(query)                                       PubMed corpus
  - compare_products(product_a, product_b, aspect)             wraps retrieve x2

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

from identify_product import ProductIdentifier   # noqa: E402
from retrieve import Retriever                   # noqa: E402

# ============================================================
# Config
# ============================================================
MODEL_NAME        = "gemini-2.5-flash"
MAX_TURNS_DEFAULT = 5

load_dotenv()
_api_key = os.environ.get("GEMINI_API_KEY")
if not _api_key:
    raise RuntimeError(
        "GEMINI_API_KEY not found. Create a .env file in the project root with:\n"
        "    GEMINI_API_KEY=your-key-here"
    )
genai.configure(api_key=_api_key)


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
# Helpers (shared with streamlit_app.py logic)
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
        "text":     chunk["text"][:500],   # cap to keep prompts compact
    }


# ============================================================
# THE FOUR TOOLS
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
    return {
        "product":      result["product"],
        "confidence":   result["confidence"],
        "is_confident": result["is_confident"],
        "alternatives": [m["product"] for m in result["top_matches"][:3]],
    }


def retrieve_product_info(query: str, product: Optional[str] = None) -> List[Dict]:
    """Search the INCIDecoder ingredient knowledge base for product info.

    Use this for any factual question about a Forever Living product's
    ingredients, their cosmetic functions (moisturizer, preservative, etc.),
    or the product's overall composition.

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


# Registry mapping tool names -> Python callables
TOOL_FUNCS = {
    "identify_product_from_image": identify_product_from_image,
    "retrieve_product_info":       retrieve_product_info,
    "search_pubmed":               search_pubmed,
    "compare_products":            compare_products,
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

You have four tools available. Use them to answer the user's question. Use ONLY
the information returned by your tools - never rely on your own knowledge for
factual claims about products, ingredients, or research.

Decision rules:
- For questions about a product's ingredients or what they do: use retrieve_product_info.
- For questions about safety, side effects, or scientific research: use search_pubmed.
- For comparisons between two products: use compare_products.
- For identifying a product from a photo: use identify_product_from_image (only when relevant).
- You may call multiple tools in sequence, building on previous results.
- When you have enough information, write the final answer and stop calling tools.

Final answer rules (same as the base RAG, applied to the synthesis you write):
- Cite sources in square brackets, e.g. [Aloe Sunscreen] for product info,
  or [PubMed: article title] for research.
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
        Trajectory events (see module docstring for the shape).
    """
    # Lazy-load models if Streamlit didn't pre-load them
    if identifier is None:
        identifier = ProductIdentifier()
    if retriever is None:
        retriever = Retriever()

    # Set per-request context that the module-level tool functions read
    _set_context(identifier, retriever, image_path)

    # Build system instruction
    known_products = sorted(set(L["product"] for L in identifier.labels))
    system_prompt = _build_system_prompt(known_products, has_image=image_path is not None)

    # Set up the Gemini model with our Python tools
    # Passing functions directly lets google-generativeai auto-generate the schemas
    # from the type hints and docstrings.
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        tools=list(TOOL_FUNCS.values()),
        system_instruction=system_prompt,
        generation_config=genai.types.GenerationConfig(temperature=0.0),
    )
    chat = model.start_chat(enable_automatic_function_calling=False)

    # Send the first user message
    response = chat.send_message(user_message)

    # Track recent (tool, args) pairs to detect repetition
    recent_calls: List[str] = []

    for turn in range(1, max_turns + 1):
        # Look for a function_call in the model's response parts
        fn_call = None
        for part in response.parts:
            if getattr(part, "function_call", None) and part.function_call.name:
                fn_call = part.function_call
                break

        if fn_call is None:
            # No tool call - the model has produced a final answer
            text = (response.text or "").strip() or "(model returned empty answer)"
            yield {"type": "final", "answer": text}
            return

        # Convert protobuf args to a plain dict
        args = {k: v for k, v in fn_call.args.items()} if fn_call.args else {}

        # Repetition guard: same tool+args twice in a row is a sign of being stuck
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

        # Emit the tool_call event
        yield {"type": "tool_call", "turn": turn, "name": fn_call.name, "args": args}

        # Execute the tool, catching any unexpected exception so the app never crashes
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

        # Emit the tool_result event
        yield {"type": "tool_result", "turn": turn, "result": result}

        # Send the tool result back to the model so it can decide the next step
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

    # If we exit the for-loop, we hit max_turns without a final answer
    yield {
        "type": "error",
        "message": (
            f"Agent reached the maximum number of steps ({max_turns}) without "
            f"completing the answer. Try a more specific question, or use "
            f"Simple mode for a direct retrieval."
        ),
    }


# ============================================================
# CLI testing (run a single question, print the trajectory)
# ============================================================
def _main():
    if len(sys.argv) < 2:
        print('Usage: python src/agent.py "your question here" [path/to/image.jpg]')
        sys.exit(1)

    question  = sys.argv[1]
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
