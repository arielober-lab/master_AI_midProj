"""
rag_system.py - The full RAG pipeline: retrieve + generate.

Ties together the Retriever (retrieve.py) and the answer generation
(generation.py) into a single answer() method that returns a structured
result, as required by the assignment:

    answer(question) -> {
        "answer": str,              # the generated answer text
        "sources": list[str],       # the product names used as sources
        "retrieved_chunks": list,   # the raw retrieved chunks with distances
    }

Usage as a script:
    python src/rag_system.py "What product contains MSM?"

Usage as a module (what the evaluation will do):
    from rag_system import RAGSystem
    rag = RAGSystem()
    result = rag.answer("your question")
"""

import sys

from retrieve import Retriever
from generation import generate_answer


class RAGSystem:
    """The complete RAG system: retrieval + generation."""

    def __init__(self):
        # The Retriever loads the model, index, and chunks once at startup
        self.retriever = Retriever()

    def answer(self, question, top_k=5):
        # 1. Retrieve the most relevant chunks
        retrieved = self.retriever.retrieve(question, top_k=top_k)

        # 2. Generate an answer grounded in those chunks
        answer_text = generate_answer(question, retrieved)

        # 3. Collect the unique source products used (for citation)
        sources = []
        seen = set()
        for r in retrieved:
            product = r["chunk"]["metadata"].get("product_name")
            if product and product not in seen:
                seen.add(product)
                sources.append(product)

        return {
            "answer": answer_text,
            "sources": sources,
            "retrieved_chunks": retrieved,
        }


def main():
    if len(sys.argv) < 2:
        print('Usage: python src/rag_system.py "your question here"')
        sys.exit(1)

    question = sys.argv[1]

    rag = RAGSystem()
    result = rag.answer(question)

    print("\n" + "=" * 60)
    print(f"QUESTION: {question}")
    print("=" * 60)
    print(f"\nANSWER:\n{result['answer']}")
    print(f"\nSOURCES: {', '.join(result['sources'])}")
    print(f"\n(Based on {len(result['retrieved_chunks'])} retrieved chunks)")


if __name__ == "__main__":
    main()
