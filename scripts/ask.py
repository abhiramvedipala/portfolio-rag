"""The whole RAG pipeline in one file: retrieve chunks, ask the LLM, print
the answer and which chunks it was based on.

Run:  python scripts/ask.py "What is Abhiram's strongest project?"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieve import retrieve  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm import get_llm  # noqa: E402

PROMPT_TEMPLATE = """Answer the question using only the context below. If the
context doesn't contain the answer, say you don't know.

Context:
{context}

Question: {question}

Answer:"""


def ask(question):
    """Retrieve context, generate an answer, return (answer, chunks used)."""
    chunks = retrieve(question)
    context = "\n\n---\n\n".join(c["text"] for c in chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    answer = get_llm().generate(prompt)
    return answer, chunks


def main():
    if len(sys.argv) < 2:
        print('usage: python scripts/ask.py "your question"')
        sys.exit(1)

    question = sys.argv[1]
    answer, chunks = ask(question)

    print(f"question: {question}\n")
    print(f"answer:\n{answer}\n")
    print("sources:")
    for c in chunks:
        print(f"  {c['id']}  similarity={c['similarity']:.4f}  {c['section']}")


if __name__ == "__main__":
    main()
