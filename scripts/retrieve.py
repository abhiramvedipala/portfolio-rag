"""Raw retrieval, no LLM. Prints the top matching chunks for a question.

Run:  python scripts/retrieve.py "What is Abhiram's strongest project?"
"""

import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "chroma_db"
COLLECTION = "portfolio"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 4


def retrieve(question, top_k=TOP_K):
    """Return the top_k chunks for a question, each with a similarity score."""
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_collection(COLLECTION)

    query_vector = model.encode(question).tolist()
    result = collection.query(query_embeddings=[query_vector], n_results=top_k)

    hits = []
    for id_, doc, meta, distance in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        # collection built with cosine space, so distance is 1 - cosine_similarity
        similarity = 1 - distance
        hits.append(
            {"id": id_, "section": meta["section"], "text": doc, "similarity": similarity}
        )
    return hits


def main():
    if len(sys.argv) < 2:
        print('usage: python scripts/retrieve.py "your question"')
        sys.exit(1)

    question = sys.argv[1]
    hits = retrieve(question)

    print(f"question: {question}\n")
    for rank, hit in enumerate(hits, start=1):
        print(f"#{rank}  {hit['id']}  similarity={hit['similarity']:.4f}  {hit['section']}")
        print(f"    {hit['text'][:200]}")
        print()


if __name__ == "__main__":
    main()
