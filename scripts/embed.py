"""Embed every chunk of the knowledge doc and store it in a local ChromaDB.

Run:  python scripts/embed.py

Rebuilds the collection from scratch each run, so running it twice is safe and
never creates duplicates.
"""

import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# scripts/ on the path so `from chunk import ...` works no matter what
# directory this is run from, and later when api/ imports scripts modules.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunk import chunk_document  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "chroma_db"
COLLECTION = "portfolio"
MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    chunks = chunk_document()
    print(f"{len(chunks)} chunks from the knowledge doc")

    print(f"loading model {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=str(DB_PATH))
    try:
        client.delete_collection(COLLECTION)
        print(f"dropped existing collection '{COLLECTION}'")
    except Exception:
        pass  # first run, nothing to drop

    # cosine distance, not the l2 default -- MiniLM vectors are meant to be
    # compared by angle, and step 2 reports similarity scores from this.
    collection = client.create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"}
    )

    for i, chunk in enumerate(chunks, start=1):
        vector = model.encode(chunk["text"]).tolist()
        collection.add(
            ids=[chunk["id"]],
            documents=[chunk["text"]],
            embeddings=[vector],
            metadatas=[{"section": chunk["section"]}],
        )
        print(f"  [{i}/{len(chunks)}] {chunk['id']}  {chunk['section']}")

    stored = collection.count()
    assert stored == len(chunks), f"stored {stored}, expected {len(chunks)}"
    print(f"done. {stored} entries in '{COLLECTION}' at {DB_PATH}")


if __name__ == "__main__":
    main()
