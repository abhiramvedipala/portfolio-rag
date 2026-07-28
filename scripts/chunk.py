"""Split the knowledge doc into retrieval-sized chunks.

NOTE: this file was recreated because scripts/chunk.py was not present in the
repo or in git history when step 1 was built. If you find your original
version, replace this file with it -- embed.py only needs chunk_document()
to return a list of chunks.

A chunk is a dict:
    {"id": "chunk-000", "section": "Projects / TypeForge", "text": "..."}

The section name is prepended to the chunk text so the embedding carries the
topic, not just the sentence. "Built with React and TanStack Start" means very
little on its own; "Projects / TypeForge -- Built with React..." matches a
question about TypeForge much better.
"""

from pathlib import Path

DOC_PATH = Path(__file__).resolve().parent.parent / "data" / "about_abhiram.md"
MAX_CHARS = 900


def chunk_document(path=DOC_PATH, max_chars=MAX_CHARS):
    """Split a markdown doc into chunks, one list entry per chunk.

    Splits on headings first, then on blank lines once a chunk gets long.
    Line-based rather than paragraph-based on purpose: this doc writes
    "## Quick Facts" immediately followed by its bullets with no blank line
    between them, so splitting on blank lines alone glues a heading to its
    body and loses the section name.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()

    chunks = []
    heading, subheading = "Overview", None
    buffer = []

    def section():
        return f"{heading} / {subheading}" if subheading else heading

    def flush():
        body = "\n".join(buffer).strip()
        buffer.clear()
        if body:
            chunks.append(
                {
                    "id": f"chunk-{len(chunks):03d}",
                    "section": section(),
                    "text": f"{section()}\n\n{body}",
                }
            )

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("#"):
            flush()
            level = len(stripped) - len(stripped.lstrip("#"))
            title = stripped.lstrip("#").strip()
            if level <= 2:
                heading, subheading = title, None
            else:
                subheading = title
            continue

        # blank line is the only place it is safe to cut mid-section
        if not stripped and sum(len(x) for x in buffer) >= max_chars:
            flush()
            continue

        buffer.append(line)

    flush()
    return chunks


if __name__ == "__main__":
    result = chunk_document()
    assert result, "no chunks produced"
    assert all(c["text"].strip() for c in result), "empty chunk produced"
    assert len({c["id"] for c in result}) == len(result), "duplicate chunk ids"
    print(f"{len(result)} chunks")
    for c in result:
        print(f"  {c['id']}  {len(c['text']):4d} chars  {c['section']}")
