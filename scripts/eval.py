"""Retrieval recall eval. For each question in data/eval_questions.json,
checks whether the expected section shows up anywhere in the top 4 chunks.

Run:  python scripts/eval.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieve import retrieve  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = ROOT / "data" / "eval_questions.json"


def main():
    eval_set = json.loads(EVAL_PATH.read_text())

    hits = 0
    for case in eval_set:
        question = case["question"]
        expected = case["expected_section"]

        chunks = retrieve(question)
        found = any(expected in c["section"] for c in chunks)
        hits += found

        status = "PASS" if found else "FAIL"
        print(f"[{status}] {question}")
        if not found:
            print(f"         expected section containing: {expected!r}")
            print(f"         got: {[c['section'] for c in chunks]}")

    recall = hits / len(eval_set)
    print(f"\nrecall@4: {hits}/{len(eval_set)} = {recall:.2%}")


if __name__ == "__main__":
    main()
