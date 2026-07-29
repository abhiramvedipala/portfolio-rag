"""FastAPI backend. Reuses ask() from scripts/ask.py -- no duplicated logic
between the CLI and the API.

Run:  uvicorn api.main:app --reload
"""

import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from ask import ask  # noqa: E402

from api.schemas import ChatRequest, ChatResponse

app = FastAPI()

# CORS_ORIGINS is a comma-separated list, e.g. "https://abhiramreddy.dev".
# Defaults to "*" so local/dev testing keeps working without config.
_origins = os.getenv("CORS_ORIGINS", "*")
allow_origins = ["*"] if _origins == "*" else [o.strip() for o in _origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    answer, chunks = ask(request.question)
    sources = [
        {"id": c["id"], "section": c["section"], "similarity": c["similarity"]}
        for c in chunks
    ]
    return ChatResponse(answer=answer, sources=sources)
