from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str


class Source(BaseModel):
    id: str
    section: str
    similarity: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
