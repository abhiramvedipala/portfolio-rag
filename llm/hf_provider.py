"""Hugging Face Inference API provider, via the router's OpenAI-compatible endpoint."""

import os

import requests
from dotenv import load_dotenv

from llm.base import LLMProvider

load_dotenv()

ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct:groq"


class HFProvider(LLMProvider):
    def __init__(self):
        self.token = os.getenv("HF_TOKEN")
        if not self.token:
            raise ValueError("HF_TOKEN is not set in .env")
        self.model = os.getenv("HF_MODEL", DEFAULT_MODEL)

    def generate(self, prompt: str) -> str:
        response = requests.post(
            ROUTER_URL,
            headers={"Authorization": f"Bearer {self.token}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
