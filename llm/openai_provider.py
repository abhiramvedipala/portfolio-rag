"""OpenAI provider. Functional, but off by default -- LLM_PROVIDER=huggingface
is what .env is set to right now. Set LLM_PROVIDER=openai to switch to this.

Uses raw requests instead of the openai SDK, since the SDK isn't in
requirements.txt and a single POST doesn't need a whole new dependency.
"""

import os

import requests
from dotenv import load_dotenv

from llm.base import LLMProvider

load_dotenv()

API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIProvider(LLMProvider):
    def __init__(self):
        self.key = os.getenv("OPENAI_API_KEY")
        if not self.key:
            raise ValueError("OPENAI_API_KEY is not set in .env")
        self.model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)

    def generate(self, prompt: str) -> str:
        response = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {self.key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
