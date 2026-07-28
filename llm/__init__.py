"""Factory that picks the LLM provider based on LLM_PROVIDER in .env.

Only the provider actually selected gets constructed, so only its key gets
validated -- an empty OPENAI_API_KEY never breaks a huggingface run, and vice
versa.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def get_llm():
    provider = os.getenv("LLM_PROVIDER", "huggingface").lower()

    if provider == "huggingface":
        from llm.hf_provider import HFProvider

        return HFProvider()

    if provider == "openai":
        from llm.openai_provider import OpenAIProvider

        return OpenAIProvider()

    raise ValueError(f"unknown LLM_PROVIDER: {provider!r}")
