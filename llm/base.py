"""Abstract interface every LLM provider implements."""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send a prompt to the model and return its text response."""
        raise NotImplementedError
