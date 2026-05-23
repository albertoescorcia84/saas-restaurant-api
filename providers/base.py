"""
Base LLM Provider — abstract interface.
All providers must implement chat() and classify().
"""
from abc import ABC, abstractmethod


class LLMProvider(ABC):

    @abstractmethod
    def chat(
        self,
        messages:    list,
        temperature: float = 0.1,
        max_tokens:  int   = 350,
        seed:        int   = 42,
    ) -> str:
        """
        Send a conversation to the model and return the text reply.
        messages: list of {"role": "system"|"user"|"assistant", "content": str}
        """
        pass

    @abstractmethod
    def classify(
        self,
        prompt:     str,
        max_tokens: int = 80,
    ) -> str:
        """
        Single-turn classification call.
        Returns the raw text response (caller parses it).
        """
        pass
