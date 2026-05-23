"""
Groq Provider — wraps the Groq SDK (LLaMA models).
"""
import logging
from groq import Groq
from .base import LLMProvider

logger = logging.getLogger("restaurant_api.providers.groq")


class GroqProvider(LLMProvider):

    def __init__(self, api_key: str, model_name: str):
        self.client     = Groq(api_key=api_key)
        self.model_name = model_name

    def chat(
        self,
        messages:    list,
        temperature: float = 0.1,
        max_tokens:  int   = 350,
        seed:        int   = 42,
    ) -> str:
        try:
            resp = self.client.chat.completions.create(
                model       = self.model_name,
                messages    = messages,
                temperature = temperature,
                max_tokens  = max_tokens,
                seed        = seed,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"[groq.chat] error: {e}")
            return ""

    def classify(
        self,
        prompt:     str,
        max_tokens: int = 80,
    ) -> str:
        try:
            resp = self.client.chat.completions.create(
                model       = self.model_name,
                messages    = [{"role": "user", "content": prompt}],
                temperature = 0.0,
                max_tokens  = max_tokens,
                seed        = 42,
            )
            return resp.choices[0].message.content.strip().lower()
        except Exception as e:
            logger.error(f"[groq.classify] error: {e}")
            return ""
