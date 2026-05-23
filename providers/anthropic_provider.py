"""
Anthropic Provider — wraps the Anthropic SDK (Claude models).

Key difference from Groq:
  - Anthropic separates the system prompt from the messages array.
  - The system message must be passed as a top-level `system` parameter,
    NOT inside the messages list.
  - Tool use / seed are handled differently (seed not supported natively).
"""
import logging
import anthropic
from .base import LLMProvider

logger = logging.getLogger("restaurant_api.providers.anthropic")


class AnthropicProvider(LLMProvider):

    def __init__(self, api_key: str, model_name: str):
        self.client     = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name

    def _split_messages(self, messages: list) -> tuple[str, list]:
        """
        Anthropic requires system prompt as a separate param.
        Splits messages into (system_text, conversation_messages).
        """
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        system_text  = "\n\n".join(system_parts) if system_parts else ""
        convo        = [m for m in messages if m.get("role") != "system"]

        # Anthropic requires alternating user/assistant roles.
        # Ensure the first message is always from the user.
        if convo and convo[0].get("role") != "user":
            convo = convo[1:]

        return system_text, convo

    def chat(
        self,
        messages:    list,
        temperature: float = 0.1,
        max_tokens:  int   = 350,
        seed:        int   = 42,  # ignored for Anthropic
    ) -> str:
        system_text, convo = self._split_messages(messages)
        if not convo:
            return ""
        try:
            kwargs = dict(
                model       = self.model_name,
                messages    = convo,
                temperature = temperature,
                max_tokens  = max_tokens,
            )
            if system_text:
                kwargs["system"] = system_text

            resp = self.client.messages.create(**kwargs)
            return resp.content[0].text if resp.content else ""
        except Exception as e:
            logger.error(f"[anthropic.chat] error: {e}")
            return ""

    def classify(
        self,
        prompt:     str,
        max_tokens: int = 80,
    ) -> str:
        try:
            resp = self.client.messages.create(
                model       = self.model_name,
                messages    = [{"role": "user", "content": prompt}],
                temperature = 0.0,
                max_tokens  = max_tokens,
            )
            return resp.content[0].text.strip().lower() if resp.content else ""
        except Exception as e:
            logger.error(f"[anthropic.classify] error: {e}")
            return ""
