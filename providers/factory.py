"""
Provider Factory — returns the correct LLMProvider based on the
`provider` field stored in the `llm_models` DB table.

Usage:
    from providers.factory import get_provider
    llm = get_provider(ctx.provider, ctx.api_key, ctx.model_name)
    reply = llm.chat(session.messages)
"""
from .base               import LLMProvider
from .groq_provider      import GroqProvider
from .anthropic_provider import AnthropicProvider

_REGISTRY: dict[str, type[LLMProvider]] = {
    "groq":      GroqProvider,
    "anthropic": AnthropicProvider,
}


def get_provider(provider: str, api_key: str, model_name: str) -> LLMProvider:
    """
    Returns an initialized LLMProvider instance.
    provider: 'groq' | 'anthropic'
    Raises ValueError if the provider is unknown.
    """
    cls = _REGISTRY.get(provider.lower())
    if not cls:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return cls(api_key=api_key, model_name=model_name)
