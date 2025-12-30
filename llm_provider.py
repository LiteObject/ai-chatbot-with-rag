"""
LLM Provider module.
Factory for creating LLM instances from different providers.

Supported providers:
- OpenAI: GPT-4, GPT-4o, GPT-4o-mini, etc.
- Azure OpenAI: Azure-hosted OpenAI models
- Anthropic: Claude models
- Ollama: Local models (Llama, Mistral, etc.)

To add a new provider:
1. Create a _get_<provider>_llm() function
2. Add it to the _LLM_FACTORIES dictionary
3. Update requirements.txt with the new package
"""

from langchain_core.language_models.chat_models import BaseChatModel

from config import (
    LLM_PROVIDER,
    CHAT_MODEL,
    CHAT_TEMPERATURE,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_DEPLOYMENT_NAME,
    OLLAMA_BASE_URL,
)


def _get_openai_llm() -> BaseChatModel:
    """Create an OpenAI LLM instance."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=CHAT_MODEL, temperature=CHAT_TEMPERATURE)


def _get_azure_llm() -> BaseChatModel:
    """Create an Azure OpenAI LLM instance."""
    from langchain_openai import AzureChatOpenAI

    return AzureChatOpenAI(
        azure_deployment=AZURE_DEPLOYMENT_NAME or CHAT_MODEL,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
        temperature=CHAT_TEMPERATURE,
    )


def _get_anthropic_llm() -> BaseChatModel:
    """Create an Anthropic Claude LLM instance."""
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=CHAT_MODEL, temperature=CHAT_TEMPERATURE)


def _get_ollama_llm() -> BaseChatModel:
    """Create an Ollama LLM instance for local models."""
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=CHAT_MODEL, base_url=OLLAMA_BASE_URL, temperature=CHAT_TEMPERATURE
    )


# Factory registry
_LLM_FACTORIES = {
    "openai": _get_openai_llm,
    "azure": _get_azure_llm,
    "anthropic": _get_anthropic_llm,
    "ollama": _get_ollama_llm,
}


def get_llm(provider: str | None = None) -> BaseChatModel:
    """Get an LLM instance based on the configured provider.

    Args:
        provider: LLM provider name. Uses LLM_PROVIDER from config if not specified.
                 Supported: openai, azure, anthropic, ollama

    Returns:
        A LangChain chat model instance.

    Raises:
        ValueError: If the provider is not supported.

    Example:
        >>> llm = get_llm()  # Uses default from config
        >>> llm = get_llm("ollama")  # Override to use Ollama
    """
    provider = (provider or LLM_PROVIDER).lower()

    factory = _LLM_FACTORIES.get(provider)
    if not factory:
        supported = ", ".join(_LLM_FACTORIES.keys())
        raise ValueError(f"Unknown LLM provider: {provider}. Supported: {supported}")

    return factory()


def get_supported_providers() -> list[str]:
    """Get list of supported LLM providers."""
    return list(_LLM_FACTORIES.keys())
