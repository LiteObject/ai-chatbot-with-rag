"""
Embedding Provider module.
Factory for creating embedding instances from different providers.

Supported providers:
- OpenAI: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002
- Azure OpenAI: Azure-hosted OpenAI embedding models
- HuggingFace: sentence-transformers models (runs locally)
- Ollama: Local embedding models

To add a new provider:
1. Create a _get_<provider>_embeddings() function
2. Add it to the _EMBEDDING_FACTORIES dictionary
3. Update requirements.txt with the new package
"""

from langchain_core.embeddings import Embeddings

from config import (
    EMBEDDING_PROVIDER,
    EMBEDDING_MODEL,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_EMBEDDING_DEPLOYMENT,
    OLLAMA_BASE_URL,
)


def _get_openai_embeddings() -> Embeddings:
    """Create OpenAI embeddings instance."""
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def _get_azure_embeddings() -> Embeddings:
    """Create Azure OpenAI embeddings instance."""
    from langchain_openai import AzureOpenAIEmbeddings

    return AzureOpenAIEmbeddings(
        azure_deployment=AZURE_EMBEDDING_DEPLOYMENT or EMBEDDING_MODEL,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
    )


def _get_huggingface_embeddings() -> Embeddings:
    """Create HuggingFace embeddings instance (runs locally)."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _get_ollama_embeddings() -> Embeddings:
    """Create Ollama embeddings instance (runs locally)."""
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)


# Factory registry
_EMBEDDING_FACTORIES = {
    "openai": _get_openai_embeddings,
    "azure": _get_azure_embeddings,
    "huggingface": _get_huggingface_embeddings,
    "ollama": _get_ollama_embeddings,
}


def get_embeddings(provider: str | None = None) -> Embeddings:
    """Get an embeddings instance based on the configured provider.

    Args:
        provider: Embedding provider name. Uses EMBEDDING_PROVIDER from config if not specified.
                 Supported: openai, azure, huggingface, ollama

    Returns:
        A LangChain embeddings instance.

    Raises:
        ValueError: If the provider is not supported.

    Example:
        >>> embeddings = get_embeddings()  # Uses default from config
        >>> embeddings = get_embeddings("huggingface")  # Override to use HuggingFace
    """
    provider = (provider or EMBEDDING_PROVIDER).lower()

    factory = _EMBEDDING_FACTORIES.get(provider)
    if not factory:
        supported = ", ".join(_EMBEDDING_FACTORIES.keys())
        raise ValueError(
            f"Unknown embedding provider: {provider}. Supported: {supported}"
        )

    return factory()


def get_supported_providers() -> list[str]:
    """Get list of supported embedding providers."""
    return list(_EMBEDDING_FACTORIES.keys())
