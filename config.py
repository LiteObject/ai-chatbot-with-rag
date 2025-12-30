"""
Configuration module for the RAG Chatbot.
Loads settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Directory paths
BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"
VECTOR_STORE_DIR = BASE_DIR / "vector_store_db"

# Vector store configuration
VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "chroma")  # chroma, faiss

# Document processing
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# LLM Provider configuration
# Supported: openai, azure, anthropic, ollama
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0.7"))

# Embedding Provider configuration
# Supported: openai, azure, huggingface, ollama
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Azure-specific configuration (if using Azure)
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "")

# Ollama-specific configuration (if using Ollama)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Retrieval configuration
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "3"))


def validate_config() -> bool:
    """Validate that required configuration is present."""
    provider = LLM_PROVIDER.lower()

    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            print("\nError: OPENAI_API_KEY not found!")
            print("Please create a .env file with your OpenAI API key.")
            return False
    elif provider == "azure":
        if not os.getenv("AZURE_OPENAI_API_KEY"):
            print("\nError: AZURE_OPENAI_API_KEY not found!")
            return False
        if not AZURE_OPENAI_ENDPOINT:
            print("\nError: AZURE_OPENAI_ENDPOINT not found!")
            return False
    elif provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("\nError: ANTHROPIC_API_KEY not found!")
            return False
    elif provider == "ollama":
        pass  # Ollama doesn't require API key
    else:
        print(f"\nError: Unknown LLM provider: {provider}")
        print("Supported providers: openai, azure, anthropic, ollama")
        return False

    return True
