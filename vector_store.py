"""
Vector store module.
Provides a factory for creating/loading vector stores.

Supports multiple backends:
- ChromaDB (default): Local, persistent, easy setup
- FAISS: Fast, in-memory or persistent, good for large datasets

To add a new vector store:
1. Create a new _create_<store>() and _load_<store>() function
2. Add the store type to the factory functions
3. Update requirements.txt with the new package
"""

from typing import Optional

from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings

from config import VECTOR_STORE_DIR, VECTOR_STORE_TYPE
from embedding_provider import get_embeddings


# =============================================================================
# ChromaDB Implementation
# =============================================================================


def _create_chroma(chunks: list, embeddings: Embeddings) -> VectorStore:
    """Create a new ChromaDB vector store."""
    from langchain_chroma import Chroma

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTOR_STORE_DIR / "chroma"),
    )
    return vector_store


def _load_chroma(embeddings: Embeddings) -> Optional[VectorStore]:
    """Load an existing ChromaDB vector store."""
    from langchain_chroma import Chroma

    chroma_path = VECTOR_STORE_DIR / "chroma"
    if not chroma_path.exists():
        return None

    vector_store = Chroma(
        persist_directory=str(chroma_path), embedding_function=embeddings
    )
    return vector_store


# =============================================================================
# FAISS Implementation
# =============================================================================


def _create_faiss(chunks: list, embeddings: Embeddings) -> VectorStore:
    """Create a new FAISS vector store."""
    from langchain_community.vectorstores import FAISS

    vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
    # Persist to disk
    faiss_path = VECTOR_STORE_DIR / "faiss"
    faiss_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(faiss_path))
    return vector_store


def _load_faiss(embeddings: Embeddings) -> Optional[VectorStore]:
    """Load an existing FAISS vector store."""
    from langchain_community.vectorstores import FAISS

    faiss_path = VECTOR_STORE_DIR / "faiss"
    if not (faiss_path / "index.faiss").exists():
        return None

    vector_store = FAISS.load_local(
        str(faiss_path), embeddings, allow_dangerous_deserialization=True
    )
    return vector_store


# =============================================================================
# Factory Functions
# =============================================================================

_CREATORS = {
    "chroma": _create_chroma,
    "faiss": _create_faiss,
}

_LOADERS = {
    "chroma": _load_chroma,
    "faiss": _load_faiss,
}


def create_vector_store(
    chunks: list,
    embeddings: Optional[Embeddings] = None,
    store_type: Optional[str] = None,
) -> Optional[VectorStore]:
    """Create a new vector store from document chunks.

    Args:
        chunks: List of document chunks to store.
        embeddings: Embeddings model (uses configured provider if not provided).
        store_type: Type of vector store ('chroma', 'faiss').
                   Uses VECTOR_STORE_TYPE from config if not provided.

    Returns:
        VectorStore instance or None if no chunks provided.
    """
    if embeddings is None:
        embeddings = get_embeddings()

    if not chunks:
        print("No chunks provided to create vector store")
        return None

    store_type = store_type or VECTOR_STORE_TYPE
    creator = _CREATORS.get(store_type)

    if not creator:
        raise ValueError(
            f"Unknown vector store type: {store_type}. "
            f"Supported: {list(_CREATORS.keys())}"
        )

    # Ensure directory exists
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    vector_store = creator(chunks, embeddings)
    print(f"Created new {store_type} vector store")
    return vector_store


def load_vector_store(
    embeddings: Optional[Embeddings] = None, store_type: Optional[str] = None
) -> Optional[VectorStore]:
    """Load an existing vector store from disk.

    Args:
        embeddings: Embeddings model (uses configured provider if not provided).
        store_type: Type of vector store ('chroma', 'faiss').
                   Uses VECTOR_STORE_TYPE from config if not provided.

    Returns:
        VectorStore instance or None if not found.
    """
    if embeddings is None:
        embeddings = get_embeddings()

    store_type = store_type or VECTOR_STORE_TYPE
    loader = _LOADERS.get(store_type)

    if not loader:
        raise ValueError(
            f"Unknown vector store type: {store_type}. "
            f"Supported: {list(_LOADERS.keys())}"
        )

    vector_store = loader(embeddings)

    if vector_store:
        print(f"Loaded existing {store_type} vector store")
    else:
        print(f"No existing {store_type} vector store found")

    return vector_store


def get_or_create_vector_store(
    chunks: list,
    embeddings: Optional[Embeddings] = None,
    store_type: Optional[str] = None,
) -> Optional[VectorStore]:
    """Get existing vector store or create new one from chunks.

    Args:
        chunks: List of document chunks (used only if creating new store).
        embeddings: Embeddings model (uses configured provider if not provided).
        store_type: Type of vector store ('chroma', 'faiss').

    Returns:
        VectorStore instance or None if failed.
    """
    if embeddings is None:
        embeddings = get_embeddings()

    if chunks:
        return create_vector_store(chunks, embeddings, store_type)
    else:
        return load_vector_store(embeddings, store_type)
