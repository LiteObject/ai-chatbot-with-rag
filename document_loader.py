"""
Document loading and text splitting module.
Handles PDF loading and chunking for RAG.
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def load_documents(docs_dir: Path = DOCS_DIR) -> list:
    """Load all PDF documents from the specified directory."""
    if not docs_dir.exists():
        print(f"Creating documents directory: {docs_dir}")
        docs_dir.mkdir(parents=True, exist_ok=True)
        return []

    loader = PyPDFDirectoryLoader(str(docs_dir))
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from PDF files")
    return documents


def split_documents(documents: list) -> list:
    """Split documents into smaller chunks for better retrieval."""
    if not documents:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def load_and_split_documents(docs_dir: Path = DOCS_DIR) -> list:
    """Load and split documents in one call."""
    documents = load_documents(docs_dir)
    return split_documents(documents)
