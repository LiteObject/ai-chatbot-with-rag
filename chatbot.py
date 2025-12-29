"""
AI Chatbot with RAG (Retrieval-Augmented Generation)
Uses LangChain + OpenAI + ChromaDB to answer questions from PDF documents.
"""

import os
from pathlib import Path

from typing import Optional

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

# Configuration
DOCS_DIR = Path(__file__).parent / "docs"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_documents(docs_dir: Path) -> list:
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
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def create_vector_store(chunks: list, embeddings: OpenAIEmbeddings) -> Optional[Chroma]:
    """Create or load ChromaDB vector store."""
    if chunks:
        # Create new vector store from documents
        vector_store = Chroma.from_documents(
            documents=chunks, embedding=embeddings, persist_directory=str(CHROMA_DIR)
        )
        print("Created new vector store")
    elif CHROMA_DIR.exists():
        # Load existing vector store
        vector_store = Chroma(
            persist_directory=str(CHROMA_DIR), embedding_function=embeddings
        )
        print("Loaded existing vector store")
    else:
        print("No documents found and no existing vector store!")
        return None

    return vector_store


def create_chatbot(vector_store: Chroma) -> ConversationalRetrievalChain:
    """Create the conversational RAG chatbot."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )

    chatbot = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )

    return chatbot


def chat(chatbot: ConversationalRetrievalChain, question: str) -> str:
    """Send a question to the chatbot and get a response."""
    response = chatbot.invoke({"question": question})
    return response["answer"]


def main():
    """Main function to run the chatbot."""
    print("=" * 50)
    print("AI Chatbot with RAG")
    print("=" * 50)

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\nError: OPENAI_API_KEY not found!")
        print("Please create a .env file with your OpenAI API key.")
        print("See .env.example for the format.")
        return

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Load and process documents
    print("\nLoading documents...")
    documents = load_documents(DOCS_DIR)

    if documents:
        chunks = split_documents(documents)
        vector_store = create_vector_store(chunks, embeddings)
    else:
        # Try to load existing vector store
        vector_store = create_vector_store([], embeddings)

    if not vector_store:
        print("\nPlease add PDF files to the 'docs' folder and restart.")
        return

    # Create chatbot
    chatbot = create_chatbot(vector_store)
    print("\nChatbot ready! Type 'quit' to exit.\n")

    # Chat loop
    while True:
        try:
            question = input("You: ").strip()

            if not question:
                continue

            if question.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            response = chat(chatbot, question)
            print(f"\nAssistant: {response}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
