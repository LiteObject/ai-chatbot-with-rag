"""
Main entry point for the RAG Chatbot CLI.
"""

from config import validate_config, DOCS_DIR
from document_loader import load_and_split_documents
from vector_store import get_or_create_vector_store
from chatbot import RAGChatbot


def main():
    """Main function to run the chatbot."""
    print("=" * 50)
    print("AI Chatbot with RAG")
    print("=" * 50)

    # Validate configuration
    if not validate_config():
        return

    # Load and process documents
    print("\nLoading documents...")
    chunks = load_and_split_documents(DOCS_DIR)

    # Get or create vector store
    vector_store = get_or_create_vector_store(chunks)

    if not vector_store:
        print("\nPlease add PDF files to the 'docs' folder and restart.")
        return

    # Create chatbot
    chatbot = RAGChatbot(vector_store)
    print("\nChatbot ready! Type 'quit' to exit.")
    print("Type 'reset' to clear conversation history.\n")

    # Chat loop
    while True:
        try:
            question = input("You: ").strip()

            if not question:
                continue

            if question.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if question.lower() == "reset":
                chatbot.reset_conversation()
                print("Conversation history cleared.\n")
                continue

            response = chatbot.chat(question)
            print(f"\nAssistant: {response}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
