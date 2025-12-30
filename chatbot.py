"""
Chatbot module.
Contains the RAG chain and conversation logic.

This module is provider-agnostic - it works with any LLM or vector store
that LangChain supports.
"""

from langchain_core.vectorstores import VectorStore
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from config import RETRIEVER_K
from llm_provider import get_llm


# Store for chat history sessions
_chat_history_store: dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    """Get or create chat history for a session."""
    if session_id not in _chat_history_store:
        _chat_history_store[session_id] = ChatMessageHistory()
    return _chat_history_store[session_id]


def clear_session_history(session_id: str) -> None:
    """Clear chat history for a session."""
    if session_id in _chat_history_store:
        del _chat_history_store[session_id]


def clear_all_history() -> None:
    """Clear all chat history."""
    _chat_history_store.clear()


class RAGChatbot:
    """RAG-based chatbot with conversation memory.

    This class is vector store agnostic - it accepts any LangChain VectorStore
    implementation (Chroma, FAISS, Pinecone, Qdrant, etc.).
    """

    def __init__(self, vector_store: VectorStore):
        """Initialize the chatbot with a vector store.

        Args:
            vector_store: Any LangChain VectorStore implementation.
        """
        self.vector_store = vector_store
        self.chain = self._create_chain()

    def _create_chain(self):
        """Create the conversational RAG chain using LCEL."""
        llm = get_llm()
        retriever = self.vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})

        # System prompt for RAG
        system_prompt = """You are a helpful assistant that answers questions based on the provided context.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, say that you don't know.

Context:
{context}"""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Create the RAG chain
        rag_chain = (
            RunnablePassthrough.assign(
                context=lambda x: format_docs(retriever.invoke(x["input"]))
            )
            | prompt
            | llm
            | StrOutputParser()
        )

        # Wrap with message history
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        return conversational_rag_chain

    def chat(self, question: str, session_id: str = "default") -> str:
        """Send a question to the chatbot and get a response."""
        response = self.chain.invoke(
            {"input": question}, config={"configurable": {"session_id": session_id}}
        )
        return response

    def reset_conversation(self, session_id: str = "default") -> None:
        """Reset the conversation history for a session."""
        clear_session_history(session_id)
