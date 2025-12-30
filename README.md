# AI Chatbot with RAG

A simple AI chatbot that answers questions from your PDF documents using Retrieval-Augmented Generation (RAG).

## Tech Stack

- **LangChain** - Orchestration framework (LCEL)
- **OpenAI** - LLM (`gpt-4o-mini`) and Embeddings (`text-embedding-3-small`)
- **ChromaDB** - Local vector database
- **PyPDF** - PDF document parsing

## Project Structure

```
ai-chatbot-with-rag/
├── config.py           # Configuration & environment variables
├── document_loader.py  # PDF loading & text splitting
├── vector_store.py     # ChromaDB operations
├── chatbot.py          # RAGChatbot class with LCEL chain
├── main.py             # CLI entry point
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── docs/               # Place your PDF files here
└── chroma_db/          # Vector store (auto-created)
```

## Architecture

| Module | Responsibility |
|--------|----------------|
| `config.py` | Loads settings from `.env`, provides defaults |
| `document_loader.py` | Loads PDFs, splits into chunks |
| `vector_store.py` | Creates/loads ChromaDB vector store |
| `chatbot.py` | `RAGChatbot` class with conversation memory |
| `main.py` | CLI interface (swappable for web UI) |

## Getting Started

### Prerequisites

- Python 3.10+
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/LiteObject/ai-chatbot-with-rag.git
   cd ai-chatbot-with-rag
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   copy .env.example .env
   ```
   Edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-your-api-key-here
   ```

5. **Add PDF documents**
   
   Place your PDF files in the `docs/` folder.

### Usage

Run the chatbot:
```bash
python main.py
```

Example interaction:
```
==================================================
AI Chatbot with RAG
==================================================

Loading documents...
Loaded 15 pages from PDF files
Split into 42 chunks
Created new vector store

Chatbot ready! Type 'quit' to exit.
Type 'reset' to clear conversation history.

You: What is the main topic of the documents?
```

## How It Works

1. **Load PDFs**: Reads all PDF files from the `docs/` folder
2. **Chunk Text**: Splits documents into smaller chunks (1000 chars with 200 overlap)
3. **Create Embeddings**: Converts chunks to vectors using OpenAI embeddings
4. **Store in ChromaDB**: Persists vectors locally for fast retrieval
5. **Query**: Retrieves relevant chunks and sends them with your question to GPT

## Configuration

All settings are configured via environment variables in `.env`:

```bash
# Required
OPENAI_API_KEY=sk-your-api-key-here

# Optional (with defaults)
CHAT_MODEL=gpt-4o-mini
CHAT_TEMPERATURE=0.7
EMBEDDING_MODEL=text-embedding-3-small
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVER_K=3
```

## License

MIT