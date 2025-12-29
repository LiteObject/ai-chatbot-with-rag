# AI Chatbot with RAG

A simple AI chatbot that answers questions from your PDF documents using Retrieval-Augmented Generation (RAG).

## Tech Stack

- **LangChain** - Orchestration framework
- **OpenAI** - LLM (`gpt-4o-mini`) and Embeddings (`text-embedding-3-small`)
- **ChromaDB** - Local vector database
- **PyPDF** - PDF document parsing

## Project Structure

```
ai-chatbot-with-rag/
├── chatbot.py          # Main application
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── docs/               # Place your PDF files here
└── chroma_db/          # Vector store (auto-created)
```

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
python chatbot.py
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

You: What is the main topic of the documents?