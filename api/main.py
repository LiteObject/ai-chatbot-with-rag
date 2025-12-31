import sys
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent directory to path to allow imports from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot import RAGChatbot
from vector_store import load_vector_store
from cache.response_cache import ResponseCache
from middleware.rate_limit import RateLimiter

# Global instances
chatbot = None
cache = None
rate_limiter = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global chatbot, cache, rate_limiter

    # Initialize Vector Store and Chatbot
    # Note: In a real production scenario, we might want to initialize this lazily
    # or ensure the vector store connection is robust.
    vector_store = load_vector_store()
    if not vector_store:
        print("Warning: No vector store found. Chatbot will not work correctly.")

    chatbot = RAGChatbot(vector_store)

    # Initialize Cache
    cache = ResponseCache()

    # Initialize Rate Limiter
    rate_limiter = RateLimiter(cache.redis)

    yield

    # Cleanup
    if cache and cache.redis:
        await cache.redis.close()


app = FastAPI(title="RAG Chatbot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[str] = []


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request):
    session_id = request.session_id or str(uuid.uuid4())

    # Rate Limiting
    if rate_limiter:
        await rate_limiter.check_rate_limit(req)

    # Check Cache
    if cache:
        cached = await cache.get_cached_response(request.message)
        if cached:
            return ChatResponse(answer=cached, session_id=session_id)

    # Generate Response
    if chatbot:
        answer, sources = await chatbot.chat_async(
            question=request.message, session_id=session_id
        )
    else:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")

    # Cache Response
    if cache:
        await cache.cache_response(request.message, answer)

    return ChatResponse(answer=answer, session_id=session_id, sources=sources)


@app.get("/health")
async def health():
    return {"status": "healthy"}
