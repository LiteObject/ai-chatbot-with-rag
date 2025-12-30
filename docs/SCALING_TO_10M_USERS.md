# Scaling RAG Chatbot for 300,000 Daily Active Users on Azure

This document outlines the architectural changes, Azure services, and implementation steps required to scale this RAG chatbot from a single-user CLI application to a **public, stateless** production system supporting **300,000 daily active users**.

---

## Key Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Stateless** | No user accounts, no conversation history stored |
| **Privacy-first** | No PII collected, no tracking |
| **Simple document corpus** | Small, curated set of PDFs (admin-managed) |
| **Cost-efficient** | Aggressive caching, minimal infrastructure |

---

## Table of Contents

1. [Current Architecture Limitations](#current-architecture-limitations)
2. [Target Architecture Overview](#target-architecture-overview)
3. [Azure Services Required](#azure-services-required)
4. [Component-by-Component Migration](#component-by-component-migration)
5. [Implementation Phases](#implementation-phases)
6. [Cost Estimation](#cost-estimation)
7. [Performance Considerations](#performance-considerations)
8. [Security & Compliance](#security--compliance)
9. [Monitoring & Observability](#monitoring--observability)

---

## Current Architecture Limitations

| Component | Current State | Limitation |
|-----------|--------------|------------|
| **Interface** | CLI (`main.py`) | Single user, no concurrent access |
| **Conversation Memory** | In-memory Python dict | Per-session only, not scalable |
| **Vector Store** | Local ChromaDB (SQLite) | Single-node, no horizontal scaling |
| **LLM Calls** | Synchronous blocking | One request at a time |
| **Caching** | None | Repeated API calls, high latency and cost |

---

## Target Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Azure Front Door (CDN + WAF)                       │
│                              - Global load balancing                            │
│                              - DDoS protection                                  │
│                              - SSL termination                                  │
│                              - Basic rate limiting                              │
└─────────────────────────────────────────────────────────────────────────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
                    ▼                    ▼                    ▼
          ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
          │  Azure Container│  │  Azure Container│  │  Azure Container│
          │  Apps (FastAPI) │  │  Apps (FastAPI) │  │  Apps (FastAPI) │
          │  - Stateless    │  │  - Stateless    │  │  - Stateless    │
          │  - Auto-scale   │  │  - Auto-scale   │  │  - Auto-scale   │
          └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
                   │                    │                    │
                   └────────────────────┼────────────────────┘
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              │                         │                         │
              ▼                         ▼                         ▼
    ┌───────────────┐         ┌───────────────┐         ┌───────────────┐
    │ Azure Cache   │         │ Azure AI      │         │ Azure OpenAI  │
    │ for Redis     │         │ Search        │         │ Service       │
    │               │         │ (Vector DB)   │         │               │
    │ - Response    │         │ - Embeddings  │         │ - GPT-4o-mini │
    │   cache       │         │ - Small index │         │ - Embeddings  │
    │ - Rate limit  │         │               │         │               │
    └───────────────┘         └───────────────┘         └───────────────┘
```

---

## Azure Services Required

### Core Services

| Service | Purpose | SKU Recommendation | Est. Monthly Cost |
|---------|---------|-------------------|-------------------|
| **Azure Container Apps** | Stateless API hosting | Consumption plan | $500 - $1,500 |
| **Azure OpenAI Service** | LLM and embeddings | Pay-as-you-go (S0) | $8,000 - $15,000 |
| **Azure AI Search** | Vector database | S1 (1 unit) | $250 |
| **Azure Cache for Redis** | Response and rate limit cache | Standard C2 (6GB) | $200 |

### Supporting Services

| Service | Purpose | Est. Monthly Cost |
|---------|---------|-------------------|
| **Azure Front Door** | CDN, WAF, DDoS, rate limiting | $350 |
| **Azure Key Vault** | Secrets management | $10 |
| **Azure Monitor + App Insights** | Observability | $200 |

---

## Component-by-Component Migration

### 1. Web API Layer (Replace CLI)

**Current:** `main.py` with CLI input loop

**Target:** Stateless FastAPI with no authentication

```python
# api/main.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid

app = FastAPI(title="RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for your frontend domain
    allow_methods=["POST"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None  # Optional, for multi-turn in same session

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[str] = []

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Stateless chat endpoint.
    - No user tracking
    - Session ID only for optional multi-turn context (in-memory, ephemeral)
    - No data persisted
    """
    session_id = request.session_id or str(uuid.uuid4())
    
    # Check response cache first
    cached = await cache.get_cached_response(request.message)
    if cached:
        return ChatResponse(answer=cached, session_id=session_id)
    
    # Generate response
    answer, sources = await chatbot.chat_async(
        question=request.message,
        session_id=session_id
    )
    
    # Cache for future identical questions
    await cache.cache_response(request.message, answer)
    
    return ChatResponse(
        answer=answer,
        session_id=session_id,
        sources=sources
    )

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

**Azure Deployment:**

- Azure Container Apps with min 2 replicas
- Scale rule: HTTP concurrent requests > 50
- Max replicas: 100

---

### 2. Vector Store (Replace ChromaDB)

**Current:** Local ChromaDB with SQLite backend

**Target:** Azure AI Search (pre-indexed at deployment)

```python
# vector_store_azure.py
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from langchain_core.vectorstores import VectorStore

class AzureAISearchVectorStore(VectorStore):
    def __init__(self):
        self.client = SearchClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            index_name="rag-documents",
            credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
        )
    
    async def similarity_search(
        self, 
        query_embedding: list[float], 
        k: int = 4
    ) -> list[Document]:
        results = self.client.search(
            search_text=None,
            vector_queries=[{
                "vector": query_embedding,
                "k_nearest_neighbors": k,
                "fields": "content_vector"
            }],
            select=["content", "source", "page"]
        )
        return [
            Document(
                page_content=r["content"],
                metadata={"source": r["source"], "page": r["page"]}
            )
            for r in results
        ]
```

**Document Indexing (One-time at deployment):**

```python
# scripts/index_documents.py
"""
Run this script during CI/CD deployment when documents change.
No need for runtime ingestion pipeline.
"""

async def index_all_documents():
    # Load PDFs from local docs/ folder
    documents = load_documents("docs/")
    chunks = split_documents(documents)
    
    # Generate embeddings
    embeddings = await generate_embeddings(chunks)
    
    # Upload to Azure AI Search
    await upload_to_search_index(chunks, embeddings)
    
    print(f"Indexed {len(chunks)} chunks from {len(documents)} documents")

if __name__ == "__main__":
    asyncio.run(index_all_documents())
```

---

### 3. Caching Layer (Critical for Cost and Performance)

**Purpose:** Cache responses to reduce LLM calls by 60-80%

```python
# cache/response_cache.py
import hashlib
import json
from redis import asyncio as aioredis

class ResponseCache:
    def __init__(self):
        self.redis = aioredis.from_url(
            os.getenv("REDIS_URL"),
            decode_responses=True
        )
    
    def _hash_question(self, question: str) -> str:
        """Normalize and hash question for cache key."""
        normalized = question.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]
    
    async def get_cached_response(self, question: str) -> str | None:
        """Check if we have answered this question before."""
        key = f"resp:{self._hash_question(question)}"
        return await self.redis.get(key)
    
    async def cache_response(
        self, 
        question: str, 
        answer: str,
        ttl: int = 3600  # 1 hour default
    ):
        """Cache response for identical future questions."""
        key = f"resp:{self._hash_question(question)}"
        await self.redis.setex(key, ttl, answer)
    
    async def get_cached_embedding(self, text: str) -> list[float] | None:
        """Cache embeddings to reduce API calls."""
        key = f"emb:{self._hash_question(text)}"
        cached = await self.redis.get(key)
        return json.loads(cached) if cached else None
    
    async def cache_embedding(self, text: str, embedding: list[float]):
        key = f"emb:{self._hash_question(text)}"
        await self.redis.setex(key, 86400, json.dumps(embedding))  # 24h TTL
```

**Cache Strategy:**

| Cache Type | TTL | Expected Hit Rate |
|------------|-----|-------------------|
| Exact question match | 1 hour | 40-60% |
| Embeddings | 24 hours | 80%+ |
| Rate limit counters | 1 minute | N/A |

---

### 4. Rate Limiting (Abuse Protection)

Since there is no authentication, rate limiting by IP is essential:

```python
# middleware/rate_limit.py
from fastapi import Request, HTTPException

class RateLimiter:
    def __init__(self, redis, requests_per_minute: int = 10):
        self.redis = redis
        self.rpm = requests_per_minute
    
    async def check_rate_limit(self, request: Request):
        """Limit requests per IP address."""
        client_ip = request.client.host
        key = f"ratelimit:{client_ip}"
        
        current = await self.redis.incr(key)
        if current == 1:
            await self.redis.expire(key, 60)  # 1 minute window
        
        if current > self.rpm:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please wait a minute."
            )
```

---

### 5. Session Memory (Ephemeral, In-Memory)

For multi-turn conversations within a session, use a simple TTL-based memory:

```python
# memory/ephemeral_memory.py

class EphemeralMemory:
    """
    Short-lived conversation memory.
    - Stored in Redis with short TTL
    - No permanent storage
    - Automatically expires
    """
    
    def __init__(self, redis, ttl: int = 900):  # 15 minutes
        self.redis = redis
        self.ttl = ttl
    
    async def get_history(self, session_id: str) -> list[dict]:
        key = f"session:{session_id}"
        data = await self.redis.get(key)
        return json.loads(data) if data else []
    
    async def add_message(self, session_id: str, role: str, content: str):
        key = f"session:{session_id}"
        history = await self.get_history(session_id)
        
        # Keep only last 10 messages
        history.append({"role": role, "content": content})
        history = history[-10:]
        
        await self.redis.setex(key, self.ttl, json.dumps(history))
    
    async def clear(self, session_id: str):
        await self.redis.delete(f"session:{session_id}")
```

---

### 6. LLM Provider (Add Async and Retry)

**Current:** Synchronous OpenAI calls

**Target:** Azure OpenAI with async, retry, and circuit breaker

```python
# llm_provider_azure.py
from openai import AsyncAzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import circuitbreaker

class AzureOpenAIProvider:
    def __init__(self):
        self.client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2024-02-01",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    
    @circuitbreaker.circuit(failure_threshold=5, recovery_timeout=30)
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def generate(self, messages: list[dict]) -> str:
        response = await self.client.chat.completions.create(
            model=self.deployment,
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    
    async def generate_stream(self, messages: list[dict]):
        """Streaming for real-time responses."""
        async for chunk in await self.client.chat.completions.create(
            model=self.deployment,
            messages=messages,
            stream=True
        ):
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
```

**Azure OpenAI Provisioning:**

- Standard Pay-as-you-go is sufficient for this volume
- Estimate: 300k users x 5 msgs/day x 1000 tokens = 1.5B tokens/month
- Cost Strategy: Use gpt-4o-mini for 80% of queries to keep costs under $15k/month

---

## Implementation Phases

### Phase 1: API Foundation (Weeks 1-2)

- [ ] Create FastAPI wrapper around existing chatbot
- [ ] Add response caching with Redis
- [ ] Implement IP-based rate limiting
- [ ] Deploy to Azure Container Apps
- [ ] Set up Azure Front Door

### Phase 2: Vector Store Migration (Weeks 3-4)

- [ ] Set up Azure AI Search index
- [ ] Create one-time indexing script
- [ ] Migrate from ChromaDB to Azure AI Search
- [ ] Test retrieval quality

### Phase 3: Production Hardening (Weeks 5-6)

- [ ] Azure Front Door with WAF
- [ ] Auto-scaling configuration
- [ ] Health checks and monitoring
- [ ] Load testing

---

## Cost Estimation

### Monthly Cost Breakdown (300K Daily Active Users)

| Category | Service | Monthly Cost |
|----------|---------|--------------|
| **Compute** | Container Apps (5-15 replicas) | $800 |
| **AI/ML** | Azure OpenAI (GPT-4o-mini) | $10,000 |
| **Search** | Azure AI Search S1 | $250 |
| **Cache** | Redis Standard C2 | $200 |
| **Networking** | Front Door | $350 |
| **Monitoring** | App Insights | $200 |
| **Security** | Key Vault | $10 |
| | | |
| **Total** | | **~$11,800/month** |

**Cost Optimization (already applied):**

- No database (stateless design)
- No Service Bus (static documents)
- No user authentication infrastructure
- Aggressive response caching (60%+ cache hit)
- GPT-4o-mini as default model

---

## Performance Considerations

### Latency Targets

| Metric | Target | Strategy |
|--------|--------|----------|
| P50 Response Time | < 400ms | Redis cache |
| P99 Response Time | < 2s | Timeouts, circuit breakers |
| Cache Hit | 60%+ | Normalize questions, aggressive caching |

### Throughput Targets

| Metric | Target |
|--------|--------|
| Concurrent Users | 3,000 - 5,000 |
| Requests/Second | 200 - 400 |
| Messages/Day | 1,500,000 |

### Load Testing

```bash
# Example k6 load test
k6 run --vus 500 --duration 30m load_test.js
```

---

## Security & Compliance

### Privacy-First Design

| Aspect | Implementation |
|--------|----------------|
| **No PII** | No user accounts, no email, no tracking |
| **No persistence** | Conversations expire after 15 minutes |
| **No cookies** | Stateless API, session ID in request body |
| **GDPR compliant** | Nothing to delete (no data stored) |

### Security Checklist

- [ ] TLS 1.3 everywhere
- [ ] WAF rules for OWASP Top 10
- [ ] DDoS Protection (Front Door included)
- [ ] Rate limiting per IP
- [ ] Input validation and sanitization
- [ ] No secrets in code (Key Vault)
- [ ] Content filtering on LLM responses

---

## Monitoring & Observability

### Key Metrics

| Metric | Alert Threshold |
|--------|-----------------|
| Error Rate | > 1% for 5 min |
| P99 Latency | > 3s |
| Cache Hit Rate | < 40% |
| Rate Limit Hits | > 1000/hour |
| LLM Token Usage | > $500/day |

### Simple Dashboard

```
+----------------------------------------------------------+
|  RAG Chatbot - Production Dashboard                      |
+------------------+-------------------+-------------------+
|  Requests/min    |  Cache Hit Rate   |  Error Rate       |
|  ======== 234    |  ======== 64%     |  == 0.2%          |
+------------------+-------------------+-------------------+
|  Response Time (P95): 847ms                              |
|  Daily Token Cost: $312                                  |
|  Rate Limited IPs: 47                                    |
+----------------------------------------------------------+
```

### Distributed Tracing

```python
# Enable OpenTelemetry
from opentelemetry import trace
from azure.monitor.opentelemetry import configure_azure_monitor

configure_azure_monitor(
    connection_string=os.getenv("APPINSIGHTS_CONNECTION_STRING")
)

tracer = trace.get_tracer(__name__)

@tracer.start_as_current_span("chat_request")
async def handle_chat(request: ChatRequest):
    with tracer.start_span("retrieve_context"):
        context = await vector_store.search(request.message)
    
    with tracer.start_span("generate_response"):
        response = await llm.generate(context, request.message)
    
    return response
```

---

## Summary

For a **public, stateless chatbot** with a small document corpus:

| Aspect | Approach |
|--------|----------|
| **User data** | None stored |
| **Conversations** | Ephemeral (15 min TTL in Redis) |
| **Documents** | Pre-indexed at deployment |
| **Scaling** | Container Apps auto-scale |
| **Cost control** | Aggressive caching + GPT-4o-mini |

**Estimated timeline:** 6 weeks  
**Estimated monthly cost:** ~$11,800

The simplified architecture removes significant infrastructure while maintaining the ability to handle 300K daily users.

---

*Document Version: 2.0*  
*Last Updated: December 2025*
