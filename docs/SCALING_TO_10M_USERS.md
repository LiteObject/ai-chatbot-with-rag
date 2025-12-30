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
└─────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Azure API Management                               │
│                              - Rate limiting (per IP/anonymous)                 │
│                              - API versioning                                   │
│                              - Abuse protection                                 │
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

**Services NOT needed (simplified architecture):**

- Azure Cosmos DB - No user data to store
- Azure Service Bus - Static documents, no ingestion pipeline
- Azure Blob Storage - Documents pre-indexed at deploy time
- Microsoft Entra ID - No user authentication

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
| **Azure Front Door** | CDN, WAF, DDoS | $350 |
| **Azure API Management** | Rate limiting | $150 (Basic) |
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

### Phase 1: API Foundation (Weeks 1-4)
- [ ] Create FastAPI wrapper around existing chatbot
- [ ] Add authentication with Microsoft Entra ID
- [ ] Deploy to Azure Container Apps
- [ ] Set up Azure API Management

### Phase 2: Data Layer Migration (Weeks 5-8)
- [ ] Migrate vector store to Azure AI Search
- [ ] Set up Azure Cosmos DB for PostgreSQL
- [ ] Implement Redis caching layer
- [ ] Create conversation persistence

### Phase 3: Async & Scale (Weeks 9-12)
- [ ] Convert to async/await patterns
- [ ] Implement Azure Service Bus for document ingestion
- [ ] Add circuit breakers and retry logic
- [ ] Set up auto-scaling rules

### Phase 4: Production Readiness (Weeks 13-16)
- [ ] Azure Front Door with WAF
- [ ] Comprehensive monitoring (App Insights)
- [ ] Load testing (10M user simulation)
- [ ] Disaster recovery setup
- [ ] Security audit

---

## Cost Estimation

### Monthly Cost Breakdown (10M Active Users)

| Category | Service | Monthly Cost |
|----------|---------|--------------|
| **Compute** | Container Apps (avg 50 replicas) | $8,000 |
| **AI/ML** | Azure OpenAI (500 PTUs) | $50,000 |
| **Database** | Cosmos DB PostgreSQL (16 nodes) | $6,000 |
| **Search** | Azure AI Search S3 | $8,000 |
| **Cache** | Redis P2 | $1,200 |
| **Networking** | Front Door + bandwidth | $3,000 |
| **Storage** | Blob (10TB) + ops | $500 |
| **Messaging** | Service Bus Premium | $700 |
| **Monitoring** | App Insights + Log Analytics | $1,500 |
| **Security** | Entra ID P1 + Key Vault | $700 |
| **API Gateway** | API Management Premium | $2,800 |
| | | |
| **Total** | | **~$82,400/month** |

**Cost Optimization Tips:**
- Use Reserved Capacity (1-3 year) for 30-50% savings
- Implement response caching to reduce LLM calls
- Use GPT-4o-mini for simple queries (10x cheaper)
- Auto-scale to zero during off-peak hours

---

## Performance Considerations

### Latency Targets

| Metric | Target | Strategy |
|--------|--------|----------|
| P50 Response Time | < 500ms | Redis cache, regional deployment |
| P99 Response Time | < 3s | Circuit breakers, timeouts |
| Embedding Lookup | < 50ms | Azure AI Search with replicas |
| Cold Start | < 2s | Container Apps min replicas |

### Throughput Targets

| Metric | Target |
|--------|--------|
| Concurrent Users | 100,000 |
| Requests/Second | 10,000 |
| Messages/Day | 50,000,000 |

### Load Testing

```bash
# Example k6 load test
k6 run --vus 1000 --duration 30m load_test.js
```

---

## Security & Compliance

### Authentication Flow

```
User ──▶ Entra ID ──▶ JWT Token ──▶ API Management ──▶ Container Apps
                          │
                          ▼
                    Token Validation
                    + Rate Limiting
                    + Scope Check
```

### Security Checklist

- [ ] TLS 1.3 everywhere
- [ ] Azure Private Link for databases
- [ ] Key Vault for all secrets
- [ ] Managed Identity (no API keys in code)
- [ ] Network Security Groups
- [ ] DDoS Protection Standard
- [ ] WAF rules for OWASP Top 10
- [ ] Data encryption at rest (AES-256)
- [ ] Audit logging to Log Analytics
- [ ] GDPR compliance (data residency, right to delete)

---

## Monitoring & Observability

### Key Metrics Dashboard

```
┌──────────────────────────────────────────────────────────────┐
│  RAG Chatbot - Production Dashboard                          │
├──────────────────┬──────────────────┬────────────────────────┤
│  Active Users    │  Requests/min    │  Error Rate            │
│  ████████ 47.2K  │  ████████ 8,234  │  ██ 0.3%              │
├──────────────────┴──────────────────┴────────────────────────┤
│  Response Time (P95)                                         │
│  ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁▂▃▄▅▆▇█▇▆▅▄▃▂▁  avg: 847ms               │
├──────────────────────────────────────────────────────────────┤
│  Token Usage (24h)                                           │
│  Input: 12.3M tokens    Output: 8.7M tokens    Cost: $1,247  │
├──────────────────────────────────────────────────────────────┤
│  Cache Hit Rate                                              │
│  Embeddings: 82%    Responses: 34%    Sessions: 97%         │
└──────────────────────────────────────────────────────────────┘
```

### Alerts Configuration

| Alert | Condition | Action |
|-------|-----------|--------|
| High Error Rate | > 1% for 5 min | Page on-call |
| Latency Spike | P99 > 5s | Scale up + notify |
| LLM Quota Warning | > 80% usage | Notify team |
| Database CPU | > 80% for 10 min | Auto-scale |
| Cache Miss Rate | > 50% | Investigate |

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

Scaling from a single-user CLI to 10M users requires:

1. **Replacing local components** with managed Azure services
2. **Adding async patterns** throughout the codebase
3. **Implementing multi-layer caching** for cost and latency
4. **Building proper observability** for production operations
5. **Investing in security** and compliance from day one

The modular architecture already in place (swappable providers) provides a solid foundation. The estimated timeline is **16 weeks** with an estimated monthly infrastructure cost of **~$82,000**.

---

*Document Version: 1.0*  
*Last Updated: December 2025*
