# Scaling RAG Chatbot to 10 Million Users on Azure

This document outlines the architectural changes, Azure services, and implementation steps required to scale this RAG chatbot from a single-user CLI application to a production system supporting 10 million active users.

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
| **Conversation Memory** | In-memory Python dict | Lost on restart, no persistence, ~100MB RAM limit |
| **Vector Store** | Local ChromaDB (SQLite) | Single-node, ~1M vectors max, no horizontal scaling |
| **LLM Calls** | Synchronous blocking | One request at a time, timeout issues |
| **Document Processing** | On-demand at startup | Slow cold starts, repeated processing |
| **Authentication** | None | No user isolation or access control |
| **Caching** | None | Repeated API calls, high latency & cost |
| **Error Handling** | Basic try/catch | No retry logic, circuit breakers |

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
│                              - Rate limiting (per user/tier)                    │
│                              - API versioning                                   │
│                              - Authentication (OAuth2/JWT)                      │
└─────────────────────────────────────────────────────────────────────────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
                    ▼                    ▼                    ▼
          ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
          │  Azure Container│  │  Azure Container│  │  Azure Container│
          │  Apps (FastAPI) │  │  Apps (FastAPI) │  │  Apps (FastAPI) │
          │  - Auto-scale   │  │  - Auto-scale   │  │  - Auto-scale   │
          │  - 0-1000 nodes │  │  - 0-1000 nodes │  │  - 0-1000 nodes │
          └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
                   │                    │                    │
                   └────────────────────┼────────────────────┘
                                        │
        ┌───────────────┬───────────────┼───────────────┬───────────────┐
        │               │               │               │               │
        ▼               ▼               ▼               ▼               ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│ Azure Cache   │ │ Azure Cosmos  │ │ Azure AI      │ │ Azure Service │ │ Azure OpenAI  │
│ for Redis     │ │ DB for        │ │ Search        │ │ Bus           │ │ Service       │
│               │ │ PostgreSQL    │ │ (Vector DB)   │ │ (Queue)       │ │               │
│ - Sessions    │ │ - History     │ │ - Embeddings  │ │ - Async jobs  │ │ - GPT-4       │
│ - Cache       │ │ - Users       │ │ - 1B+ vectors │ │ - Doc ingest  │ │ - Embeddings  │
└───────────────┘ └───────────────┘ └───────────────┘ └───────────────┘ └───────────────┘
                                        │
                                        ▼
                            ┌───────────────────────┐
                            │  Azure Blob Storage   │
                            │  - PDF documents      │
                            │  - Processed chunks   │
                            └───────────────────────┘
```

---

## Azure Services Required

### Core Services

| Service | Purpose | SKU Recommendation | Est. Monthly Cost |
|---------|---------|-------------------|-------------------|
| **Azure Container Apps** | API hosting with auto-scale | Consumption plan | $2,000 - $15,000 |
| **Azure OpenAI Service** | LLM & embeddings | PTU (Provisioned) | $20,000 - $100,000+ |
| **Azure AI Search** | Vector database | S2 or S3 | $2,500 - $10,000 |
| **Azure Cache for Redis** | Session & response cache | P2 (13GB) | $1,200 |
| **Azure Cosmos DB for PostgreSQL** | Conversation history | Citus cluster | $3,000 - $8,000 |
| **Azure Service Bus** | Async job queue | Premium | $700 |
| **Azure Blob Storage** | Document storage | Hot tier | $500 |

### Supporting Services

| Service | Purpose | Est. Monthly Cost |
|---------|---------|-------------------|
| **Azure Front Door** | Global CDN, WAF, DDoS | $1,500 |
| **Azure API Management** | Rate limiting, auth | $2,800 (Premium) |
| **Azure Key Vault** | Secrets management | $50 |
| **Azure Monitor + App Insights** | Observability | $1,000 |
| **Microsoft Entra ID** | Authentication | $600 (P1) |

---

## Component-by-Component Migration

### 1. Web API Layer (Replace CLI)

**Current:** `main.py` with CLI input loop

**Target:** FastAPI with async endpoints

```python
# api/main.py
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
import asyncio

app = FastAPI(title="RAG Chatbot API")

@app.post("/chat")
async def chat(
    request: ChatRequest,
    user: User = Depends(get_current_user)
):
    """Async chat endpoint with user context."""
    response = await chatbot.chat_async(
        question=request.message,
        session_id=f"{user.id}:{request.session_id}"
    )
    return ChatResponse(message=response)

@app.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    user: User = Depends(get_current_user)
):
    """Server-Sent Events for streaming responses."""
    return StreamingResponse(
        chatbot.chat_stream(request.message, user.id),
        media_type="text/event-stream"
    )
```

**Azure Deployment:**
- Azure Container Apps with min 3 replicas
- Scale rule: HTTP concurrent requests > 100
- Max replicas: 1000

---

### 2. Vector Store (Replace ChromaDB)

**Current:** Local ChromaDB with SQLite backend (~1M vector limit)

**Target:** Azure AI Search with vector search capability

```python
# vector_store_azure.py
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    VectorSearch,
    HnswAlgorithmConfiguration,
)

class AzureAISearchVectorStore:
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
        results = await self.client.search(
            search_text=None,
            vector_queries=[{
                "vector": query_embedding,
                "k_nearest_neighbors": k,
                "fields": "content_vector"
            }]
        )
        return [self._to_document(r) async for r in results]
```

**Scaling Benefits:**
- Handles billions of vectors
- Built-in replication and sharding
- 99.9% SLA
- Semantic ranking and hybrid search

---

### 3. Conversation Memory (Replace In-Memory Dict)

**Current:** Python dict in memory

**Target:** Azure Cosmos DB for PostgreSQL + Redis cache

```python
# memory/conversation_store.py
from redis import asyncio as aioredis
import asyncpg

class ConversationStore:
    def __init__(self):
        self.redis = aioredis.from_url(os.getenv("REDIS_URL"))
        self.pg_pool = None
    
    async def get_history(self, session_id: str) -> list[dict]:
        # Try cache first
        cached = await self.redis.get(f"history:{session_id}")
        if cached:
            return json.loads(cached)
        
        # Fall back to database
        async with self.pg_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT role, content, created_at
                FROM conversation_history
                WHERE session_id = $1
                ORDER BY created_at DESC
                LIMIT 20
            """, session_id)
        
        history = [dict(r) for r in rows]
        
        # Cache for 30 minutes
        await self.redis.setex(
            f"history:{session_id}",
            1800,
            json.dumps(history)
        )
        return history
    
    async def add_message(self, session_id: str, role: str, content: str):
        async with self.pg_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO conversation_history 
                (session_id, role, content)
                VALUES ($1, $2, $3)
            """, session_id, role, content)
        
        # Invalidate cache
        await self.redis.delete(f"history:{session_id}")
```

**Database Schema:**

```sql
-- Citus distributed table for horizontal scaling
CREATE TABLE conversation_history (
    id BIGSERIAL,
    session_id TEXT NOT NULL,
    user_id UUID NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    tokens_used INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (session_id, id)
);

-- Distribute by session_id for locality
SELECT create_distributed_table('conversation_history', 'session_id');

-- Index for user queries
CREATE INDEX idx_history_user ON conversation_history(user_id, created_at DESC);
```

---

### 4. LLM Provider (Add Async + Rate Limiting)

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
- Use PTU (Provisioned Throughput Units) for predictable performance
- Estimate: 10M users × 5 msgs/day × 500 tokens = 25B tokens/month
- Required: ~500-1000 PTUs for GPT-4

---

### 5. Document Ingestion Pipeline

**Current:** Load PDFs on startup

**Target:** Async pipeline with Azure Service Bus

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Upload    │───▶│  Service    │───▶│   Worker    │───▶│  AI Search  │
│   API       │    │   Bus       │    │  (Chunking) │    │  (Index)    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
      │                                      │
      ▼                                      ▼
┌─────────────┐                       ┌─────────────┐
│    Blob     │                       │   Azure     │
│   Storage   │                       │   OpenAI    │
│   (PDFs)    │                       │ (Embeddings)│
└─────────────┘                       └─────────────┘
```

```python
# workers/document_processor.py
from azure.servicebus.aio import ServiceBusClient
from azure.storage.blob.aio import BlobServiceClient

class DocumentProcessor:
    async def process_message(self, message):
        # Download PDF from Blob Storage
        blob_url = json.loads(str(message))["blob_url"]
        pdf_content = await self.download_blob(blob_url)
        
        # Extract and chunk text
        chunks = self.chunk_document(pdf_content)
        
        # Generate embeddings in batches
        embeddings = await self.batch_embed(chunks)
        
        # Index in Azure AI Search
        await self.index_documents(chunks, embeddings)
        
        # Mark complete
        await message.complete()
```

---

### 6. Caching Strategy

**Multi-level caching with Redis:**

```python
# cache/manager.py
class CacheManager:
    def __init__(self, redis: Redis):
        self.redis = redis
    
    async def get_or_compute_embedding(
        self, 
        text: str, 
        compute_fn
    ) -> list[float]:
        """Cache embeddings to reduce API calls."""
        cache_key = f"emb:{hashlib.md5(text.encode()).hexdigest()}"
        
        cached = await self.redis.get(cache_key)
        if cached:
            return json.loads(cached)
        
        embedding = await compute_fn(text)
        await self.redis.setex(cache_key, 86400, json.dumps(embedding))
        return embedding
    
    async def get_cached_response(
        self,
        question: str,
        context_hash: str
    ) -> str | None:
        """Cache frequent Q&A pairs."""
        cache_key = f"qa:{hashlib.md5(f'{question}:{context_hash}'.encode()).hexdigest()}"
        return await self.redis.get(cache_key)
```

**Cache Hit Targets:**
- Embeddings: 80%+ hit rate (same questions asked repeatedly)
- Responses: 30-40% hit rate (common questions)
- Session data: 95%+ hit rate

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
