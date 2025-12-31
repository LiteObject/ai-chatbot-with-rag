import hashlib
import json
import os
from redis import asyncio as aioredis


class ResponseCache:
    def __init__(self):
        self.redis = aioredis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379"), decode_responses=True
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
        self, question: str, answer: str, ttl: int = 3600  # 1 hour default
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
