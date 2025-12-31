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
                status_code=429, detail="Rate limit exceeded. Please wait a minute."
            )
