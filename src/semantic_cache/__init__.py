"""
Semantic Caching System for AI Agents
Production-ready vector similarity caching using Redis Stack
"""

from .cache import SemanticCache, CacheResult, CacheStats
from .redis_client import RedisVectorClient
from .agent import SemanticAgent
from .monitoring import CacheMonitor

__version__ = "1.0.0"
__all__ = [
    "SemanticCache",
    "CacheResult",
    "CacheStats",
    "RedisVectorClient",
    "SemanticAgent",
    "CacheMonitor",
]
