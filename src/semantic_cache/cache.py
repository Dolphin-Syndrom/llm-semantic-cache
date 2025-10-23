import hashlib
import time
from typing import Optional, List
from dataclasses import dataclass
import structlog
import numpy as np
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field

from .redis_client import RedisVectorClient

logger = structlog.get_logger(__name__)


class CacheResult(BaseModel):
    """Result from cache query"""
    hit: bool = Field(description="Whether cache hit occurred")
    response: Optional[str] = Field(default=None, description="Cached response")
    similarity: float = Field(default=0.0, description="Similarity score (0-1)")
    original_query: Optional[str] = Field(default=None, description="Original cached query")
    cache_key: Optional[str] = Field(default=None, description="Cache key used")
    latency_ms: float = Field(default=0.0, description="Cache lookup latency in milliseconds")


class CacheStats(BaseModel):
    """Cache statistics"""
    total_entries: int = Field(description="Total number of cached entries")
    hit_rate: float = Field(description="Cache hit rate (0-1)")
    avg_latency_ms: float = Field(description="Average cache lookup latency")
    llm_calls_saved: int = Field(description="Number of LLM calls saved by cache")
    total_queries: int = Field(description="Total queries processed")
    storage_usage_mb: float = Field(description="Approximate storage usage in MB")


class SemanticCache:
    """
    Semantic caching system using vector similarity search
    
    Caches LLM responses based on semantic similarity of queries,
    reducing costs and latency for similar questions.
    """
    
    def __init__(
        self,
        redis_client: RedisVectorClient,
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.85,
        ttl_seconds: int = 86400,
        max_results: int = 3,
    ):
        """
        Initialize semantic cache
        
        Args:
            redis_client: Redis vector client instance
            embedding_model: Sentence transformer model name
            similarity_threshold: Minimum similarity for cache hit (0-1)
            ttl_seconds: Time to live for cache entries (default 24h)
            max_results: Maximum results to return from vector search
        """
        self.redis_client = redis_client
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.max_results = max_results
        
        # Load embedding model
        logger.info("loading_embedding_model", model=embedding_model)
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Statistics tracking
        self._stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_latency_ms": 0.0,
        }
        
        logger.info(
            "semantic_cache_initialized",
            threshold=similarity_threshold,
            ttl_seconds=ttl_seconds,
            embedding_dim=self.embedding_dim,
        )
    
    def _generate_cache_key(
        self,
        text: str,
        agent_id: str,
        session_id: str,
    ) -> str:
        """
        Generate unique cache key
        
        Args:
            text: Query text
            agent_id: Agent identifier
            session_id: Session identifier
            
        Returns:
            Unique cache key
        """
        # Create hash of text + agent_id + session_id
        content = f"{agent_id}:{session_id}:{text}"
        hash_digest = hashlib.sha256(content.encode()).hexdigest()
        return f"{agent_id}:{session_id}:{hash_digest[:16]}"
    
    def _encode_query(self, text: str) -> np.ndarray:
        """
        Generate embedding for query text
        
        Args:
            text: Query text
            
        Returns:
            Embedding vector
        """
        embedding = self.embedding_model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Normalize for cosine similarity
        )
        return embedding
    
    async def query(
        self,
        text: str,
        agent_id: str,
        session_id: str,
        max_results: Optional[int] = None,
    ) -> CacheResult:
        """
        Query cache for similar entries
        
        Args:
            text: Query text to search for
            agent_id: Agent identifier for namespacing
            session_id: Session identifier for namespacing
            max_results: Override default max results
            
        Returns:
            CacheResult with hit status and response if found
        """
        start_time = time.time()
        
        try:
            # Update stats
            self._stats["total_queries"] += 1
            
            # Generate embedding
            embedding = self._encode_query(text)
            
            # Search for similar vectors
            results = await self.redis_client.vector_search(
                query_vector=embedding,
                agent_id=agent_id,
                session_id=session_id,
                top_k=max_results or self.max_results,
            )
            
            # Check if best match exceeds threshold
            if results and results[0]["similarity"] >= self.similarity_threshold:
                best_match = results[0]
                latency_ms = (time.time() - start_time) * 1000
                
                # Update stats
                self._stats["cache_hits"] += 1
                self._stats["total_latency_ms"] += latency_ms
                
                logger.info(
                    "cache_hit",
                    similarity=best_match["similarity"],
                    threshold=self.similarity_threshold,
                    latency_ms=f"{latency_ms:.2f}",
                    agent_id=agent_id,
                )
                
                return CacheResult(
                    hit=True,
                    response=best_match["response"],
                    similarity=best_match["similarity"],
                    original_query=best_match["query"],
                    cache_key=best_match["key"],
                    latency_ms=latency_ms,
                )
            else:
                # Cache miss
                latency_ms = (time.time() - start_time) * 1000
                self._stats["cache_misses"] += 1
                self._stats["total_latency_ms"] += latency_ms
                
                best_similarity = results[0]["similarity"] if results else 0.0
                
                logger.info(
                    "cache_miss",
                    best_similarity=best_similarity,
                    threshold=self.similarity_threshold,
                    latency_ms=f"{latency_ms:.2f}",
                    agent_id=agent_id,
                )
                
                return CacheResult(
                    hit=False,
                    similarity=best_similarity,
                    latency_ms=latency_ms,
                )
                
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error("cache_query_failed", error=str(e), latency_ms=f"{latency_ms:.2f}")
            
            # Return miss on error to fallback to LLM
            return CacheResult(
                hit=False,
                latency_ms=latency_ms,
            )
    
    async def store(
        self,
        text: str,
        response: str,
        agent_id: str,
        session_id: str,
    ) -> str:
        """
        Store query-response pair in cache
        
        Args:
            text: Query text
            response: LLM response to cache
            agent_id: Agent identifier for namespacing
            session_id: Session identifier for namespacing
            
        Returns:
            Cache key where entry was stored
        """
        start_time = time.time()
        
        try:
            # Generate embedding
            embedding = self._encode_query(text)
            
            # Generate cache key
            cache_key = self._generate_cache_key(text, agent_id, session_id)
            
            # Store in Redis
            await self.redis_client.store_vector(
                key=cache_key,
                embedding=embedding,
                query=text,
                response=response,
                agent_id=agent_id,
                session_id=session_id,
                ttl_seconds=self.ttl_seconds,
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            logger.info(
                "cache_stored",
                cache_key=cache_key,
                agent_id=agent_id,
                session_id=session_id,
                latency_ms=f"{latency_ms:.2f}",
            )
            
            return cache_key
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error("cache_store_failed", error=str(e), latency_ms=f"{latency_ms:.2f}")
            raise
    
    async def stats(self, agent_id: Optional[str] = None) -> CacheStats:
        """
        Get cache statistics
        
        Args:
            agent_id: Optional filter by agent ID
            
        Returns:
            CacheStats with performance metrics
        """
        try:
            # Get Redis stats
            redis_stats = await self.redis_client.get_stats(agent_id=agent_id)
            
            # Calculate hit rate
            total_queries = self._stats["total_queries"]
            cache_hits = self._stats["cache_hits"]
            hit_rate = cache_hits / total_queries if total_queries > 0 else 0.0
            
            # Calculate average latency
            avg_latency = (
                self._stats["total_latency_ms"] / total_queries
                if total_queries > 0
                else 0.0
            )
            
            # Estimate storage (approximate)
            # Each entry: ~1KB (embedding 384*4 + text + metadata)
            storage_mb = redis_stats["total_entries"] * 1.0 / 1024
            
            return CacheStats(
                total_entries=redis_stats["total_entries"],
                hit_rate=hit_rate,
                avg_latency_ms=avg_latency,
                llm_calls_saved=cache_hits,
                total_queries=total_queries,
                storage_usage_mb=storage_mb,
            )
            
        except Exception as e:
            logger.error("stats_failed", error=str(e))
            
            # Return basic stats on error
            return CacheStats(
                total_entries=0,
                hit_rate=0.0,
                avg_latency_ms=0.0,
                llm_calls_saved=self._stats["cache_hits"],
                total_queries=self._stats["total_queries"],
                storage_usage_mb=0.0,
            )
    
    async def clear(self, agent_id: Optional[str] = None, session_id: Optional[str] = None) -> int:
        """
        Clear cache entries
        
        Args:
            agent_id: Optional filter by agent ID
            session_id: Optional filter by session ID
            
        Returns:
            Number of entries cleared
        """
        # TODO: Implement selective cache clearing
        logger.warning("cache_clear_not_implemented")
        return 0
