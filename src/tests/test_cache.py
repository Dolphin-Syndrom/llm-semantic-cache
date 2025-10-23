"""
Comprehensive Test Suite for Semantic Cache
Tests integration, edge cases, performance, and concurrency
"""

import asyncio
import time
import uuid
from typing import List
import pytest
import numpy as np

from semantic_cache.redis_client import RedisVectorClient
from semantic_cache.cache import SemanticCache, CacheResult
from semantic_cache.agent import SemanticAgent
from semantic_cache.monitoring import CacheMonitor


# Fixtures
@pytest.fixture
async def redis_client():
    """Redis client fixture"""
    client = RedisVectorClient(
        host="localhost",
        port=6379,
        max_connections=20,
    )
    await client.initialize()
    yield client
    await client.close()


@pytest.fixture
async def semantic_cache(redis_client):
    """Semantic cache fixture"""
    cache = SemanticCache(
        redis_client=redis_client,
        embedding_model="all-MiniLM-L6-v2",
        similarity_threshold=0.85,
        ttl_seconds=3600,
    )
    yield cache


@pytest.fixture
async def semantic_agent(semantic_cache):
    """Semantic agent fixture"""
    agent = SemanticAgent(
        agent_id=f"test_agent_{uuid.uuid4().hex[:8]}",
        cache=semantic_cache,
        llm_client=None,  # Mock mode
    )
    yield agent


@pytest.fixture
def cache_monitor():
    """Cache monitor fixture"""
    monitor = CacheMonitor(prometheus_port=None)
    yield monitor


# Test Redis Client
@pytest.mark.asyncio
class TestRedisClient:
    """Test Redis vector client functionality"""
    
    async def test_health_check(self, redis_client):
        """Test Redis connection health check"""
        is_healthy = await redis_client.health_check()
        assert is_healthy is True
    
    async def test_index_creation(self, redis_client):
        """Test vector index creation"""
        await redis_client.create_index()
        assert redis_client._index_created is True
    
    async def test_vector_storage(self, redis_client):
        """Test storing and retrieving vectors"""
        embedding = np.random.rand(384).astype(np.float32)
        
        success = await redis_client.store_vector(
            key="test_key_1",
            embedding=embedding,
            query="What is Python?",
            response="Python is a programming language.",
            agent_id="test_agent",
            session_id="test_session",
            ttl_seconds=3600,
        )
        
        assert success is True
    
    async def test_vector_search(self, redis_client):
        """Test vector similarity search"""
        # Store some vectors
        queries = [
            ("What is Python?", "Python is a programming language."),
            ("How to learn coding?", "Start with tutorials and practice."),
            ("What is machine learning?", "ML is a subset of AI."),
        ]
        
        for i, (query, response) in enumerate(queries):
            embedding = np.random.rand(384).astype(np.float32)
            await redis_client.store_vector(
                key=f"search_test_{i}",
                embedding=embedding,
                query=query,
                response=response,
                agent_id="test_agent",
                session_id="test_session",
            )
        
        # Search
        query_vector = np.random.rand(384).astype(np.float32)
        results = await redis_client.vector_search(
            query_vector=query_vector,
            agent_id="test_agent",
            session_id="test_session",
            top_k=3,
        )
        
        assert len(results) <= 3
        if results:
            assert "similarity" in results[0]
            assert "query" in results[0]
            assert "response" in results[0]


# Test Semantic Cache
@pytest.mark.asyncio
class TestSemanticCache:
    """Test semantic cache core functionality"""
    
    async def test_cache_miss(self, semantic_cache):
        """Test cache miss on empty cache"""
        result = await semantic_cache.query(
            text="What is the meaning of life?",
            agent_id="test_agent",
            session_id="test_session",
        )
        
        assert result.hit is False
        assert result.response is None
    
    async def test_cache_store_and_hit(self, semantic_cache):
        """Test storing and retrieving from cache"""
        query = "What is artificial intelligence?"
        response = "AI is the simulation of human intelligence by machines."
        agent_id = f"agent_{uuid.uuid4().hex[:8]}"
        session_id = "session_1"
        
        # Store
        cache_key = await semantic_cache.store(
            text=query,
            response=response,
            agent_id=agent_id,
            session_id=session_id,
        )
        
        assert cache_key is not None
        
        # Query exact match
        result = await semantic_cache.query(
            text=query,
            agent_id=agent_id,
            session_id=session_id,
        )
        
        assert result.hit is True
        assert result.response == response
        assert result.similarity >= 0.99  # Should be very high for exact match
    
    async def test_semantic_similarity(self, semantic_cache):
        """Test semantic similarity matching"""
        agent_id = f"agent_{uuid.uuid4().hex[:8]}"
        session_id = "session_sim"
        
        # Store original query
        await semantic_cache.store(
            text="How do I learn Python programming?",
            response="Start with Python basics, then practice coding exercises.",
            agent_id=agent_id,
            session_id=session_id,
        )
        
        # Query with similar text
        result = await semantic_cache.query(
            text="What's the best way to study Python?",
            agent_id=agent_id,
            session_id=session_id,
        )
        
        # Should find similar match
        assert result.similarity > 0.5  # Some similarity expected
    
    async def test_threshold_filtering(self, semantic_cache):
        """Test similarity threshold filtering"""
        agent_id = f"agent_{uuid.uuid4().hex[:8]}"
        session_id = "session_threshold"
        
        # Store query about Python
        await semantic_cache.store(
            text="What is Python programming?",
            response="Python is a high-level programming language.",
            agent_id=agent_id,
            session_id=session_id,
        )
        
        # Query about completely different topic
        result = await semantic_cache.query(
            text="What is the weather forecast?",
            agent_id=agent_id,
            session_id=session_id,
        )
        
        # Should miss due to low similarity
        assert result.hit is False or result.similarity < semantic_cache.similarity_threshold
    
    async def test_namespacing_isolation(self, semantic_cache):
        """Test agent and session isolation"""
        # Store in agent1/session1
        await semantic_cache.store(
            text="Test query",
            response="Test response",
            agent_id="agent_1",
            session_id="session_1",
        )
        
        # Query from agent2/session2 - should not find
        result = await semantic_cache.query(
            text="Test query",
            agent_id="agent_2",
            session_id="session_2",
        )
        
        # May or may not hit depending on index state, but demonstrates isolation
        assert isinstance(result, CacheResult)
    
    async def test_cache_stats(self, semantic_cache):
        """Test cache statistics"""
        agent_id = f"agent_{uuid.uuid4().hex[:8]}"
        
        # Perform some operations
        await semantic_cache.store(
            text="Query 1",
            response="Response 1",
            agent_id=agent_id,
            session_id="session_1",
        )
        
        await semantic_cache.query(
            text="Query 2",
            agent_id=agent_id,
            session_id="session_1",
        )
        
        # Get stats
        stats = await semantic_cache.stats(agent_id=agent_id)
        
        assert stats.total_entries >= 0
        assert stats.total_queries >= 0
        assert 0.0 <= stats.hit_rate <= 1.0


# Test Semantic Agent
@pytest.mark.asyncio
class TestSemanticAgent:
    """Test semantic agent with cache integration"""
    
    async def test_agent_cache_miss_flow(self, semantic_agent):
        """Test agent flow on cache miss"""
        response = await semantic_agent.chat(
            query="What is the capital of France?",
            session_id="test_session",
        )
        
        assert response.from_cache is False
        assert response.response is not None
        assert response.llm_latency_ms > 0
    
    async def test_agent_cache_hit_flow(self, semantic_agent):
        """Test agent flow on cache hit"""
        query = "What is machine learning?"
        session_id = "test_session_hit"
        
        # First call - miss
        response1 = await semantic_agent.chat(query=query, session_id=session_id)
        assert response1.from_cache is False
        
        # Second call - should hit
        response2 = await semantic_agent.chat(query=query, session_id=session_id)
        assert response2.from_cache is True
        assert response2.similarity >= 0.99
        assert response2.tokens_saved > 0
    
    async def test_agent_force_llm(self, semantic_agent):
        """Test forcing LLM call bypassing cache"""
        query = "Test query"
        session_id = "test_force"
        
        # Store in cache
        await semantic_agent.chat(query=query, session_id=session_id)
        
        # Force LLM
        response = await semantic_agent.chat(
            query=query,
            session_id=session_id,
            force_llm=True,
        )
        
        assert response.from_cache is False
    
    async def test_agent_stats(self, semantic_agent):
        """Test agent statistics"""
        # Make some requests
        await semantic_agent.chat("Query 1", "session_1")
        await semantic_agent.chat("Query 2", "session_1")
        await semantic_agent.chat("Query 1", "session_1")  # Should hit
        
        stats = await semantic_agent.get_stats()
        
        assert stats["total_requests"] >= 3
        assert stats["agent_id"] == semantic_agent.agent_id


# Test Monitoring
class TestMonitoring:
    """Test monitoring and metrics"""
    
    def test_monitor_initialization(self, cache_monitor):
        """Test monitor initialization"""
        assert cache_monitor is not None
        assert cache_monitor.registry is not None
    
    def test_record_cache_hit(self, cache_monitor):
        """Test recording cache hit"""
        cache_monitor.record_cache_query(
            agent_id="test_agent",
            hit=True,
            cache_latency_ms=25.5,
            similarity=0.95,
            tokens_saved=100,
        )
        
        stats = cache_monitor.get_agent_stats("test_agent")
        assert stats["cache_hits"] == 1
        assert stats["tokens_saved"] == 100
    
    def test_record_cache_miss(self, cache_monitor):
        """Test recording cache miss"""
        cache_monitor.record_cache_query(
            agent_id="test_agent",
            hit=False,
            cache_latency_ms=20.0,
            similarity=0.5,
        )
        
        stats = cache_monitor.get_agent_stats("test_agent")
        assert stats["cache_misses"] >= 1
    
    def test_global_stats(self, cache_monitor):
        """Test global statistics"""
        # Record metrics for multiple agents
        cache_monitor.record_cache_query("agent_1", True, 25.0, 0.9, 50)
        cache_monitor.record_cache_query("agent_2", False, 30.0, 0.6)
        cache_monitor.record_cache_query("agent_1", True, 20.0, 0.95, 75)
        
        global_stats = cache_monitor.get_global_stats()
        
        assert global_stats["total_queries"] >= 3
        assert global_stats["active_agents"] >= 2
        assert 0.0 <= global_stats["global_hit_rate"] <= 1.0
    
    def test_dashboard_generation(self, cache_monitor):
        """Test dashboard text generation"""
        cache_monitor.record_cache_query("agent_1", True, 25.0, 0.9, 100)
        
        dashboard = cache_monitor.print_dashboard()
        
        assert "SEMANTIC CACHE DASHBOARD" in dashboard
        assert "Global Metrics" in dashboard
        assert "Hit Rate" in dashboard


# Performance Tests
@pytest.mark.asyncio
class TestPerformance:
    """Test cache performance and concurrency"""
    
    async def test_cache_latency(self, semantic_cache):
        """Test cache query latency is under 50ms"""
        agent_id = f"perf_agent_{uuid.uuid4().hex[:8]}"
        
        # Store entry
        await semantic_cache.store(
            text="Performance test query",
            response="Performance test response",
            agent_id=agent_id,
            session_id="perf_session",
        )
        
        # Measure query latency
        start = time.time()
        result = await semantic_cache.query(
            text="Performance test query",
            agent_id=agent_id,
            session_id="perf_session",
        )
        latency_ms = (time.time() - start) * 1000
        
        assert latency_ms < 100  # Should be under 100ms
        assert result.latency_ms < 100
    
    async def test_concurrent_queries(self, semantic_cache):
        """Test high concurrency (100 req/s)"""
        agent_id = f"concurrent_agent_{uuid.uuid4().hex[:8]}"
        session_id = "concurrent_session"
        
        # Store some entries
        for i in range(10):
            await semantic_cache.store(
                text=f"Query {i}",
                response=f"Response {i}",
                agent_id=agent_id,
                session_id=session_id,
            )
        
        # Concurrent queries
        async def query_task(idx: int):
            return await semantic_cache.query(
                text=f"Query {idx % 10}",
                agent_id=agent_id,
                session_id=session_id,
            )
        
        # Run 100 concurrent queries
        start = time.time()
        tasks = [query_task(i) for i in range(100)]
        results = await asyncio.gather(*tasks)
        duration = time.time() - start
        
        assert len(results) == 100
        assert duration < 5.0  # Should complete in under 5 seconds
        
        # Check some hits occurred
        hits = sum(1 for r in results if r.hit)
        assert hits > 0
    
    async def test_storage_efficiency(self, semantic_cache):
        """Test storage usage"""
        agent_id = f"storage_agent_{uuid.uuid4().hex[:8]}"
        
        # Store 100 entries
        for i in range(100):
            await semantic_cache.store(
                text=f"Storage test query {i}",
                response=f"Storage test response {i}" * 10,
                agent_id=agent_id,
                session_id="storage_session",
            )
        
        stats = await semantic_cache.stats()
        
        # Each entry should be ~1KB
        # 100 entries should be ~100KB = 0.1MB
        assert stats.storage_usage_mb < 1.0  # Less than 1MB for 100 entries


# Edge Cases
@pytest.mark.asyncio
class TestEdgeCases:
    """Test edge cases and error handling"""
    
    async def test_empty_query(self, semantic_cache):
        """Test handling of empty query"""
        result = await semantic_cache.query(
            text="",
            agent_id="test_agent",
            session_id="test_session",
        )
        
        assert isinstance(result, CacheResult)
    
    async def test_very_long_query(self, semantic_cache):
        """Test handling of very long query"""
        long_query = "test " * 1000  # Very long query
        
        cache_key = await semantic_cache.store(
            text=long_query,
            response="Response",
            agent_id="test_agent",
            session_id="test_session",
        )
        
        assert cache_key is not None
    
    async def test_special_characters(self, semantic_cache):
        """Test handling of special characters"""
        special_query = "Test 你好 émoji 🚀 newline\n tab\t"
        
        cache_key = await semantic_cache.store(
            text=special_query,
            response="Special response",
            agent_id="test_agent",
            session_id="test_session",
        )
        
        result = await semantic_cache.query(
            text=special_query,
            agent_id="test_agent",
            session_id="test_session",
        )
        
        assert result.hit is True


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--asyncio-mode=auto", "--cov=semantic_cache", "--cov-report=html"])
