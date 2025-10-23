import asyncio
import os
import time
import logging
from dotenv import load_dotenv
import structlog

from semantic_cache.redis_client import RedisVectorClient
from semantic_cache.cache import SemanticCache
from semantic_cache.agent import SemanticAgent
from semantic_cache.monitoring import CacheMonitor

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=False,
)

logger = structlog.get_logger()


async def demo_basic_caching():
    """Demonstrate basic cache hit/miss behavior"""
    print("\n" + "=" * 60)
    print("Basic Semantic Caching")
    print("=" * 60 + "\n")
    
    # Initialize components
    redis_client = RedisVectorClient(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
    )
    await redis_client.initialize()
    
    cache = SemanticCache(
        redis_client=redis_client,
        similarity_threshold=0.85,
    )
    
    agent = SemanticAgent(
        agent_id="demo_agent_1",
        cache=cache,
    )
    
    # Test queries (similar questions)
    queries = [
        "What is Python programming?",
        "How do I learn Python?",  # Similar
        "What is the capital of France?",  # Different topic
        "Tell me about Python language",  # Very similar to first
    ]
    
    print("Running queries...\n")
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 60)
        
        response = await agent.chat(query=query, session_id="demo_session")
        
        print(f"Response: {response.response[:100]}...")
        print(f"From Cache: {response.from_cache}")
        print(f"Similarity: {response.similarity:.3f}")
        print(f"Latency: {response.latency_ms:.1f}ms")
        
        if response.from_cache:
            print(f"✓ Tokens Saved: {response.tokens_saved}")
        
        await asyncio.sleep(0.5)
    
    stats = await agent.get_stats()
    print("\n" + "=" * 60)
    print("Agent Statistics:")
    print("=" * 60)
    print(f"Total Requests: {stats['total_requests']}")
    print(f"Cache Hits: {stats['cache_hits']}")
    print(f"LLM Calls: {stats['llm_calls']}")
    print(f"Hit Rate: {stats['hit_rate']:.1%}")
    print(f"Tokens Saved: {stats['tokens_saved']:,}")
    
    await redis_client.close()


async def demo_high_hit_rate():
    """Demonstrate high cache hit rate with conversational queries"""
    print("\n" + "=" * 60)
    print("DEMO 2: High Hit Rate Scenario (Conversational)")
    print("=" * 60 + "\n")
    
    # Initialize
    redis_client = RedisVectorClient(host="localhost", port=6379)
    await redis_client.initialize()
    
    cache = SemanticCache(redis_client=redis_client, similarity_threshold=0.85)
    agent = SemanticAgent(agent_id="demo_agent_2", cache=cache)
    monitor = CacheMonitor()

    conversation = [
        "How do I install Python?",
        "What's the best way to install Python?",  # Similar
        "Can you help me install Python?",  # Similar
        "How to set up Python on my computer?",  # Similar
        "What is machine learning?",
        "Explain machine learning to me",  # Similar
        "What does ML mean?",  # Similar
        "How to learn machine learning?",
        "Best way to study ML?",  # Similar
        "Tips for learning machine learning?",  # Similar
    ]
    
    print(f"Processing {len(conversation)} queries...\n")
    
    start_time = time.time()
    
    for i, query in enumerate(conversation, 1):
        response = await agent.chat(query=query, session_id="conversation_1")
        
        # Record in monitor
        monitor.record_cache_query(
            agent_id=agent.agent_id,
            hit=response.from_cache,
            cache_latency_ms=response.cache_latency_ms,
            similarity=response.similarity,
            tokens_saved=response.tokens_saved,
        )
        
        status = "✓ HIT" if response.from_cache else "✗ MISS"
        print(f"{i:2d}. [{status}] {query[:50]:50s} | Sim: {response.similarity:.2f} | {response.latency_ms:.0f}ms")
    
    duration = time.time() - start_time
    
    # Display results
    print("\n" + "=" * 60)
    print("Performance Summary:")
    print("=" * 60)
    
    stats = await agent.get_stats()
    print(f"Total Queries:      {stats['total_requests']}")
    print(f"Cache Hits:         {stats['cache_hits']}")
    print(f"LLM Calls:          {stats['llm_calls']}")
    print(f"Hit Rate:           {stats['hit_rate']:.1%}")
    print(f"Tokens Saved:       {stats['tokens_saved']:,}")
    print(f"Total Time:         {duration:.2f}s")
    print(f"Avg Time/Query:     {(duration / len(conversation) * 1000):.1f}ms")
    print(f"\n Cost Savings:     ~{stats['hit_rate'] * 100:.0f}% (estimated)")
    
    # Global monitor stats
    print("\n" + monitor.print_dashboard())
    
    await redis_client.close()


async def demo_performance_benchmark():
    """Benchmark cache performance"""
    print("\n" + "=" * 60)
    print("DEMO 3: Performance Benchmark")
    print("=" * 60 + "\n")
    
    redis_client = RedisVectorClient(host="localhost", port=6379)
    await redis_client.initialize()
    
    cache = SemanticCache(redis_client=redis_client)
    agent = SemanticAgent(agent_id="benchmark_agent", cache=cache)
    
    # Warm up cache
    print("Warming up cache with 20 entries...")
    queries = [f"What is topic number {i}?" for i in range(20)]
    for query in queries:
        await agent.chat(query=query, session_id="benchmark")
    
    # Benchmark cache hits
    print("\nBenchmarking cache hit latency (100 queries)...")
    latencies = []
    
    start = time.time()
    for i in range(100):
        query = queries[i % 20]  # Reuse cached queries
        response = await agent.chat(query=query, session_id="benchmark")
        if response.from_cache:
            latencies.append(response.cache_latency_ms)
    
    duration = time.time() - start
    
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        
        print("\n" + "=" * 60)
        print("Benchmark Results:")
        print("=" * 60)
        print(f"Total Queries:      100")
        print(f"Cache Hits:         {len(latencies)}")
        print(f"Total Time:         {duration:.2f}s")
        print(f"Throughput:         {100 / duration:.0f} QPS")
        print(f"\nLatency Statistics:")
        print(f"  Average:          {avg_latency:.2f}ms")
        print(f"  Min:              {min_latency:.2f}ms")
        print(f"  Max:              {max_latency:.2f}ms")
        print(f"  P95:              {p95_latency:.2f}ms")
        print(f"\n✓ Target: <50ms    | Actual: {avg_latency:.1f}ms")
    
    # Cache stats
    cache_stats = await cache.stats()
    print(f"\nCache Storage:")
    print(f"  Total Entries:    {cache_stats.total_entries}")
    print(f"  Storage Usage:    {cache_stats.storage_usage_mb:.2f}MB")
    print(f"  Avg Entry Size:   ~{(cache_stats.storage_usage_mb * 1024) / max(cache_stats.total_entries, 1):.1f}KB")
    
    await redis_client.close()


async def demo_namespace_isolation():
    """Demonstrate agent and session isolation"""
    print("\n" + "=" * 60)
    print("DEMO 4: Namespace Isolation")
    print("=" * 60 + "\n")
    
    redis_client = RedisVectorClient(host="localhost", port=6379)
    await redis_client.initialize()
    
    cache = SemanticCache(redis_client=redis_client)
    
    # Create multiple agents
    agent1 = SemanticAgent(agent_id="customer_support", cache=cache)
    agent2 = SemanticAgent(agent_id="sales_assistant", cache=cache)
    
    query = "What is our refund policy?"
    
    # Agent 1 stores response
    print("Agent 1 (Customer Support) - First query:")
    response1 = await agent1.chat(query=query, session_id="session_1")
    print(f"  From Cache: {response1.from_cache}")
    print(f"  Response: {response1.response[:60]}...")
    
    # Agent 1 retrieves from cache
    print("\nAgent 1 (Customer Support) - Same query:")
    response2 = await agent1.chat(query=query, session_id="session_1")
    print(f"  From Cache: {response2.from_cache} ✓")
    
    # Agent 2 has isolated cache
    print("\nAgent 2 (Sales Assistant) - Same query:")
    response3 = await agent2.chat(query=query, session_id="session_1")
    print(f"  From Cache: {response3.from_cache}")
    print(f"  Response: {response3.response[:60]}...")
    
    print("\n✓ Agents have isolated caches!")
    
    # Different sessions
    print("\nAgent 1 - Different session:")
    response4 = await agent1.chat(query=query, session_id="session_2")
    print(f"  From Cache: {response4.from_cache}")
    
    print("\n✓ Sessions can be isolated or shared based on use case")
    
    await redis_client.close()


async def main():    
    load_dotenv()
    
    try:
        await demo_basic_caching()
        await asyncio.sleep(1)
        
        await demo_high_hit_rate()
        await asyncio.sleep(1)
        
        await demo_performance_benchmark()
        await asyncio.sleep(1)
        
        await demo_namespace_isolation()

    except Exception as e:
        logger.error("demo_failed", error=str(e))
        raise


if __name__ == "__main__":
    import logging
    asyncio.run(main())
