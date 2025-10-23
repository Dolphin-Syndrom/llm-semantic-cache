import time
from typing import Dict, Any, Optional
from collections import defaultdict
import structlog
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, start_http_server

logger = structlog.get_logger(__name__)


class CacheMonitor:
    """
    Monitoring system for semantic cache
    
    Tracks:
    - Hit rate per agent
    - Cache latency
    - LLM calls saved
    - Storage usage
    - Query patterns
    """
    
    def __init__(
        self,
        prometheus_port: Optional[int] = None,
        registry: Optional[CollectorRegistry] = None,
    ):
        """
        Initialize cache monitor
        
        Args:
            prometheus_port: Port for Prometheus metrics endpoint
            registry: Custom Prometheus registry (for testing)
        """
        self.registry = registry or CollectorRegistry()
        self.prometheus_port = prometheus_port
        
        # Prometheus metrics
        self._init_metrics()
        
        # In-memory stats
        self.agent_stats = defaultdict(lambda: {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_latency": 0.0,
            "llm_latency": 0.0,
            "tokens_saved": 0,
        })
        
        # Start Prometheus server if port specified
        if prometheus_port:
            try:
                start_http_server(prometheus_port, registry=self.registry)
                logger.info("prometheus_server_started", port=prometheus_port)
            except Exception as e:
                logger.error("prometheus_server_failed", error=str(e))
        
        logger.info("cache_monitor_initialized")
    
    def _init_metrics(self) -> None:
        """Initialize Prometheus metrics"""
        
        # Cache operations counter
        self.cache_queries = Counter(
            "semantic_cache_queries_total",
            "Total number of cache queries",
            ["agent_id", "result"],
            registry=self.registry,
        )
        
        # Cache hit rate gauge
        self.cache_hit_rate = Gauge(
            "semantic_cache_hit_rate",
            "Cache hit rate by agent",
            ["agent_id"],
            registry=self.registry,
        )
        
        # Cache latency histogram
        self.cache_latency = Histogram(
            "semantic_cache_latency_seconds",
            "Cache lookup latency",
            ["agent_id"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            registry=self.registry,
        )
        
        # LLM latency histogram
        self.llm_latency = Histogram(
            "semantic_cache_llm_latency_seconds",
            "LLM call latency",
            ["agent_id"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            registry=self.registry,
        )
        
        # LLM calls saved counter
        self.llm_calls_saved = Counter(
            "semantic_cache_llm_calls_saved_total",
            "Number of LLM calls saved by cache",
            ["agent_id"],
            registry=self.registry,
        )
        
        # Tokens saved counter
        self.tokens_saved = Counter(
            "semantic_cache_tokens_saved_total",
            "Estimated tokens saved by cache",
            ["agent_id"],
            registry=self.registry,
        )
        
        # Storage gauge
        self.storage_entries = Gauge(
            "semantic_cache_storage_entries",
            "Number of entries in cache",
            ["agent_id"],
            registry=self.registry,
        )
        
        # Similarity scores histogram
        self.similarity_scores = Histogram(
            "semantic_cache_similarity_scores",
            "Distribution of similarity scores",
            ["agent_id"],
            buckets=[0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
            registry=self.registry,
        )
    
    def record_cache_query(
        self,
        agent_id: str,
        hit: bool,
        cache_latency_ms: float,
        similarity: float = 0.0,
        tokens_saved: int = 0,
    ) -> None:
        """
        Record cache query metrics
        
        Args:
            agent_id: Agent identifier
            hit: Whether cache hit occurred
            cache_latency_ms: Cache lookup latency in milliseconds
            similarity: Similarity score
            tokens_saved: Tokens saved if hit
        """
        # Update counters
        result = "hit" if hit else "miss"
        self.cache_queries.labels(agent_id=agent_id, result=result).inc()
        
        # Update latency
        self.cache_latency.labels(agent_id=agent_id).observe(cache_latency_ms / 1000.0)
        
        # Update similarity
        self.similarity_scores.labels(agent_id=agent_id).observe(similarity)
        
        # Update in-memory stats
        stats = self.agent_stats[agent_id]
        stats["total_queries"] += 1
        stats["total_latency"] += cache_latency_ms
        
        if hit:
            stats["cache_hits"] += 1
            stats["tokens_saved"] += tokens_saved
            
            # Update Prometheus metrics
            self.llm_calls_saved.labels(agent_id=agent_id).inc()
            self.tokens_saved.labels(agent_id=agent_id).inc(tokens_saved)
        else:
            stats["cache_misses"] += 1
        
        # Update hit rate
        hit_rate = stats["cache_hits"] / stats["total_queries"]
        self.cache_hit_rate.labels(agent_id=agent_id).set(hit_rate)
        
        logger.debug(
            "cache_query_recorded",
            agent_id=agent_id,
            hit=hit,
            similarity=similarity,
            hit_rate=f"{hit_rate:.2%}",
        )
    
    def record_llm_call(
        self,
        agent_id: str,
        latency_ms: float,
    ) -> None:
        """
        Record LLM call metrics
        
        Args:
            agent_id: Agent identifier
            latency_ms: LLM call latency in milliseconds
        """
        self.llm_latency.labels(agent_id=agent_id).observe(latency_ms / 1000.0)
        
        stats = self.agent_stats[agent_id]
        stats["llm_latency"] += latency_ms
        
        logger.debug(
            "llm_call_recorded",
            agent_id=agent_id,
            latency_ms=f"{latency_ms:.2f}",
        )
    
    def update_storage_metrics(
        self,
        agent_id: str,
        entry_count: int,
    ) -> None:
        """
        Update storage metrics
        
        Args:
            agent_id: Agent identifier
            entry_count: Number of cache entries
        """
        self.storage_entries.labels(agent_id=agent_id).set(entry_count)
        
        logger.debug(
            "storage_metrics_updated",
            agent_id=agent_id,
            entries=entry_count,
        )
    
    def get_agent_stats(self, agent_id: str) -> Dict[str, Any]:
        """
        Get statistics for specific agent
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Dictionary with agent metrics
        """
        stats = self.agent_stats[agent_id]
        
        if stats["total_queries"] == 0:
            return {
                "agent_id": agent_id,
                "total_queries": 0,
                "hit_rate": 0.0,
                "avg_cache_latency_ms": 0.0,
                "avg_llm_latency_ms": 0.0,
                "tokens_saved": 0,
            }
        
        hit_rate = stats["cache_hits"] / stats["total_queries"]
        avg_cache_latency = stats["total_latency"] / stats["total_queries"]
        
        llm_calls = stats["cache_misses"]
        avg_llm_latency = (
            stats["llm_latency"] / llm_calls
            if llm_calls > 0
            else 0.0
        )
        
        return {
            "agent_id": agent_id,
            "total_queries": stats["total_queries"],
            "cache_hits": stats["cache_hits"],
            "cache_misses": stats["cache_misses"],
            "hit_rate": hit_rate,
            "avg_cache_latency_ms": avg_cache_latency,
            "avg_llm_latency_ms": avg_llm_latency,
            "tokens_saved": stats["tokens_saved"],
            "cost_savings_pct": hit_rate * 100,  # Approximate
        }
    
    def get_global_stats(self) -> Dict[str, Any]:
        """
        Get global statistics across all agents
        
        Returns:
            Dictionary with global metrics
        """
        total_queries = 0
        total_hits = 0
        total_tokens_saved = 0
        total_cache_latency = 0.0
        total_llm_latency = 0.0
        total_llm_calls = 0
        
        for stats in self.agent_stats.values():
            total_queries += stats["total_queries"]
            total_hits += stats["cache_hits"]
            total_tokens_saved += stats["tokens_saved"]
            total_cache_latency += stats["total_latency"]
            total_llm_latency += stats["llm_latency"]
            total_llm_calls += stats["cache_misses"]
        
        global_hit_rate = total_hits / total_queries if total_queries > 0 else 0.0
        avg_cache_latency = (
            total_cache_latency / total_queries
            if total_queries > 0
            else 0.0
        )
        avg_llm_latency = (
            total_llm_latency / total_llm_calls
            if total_llm_calls > 0
            else 0.0
        )
        
        return {
            "total_queries": total_queries,
            "total_cache_hits": total_hits,
            "total_llm_calls": total_llm_calls,
            "global_hit_rate": global_hit_rate,
            "avg_cache_latency_ms": avg_cache_latency,
            "avg_llm_latency_ms": avg_llm_latency,
            "total_tokens_saved": total_tokens_saved,
            "llm_calls_saved": total_hits,
            "cost_savings_pct": global_hit_rate * 100,
            "active_agents": len(self.agent_stats),
        }
    
    def reset_stats(self, agent_id: Optional[str] = None) -> None:
        """
        Reset statistics
        
        Args:
            agent_id: Optional agent ID to reset specific agent
        """
        if agent_id:
            if agent_id in self.agent_stats:
                del self.agent_stats[agent_id]
                logger.info("agent_stats_reset", agent_id=agent_id)
        else:
            self.agent_stats.clear()
            logger.info("all_stats_reset")
    
    def print_dashboard(self) -> str:
        """
        Generate text dashboard of metrics
        
        Returns:
            Formatted dashboard string
        """
        global_stats = self.get_global_stats()
        
        dashboard = "\n" + "=" * 60 + "\n"
        dashboard += "  SEMANTIC CACHE DASHBOARD\n"
        dashboard += "=" * 60 + "\n\n"
        
        dashboard += f"Global Metrics:\n"
        dashboard += f"  Total Queries:     {global_stats['total_queries']:,}\n"
        dashboard += f"  Cache Hits:        {global_stats['total_cache_hits']:,}\n"
        dashboard += f"  LLM Calls:         {global_stats['total_llm_calls']:,}\n"
        dashboard += f"  Hit Rate:          {global_stats['global_hit_rate']:.1%}\n"
        dashboard += f"  Tokens Saved:      {global_stats['total_tokens_saved']:,}\n"
        dashboard += f"  Cost Savings:      ~{global_stats['cost_savings_pct']:.0f}%\n"
        dashboard += f"  Avg Cache Latency: {global_stats['avg_cache_latency_ms']:.1f}ms\n"
        dashboard += f"  Avg LLM Latency:   {global_stats['avg_llm_latency_ms']:.1f}ms\n"
        dashboard += f"  Active Agents:     {global_stats['active_agents']}\n"
        
        if self.agent_stats:
            dashboard += f"\n" + "-" * 60 + "\n"
            dashboard += f"Per-Agent Metrics:\n"
            dashboard += "-" * 60 + "\n"
            
            for agent_id in sorted(self.agent_stats.keys()):
                stats = self.get_agent_stats(agent_id)
                dashboard += f"\n{agent_id}:\n"
                dashboard += f"  Queries: {stats['total_queries']:,} | "
                dashboard += f"Hit Rate: {stats['hit_rate']:.1%} | "
                dashboard += f"Tokens Saved: {stats['tokens_saved']:,}\n"
        
        dashboard += "\n" + "=" * 60 + "\n"
        
        return dashboard
