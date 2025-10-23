import time
import uuid
from typing import Optional, Dict, Any
import structlog
from groq import AsyncGroq
from pydantic import BaseModel, Field

from .cache import SemanticCache, CacheResult

logger = structlog.get_logger(__name__)


class AgentResponse(BaseModel):
    """Response from agent with metadata"""
    response: str = Field(description="Agent response text")
    from_cache: bool = Field(description="Whether response came from cache")
    similarity: float = Field(default=0.0, description="Cache similarity score")
    latency_ms: float = Field(description="Total response latency")
    cache_latency_ms: float = Field(default=0.0, description="Cache lookup latency")
    llm_latency_ms: float = Field(default=0.0, description="LLM call latency")
    cache_key: Optional[str] = Field(default=None, description="Cache key used")
    tokens_saved: int = Field(default=0, description="Estimated tokens saved")


class SemanticAgent:    
    def __init__(
        self,
        agent_id: str,
        cache: SemanticCache,
        llm_client: Optional[AsyncGroq] = None,
        model: str = "llama-3.1-8b-instant",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ):
        """
        Initialize semantic agent
        
        Args:
            agent_id: Unique agent identifier
            cache: SemanticCache instance
            llm_client: Groq async client (optional for testing)
            model: LLM model to use (e.g., llama-3.1-8b-instant, llama-3.1-70b-versatile)
            system_prompt: System prompt for agent
            temperature: LLM temperature
            max_tokens: Maximum tokens in response
        """
        self.agent_id = agent_id
        self.cache = cache
        self.llm_client = llm_client
        self.model = model
        self.system_prompt = system_prompt or "You are a helpful AI assistant."
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Stats
        self._stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "llm_calls": 0,
            "total_tokens_saved": 0,
        }
        
        logger.info(
            "semantic_agent_initialized",
            agent_id=agent_id,
            model=model,
            cache_enabled=True,
        )
    
    async def chat(
        self,
        query: str,
        session_id: Optional[str] = None,
        force_llm: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentResponse:
        """
        Process user query with cache-first approach
        
        Flow:
        1. Check cache for similar query
        2. If hit and similarity > threshold, return cached response
        3. If miss, call LLM
        4. Store LLM response in cache
        5. Return response with metadata
        
        Args:
            query: User query text
            session_id: Session identifier (default: random UUID)
            force_llm: Skip cache and force LLM call
            metadata: Additional metadata for logging
            
        Returns:
            AgentResponse with response text and performance metrics
        """
        start_time = time.time()
        session_id = session_id or str(uuid.uuid4())
        
        self._stats["total_requests"] += 1
        
        logger.info(
            "agent_request",
            agent_id=self.agent_id,
            session_id=session_id,
            query_length=len(query),
            force_llm=force_llm,
        )
        
        cache_result: Optional[CacheResult] = None
        response_text = ""
        llm_latency_ms = 0.0
        tokens_saved = 0
        
        # Step 1: Cache lookup (unless forced)
        if not force_llm:
            try:
                cache_result = await self.cache.query(
                    text=query,
                    agent_id=self.agent_id,
                    session_id=session_id,
                )
                
                # Cache hit!
                if cache_result.hit:
                    self._stats["cache_hits"] += 1
                    
                    # Estimate tokens saved (rough estimate: 1 token ~= 4 chars)
                    tokens_saved = len(cache_result.response or "") // 4
                    self._stats["total_tokens_saved"] += tokens_saved
                    
                    total_latency_ms = (time.time() - start_time) * 1000
                    
                    logger.info(
                        "agent_cache_hit",
                        agent_id=self.agent_id,
                        similarity=cache_result.similarity,
                        tokens_saved=tokens_saved,
                        latency_ms=f"{total_latency_ms:.2f}",
                    )
                    
                    return AgentResponse(
                        response=cache_result.response or "",
                        from_cache=True,
                        similarity=cache_result.similarity,
                        latency_ms=total_latency_ms,
                        cache_latency_ms=cache_result.latency_ms,
                        cache_key=cache_result.cache_key,
                        tokens_saved=tokens_saved,
                    )
                    
            except Exception as e:
                logger.error("cache_lookup_error", error=str(e))
                # Continue to LLM on cache error
        
        # Step 2: Cache miss - call LLM
        try:
            llm_start = time.time()
            
            if self.llm_client:
                # Real LLM call
                response = await self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": query},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                
                response_text = response.choices[0].message.content or ""
                llm_latency_ms = (time.time() - llm_start) * 1000
                
            else:
                # Mock response for testing
                response_text = f"Mock response for: {query[:50]}"
                llm_latency_ms = 100.0
            
            self._stats["llm_calls"] += 1
            
            logger.info(
                "agent_llm_call",
                agent_id=self.agent_id,
                llm_latency_ms=f"{llm_latency_ms:.2f}",
                response_length=len(response_text),
            )
            
        except Exception as e:
            logger.error("llm_call_failed", error=str(e))
            raise
        
        # Step 3: Store in cache
        cache_key = None
        try:
            cache_key = await self.cache.store(
                text=query,
                response=response_text,
                agent_id=self.agent_id,
                session_id=session_id,
            )
        except Exception as e:
            logger.error("cache_store_error", error=str(e))
            # Continue even if cache store fails
        
        # Return response
        total_latency_ms = (time.time() - start_time) * 1000
        
        return AgentResponse(
            response=response_text,
            from_cache=False,
            similarity=cache_result.similarity if cache_result else 0.0,
            latency_ms=total_latency_ms,
            cache_latency_ms=cache_result.latency_ms if cache_result else 0.0,
            llm_latency_ms=llm_latency_ms,
            cache_key=cache_key,
            tokens_saved=0,
        )
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get agent statistics
        
        Returns:
            Dictionary with agent performance metrics
        """
        try:
            cache_stats = await self.cache.stats(agent_id=self.agent_id)
            
            hit_rate = (
                self._stats["cache_hits"] / self._stats["total_requests"]
                if self._stats["total_requests"] > 0
                else 0.0
            )
            
            return {
                "agent_id": self.agent_id,
                "total_requests": self._stats["total_requests"],
                "cache_hits": self._stats["cache_hits"],
                "llm_calls": self._stats["llm_calls"],
                "hit_rate": hit_rate,
                "tokens_saved": self._stats["total_tokens_saved"],
                "cache_stats": cache_stats.model_dump(),
            }
            
        except Exception as e:
            logger.error("get_stats_failed", error=str(e))
            return {
                "agent_id": self.agent_id,
                "error": str(e),
            }
    
    async def clear_cache(self, session_id: Optional[str] = None) -> int:
        """
        Clear agent's cache
        
        Args:
            session_id: Optional session ID to clear specific session
            
        Returns:
            Number of entries cleared
        """
        try:
            count = await self.cache.clear(
                agent_id=self.agent_id,
                session_id=session_id,
            )
            
            logger.info(
                "agent_cache_cleared",
                agent_id=self.agent_id,
                session_id=session_id,
                count=count,
            )
            
            return count
            
        except Exception as e:
            logger.error("cache_clear_failed", error=str(e))
            return 0
