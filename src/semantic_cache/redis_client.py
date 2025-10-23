import asyncio
import time
from typing import Optional, Dict, List, Any
import structlog
import redis
import numpy as np

from redis.commands.search.field import VectorField, TextField, NumericField, TagField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

logger = structlog.get_logger(__name__)

class RedisVectorClient:
    
    INDEX_NAME = "semantic_cache_idx"
    PREFIX = "cache:"
    VECTOR_DIM = 384  # all-MiniLM-L6-v2 dimension
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        max_connections: int = 20,
        socket_timeout: int = 5,
        socket_connect_timeout: int = 5,
        retry_on_timeout: bool = True,
    ):
        """
        Initialize Redis client with connection pooling
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            max_connections: Maximum connections in pool
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Socket connect timeout in seconds
            retry_on_timeout: Retry on timeout
        """
        self.host = host
        self.port = port
        self.db = db
        
        self.pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            max_connections=max_connections,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            retry_on_timeout=retry_on_timeout,
            decode_responses=False, 
        )
        
        self.client = redis.Redis(connection_pool=self.pool)
        self._index_created = False
        
        logger.info(
            "redis_client_initialized",
            host=host,
            port=port,
            max_connections=max_connections,
        )
    
    async def initialize(self) -> None:
        """Initialize client and create index if needed"""
     
        await self.health_check()
        
     
        await self.create_index()
        
        logger.info("redis_client_ready", index=self.INDEX_NAME)
    
    async def health_check(self, max_retries: int = 5) -> bool:
        """
        Check Redis connection health with retries
        
        Args:
            max_retries: Maximum number of retry attempts
            
        Returns:
            True if healthy
            
        Raises:
            ConnectionError: If connection fails after retries
        """
        for attempt in range(max_retries):
            try:
                # Run in executor to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, self.client.ping)
                
                if result:
                    logger.info("redis_health_check_success", attempt=attempt + 1)
                    return True
                    
            except (redis.ConnectionError, redis.TimeoutError) as e:
                logger.warning(
                    "redis_health_check_failed",
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    error=str(e),
                )
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt) 
                else:
                    raise ConnectionError(f"Redis health check failed after {max_retries} attempts") from e
        
        return False
    
    async def create_index(self) -> None:
        """Create HNSW vector search index with all required fields"""
        if self._index_created:
            return
        
        try:
            loop = asyncio.get_event_loop()
            
            try:
                await loop.run_in_executor(None, self.client.ft(self.INDEX_NAME).info)
                logger.info("redis_index_exists", index=self.INDEX_NAME)
                self._index_created = True
                return
            except redis.ResponseError:
                pass
            
            schema = (
                VectorField(
                    "embedding",
                    "HNSW",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.VECTOR_DIM,
                        "DISTANCE_METRIC": "COSINE",
                        "INITIAL_CAP": 10000,
                        "M": 16, 
                        "EF_CONSTRUCTION": 200,  
                    },
                ),
                TextField("query", weight=1.0),
                TextField("response", weight=1.0),
                TagField("agent_id"),
                TagField("session_id"),
                NumericField("timestamp", sortable=True),
                NumericField("ttl"),
            )
            
            # Create index
            definition = IndexDefinition(
                prefix=[self.PREFIX],
                index_type=IndexType.HASH,
            )
            
            await loop.run_in_executor(
                None,
                lambda: self.client.ft(self.INDEX_NAME).create_index(
                    fields=schema,
                    definition=definition,
                ),
            )
            
            self._index_created = True
            logger.info(
                "redis_index_created",
                index=self.INDEX_NAME,
                prefix=self.PREFIX,
                vector_dim=self.VECTOR_DIM,
            )
            
        except redis.ResponseError as e:
            if "Index already exists" in str(e):
                self._index_created = True
                logger.info("redis_index_already_exists", index=self.INDEX_NAME)
            else:
                logger.error("redis_index_creation_failed", error=str(e))
                raise
    
    async def vector_search(
        self,
        query_vector: np.ndarray,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Perform KNN vector search
        
        Args:
            query_vector: Query embedding vector (384-dim)
            agent_id: Filter by agent ID
            session_id: Filter by session ID
            top_k: Number of results to return
            
        Returns:
            List of search results with similarity scores
        """
        try:
            
            vector_bytes = query_vector.astype(np.float32).tobytes()
            
            filters = []
            if agent_id:
                filters.append(f"@agent_id:{{{agent_id}}}")
            if session_id:
                filters.append(f"@session_id:{{{session_id}}}")
            
            filter_str = " ".join(filters) if filters else "*"
            
            query = (
                Query(f"({filter_str})=>[KNN {top_k} @embedding $vector AS score]")
                .sort_by("score")
                .return_fields("query", "response", "agent_id", "session_id", "timestamp", "score")
                .dialect(2)
            )

            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self.client.ft(self.INDEX_NAME).search(
                    query,
                    query_params={"vector": vector_bytes},
                ),
            )
            
            # Parse results
            parsed_results = []
            for doc in results.docs:
                parsed_results.append({
                    "key": doc.id,
                    "query": doc.query.decode() if isinstance(doc.query, bytes) else doc.query,
                    "response": doc.response.decode() if isinstance(doc.response, bytes) else doc.response,
                    "agent_id": doc.agent_id.decode() if isinstance(doc.agent_id, bytes) else doc.agent_id,
                    "session_id": doc.session_id.decode() if isinstance(doc.session_id, bytes) else doc.session_id,
                    "timestamp": float(doc.timestamp),
                    "similarity": 1 - float(doc.score),  # Convert distance to similarity
                })
            
            logger.debug(
                "vector_search_complete",
                results_count=len(parsed_results),
                agent_id=agent_id,
                session_id=session_id,
            )
            
            return parsed_results
            
        except Exception as e:
            logger.error("vector_search_failed", error=str(e))
            raise
    
    async def store_vector(
        self,
        key: str,
        embedding: np.ndarray,
        query: str,
        response: str,
        agent_id: str,
        session_id: str,
        ttl_seconds: int = 86400,
    ) -> bool:
        """
        Store embedding with metadata
        
        Args:
            key: Cache key
            embedding: Query embedding vector
            query: Original query text
            response: LLM response
            agent_id: Agent identifier
            session_id: Session identifier
            ttl_seconds: Time to live in seconds (default 24h)
            
        Returns:
            True if successful
        """
        try:
            # Convert vector to bytes
            vector_bytes = embedding.astype(np.float32).tobytes()
            
            # Prepare data
            data = {
                "embedding": vector_bytes,
                "query": query,
                "response": response,
                "agent_id": agent_id,
                "session_id": session_id,
                "timestamp": time.time(),
                "ttl": ttl_seconds,
            }
            
            # Store with TTL
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.client.hset(f"{self.PREFIX}{key}", mapping=data),
            )
            
            if ttl_seconds > 0:
                await loop.run_in_executor(
                    None,
                    lambda: self.client.expire(f"{self.PREFIX}{key}", ttl_seconds),
                )
            
            logger.debug(
                "vector_stored",
                key=key,
                agent_id=agent_id,
                session_id=session_id,
                ttl=ttl_seconds,
            )
            
            return True
            
        except Exception as e:
            logger.error("vector_store_failed", key=key, error=str(e))
            raise
    
    async def get_stats(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Args:
            agent_id: Filter by agent ID
            
        Returns:
            Dictionary with cache statistics
        """
        try:
            loop = asyncio.get_event_loop()
            
            index_info = await loop.run_in_executor(
                None,
                lambda: self.client.ft(self.INDEX_NAME).info(),
            )
            
            info_dict = {}
            for i in range(0, len(index_info), 2):
                key = index_info[i].decode() if isinstance(index_info[i], bytes) else index_info[i]
                value = index_info[i + 1]
                if isinstance(value, bytes):
                    value = value.decode()
                info_dict[key] = value
            
            total_docs = int(info_dict.get("num_docs", 0))
            
            # If agent_id filter, count matching keys
            if agent_id:
                filtered_count = total_docs  
            else:
                filtered_count = total_docs
            
            return {
                "total_entries": total_docs,
                "filtered_entries": filtered_count,
                "index_name": self.INDEX_NAME,
                "vector_dim": self.VECTOR_DIM,
            }
            
        except Exception as e:
            logger.error("get_stats_failed", error=str(e))
            return {
                "total_entries": 0,
                "filtered_entries": 0,
                "error": str(e),
            }
    
    async def close(self) -> None:
        """Close Redis connection pool"""
        try:
            self.pool.disconnect()
            logger.info("redis_connection_closed")
        except Exception as e:
            logger.error("redis_close_failed", error=str(e))
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            if hasattr(self, 'pool'):
                self.pool.disconnect()
        except:
            pass
