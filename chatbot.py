# =============================================================================
# TODO(v2): RETIRED IN v2. Do not add features here.
# This class does everything in one place (models, retrieval, caching, logging,
# health checks), which is why it was hard to test. In v2 its job is split:
#   codesearch/pipeline.py         the query path, step by step
#   codesearch/factory.py          picks Groq / Ollama / FakeLLM and the embedder
#   codesearch/generation/llm.py   LLM clients (generate + stream)
#   codesearch/retrieval/*         hybrid retrieval, reranker, abstain gate
#   codesearch/cache/*             two-tier cache
#   codesearch/api/routes_admin.py health and readiness checks
# Keep this file only so eval/configs/v1.yaml can reproduce the v1 baseline.
# =============================================================================



from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
# TODO(v2): deprecated import (langchain_chroma is the replacement). In v2 only
#     codesearch/index/vector_store.py imports chromadb.
from langchain_community.vectorstores import Chroma
# from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
# TODO(v2): langchain_classic is not in requirements.txt, and these chain helpers hide
#     every intermediate score. v2 writes each step explicitly in pipeline.py so
#     scores can be recorded in the explanation trace (explain/trace.py).
#     Also: ChatPromptTemplate is imported twice in this file.
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain


from typing import List, Tuple, Dict, Any, Optional
import re
import time
from redis_cache import RedisCache
from logger import get_logger

# Load environment variables
# TODO(v2): no environment loading at import time. In v2 only codesearch/config.py
#     reads env vars (pydantic-settings reads .env itself).
load_dotenv()
# os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")


# TODO(v2): the performance numbers in the docstring below come from one 35-query run
#     and are not reproducible. In v2, numbers come only from eval/run_eval.py
#     (logged to MLflow).
class ReactDotChatbot:
    """
    Production-ready chatbot for React documentation queries.
    
    Features:
    - Semantic search using ChromaDB vector database
    - LLM-powered answer generation (Llama 3.1 70B via Groq)
    - Redis caching for 1000x+ speedup on repeated queries
    - Comprehensive logging for monitoring and debugging
    - Error handling and fallback mechanisms
    
    Performance:
    - Average response time: 2.35s (uncached)
    - Cached response time: 0.003s (1049x speedup)
    - Cache hit rate: 42.86% (after 35 queries)
    - Success rate: 100%
    """
    
    def __init__(
            self,
            db_path: str = './react_db',
            # TODO(v2): hard-coded defaults move to LLMSettings / CacheSettings in codesearch/config.py.
            model_name: str = 'gemma3:4b',
            use_cache: bool = True,
            redis_host: str = 'localhost',
            redis_port: int = 6379
    ):
        """
        Initialize React Documentation Chatbot with Redis caching and logging.
        
        Args:
            db_path: Path to ChromaDB vector store
            model_name: Groq model name (default: llama-3.1-8b-instant)
            use_cache: Enable/disable Redis caching
            redis_host: Redis server host
            redis_port: Redis server port
        """
        # TODO(v2): replace every print() with structured logs: log.get_logger(__name__)
        #     (codesearch/log.py).
        print(" Initializing ReactDotChatbot...")
        
        # Initialize embeddings model
        print("    Loading embeddings model (Nomic-embed-text)...")
        # TODO(v2): needs a running Ollama server, which is why the Docker image could not
        #     answer queries. v2 default: SentenceTransformerEmbedder (bge-small) in
        #     index/embedder.py; Ollama stays an option chosen in factory.py.
        self.embeddings = OllamaEmbeddings(model='nomic-embed-text')

        # Load vector database
        print("    Loading vector database (ChromaDB)...")
        self.vector_db = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings
        )

        # Initialize LLM
        # TODO(v2): this message says Groq but the code below now uses ChatOllama.
        print("    Connecting to Groq LLM (Llama 3.1)...")
        # TODO(v2): ChatOllama's length limit is num_predict; check that max_tokens is really
        #     honored. v2: generation/llm.py with GroqLLM / OllamaLLM / FakeLLM behind the
        #     LLM protocol (interfaces.py), with timeouts, retries and stream().
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.3,  # Low temperature for factual responses
            max_tokens=500    # Limit response length for speed
        )
        
        # Create optimized prompt template
        # TODO(v2): prompt moves to prompts/answer.yaml with a version field. It must ask for
        #     [n] citations after each claim and allow a 'not in the React docs' answer.
        prompt = ChatPromptTemplate.from_template(
            """You are a React expert assistant. Use the following context from the official React documentation to answer the question.

IMPORTANT: Keep your answer concise and focused (max 300 words). Provide:
1. A direct answer to the question
2. One clear code example if relevant
3. Key points only - no lengthy explanations

If you don't know the answer, say so briefly.

Context:
{context}

Question: {input}

Answer:"""
        )

        # Build RAG chain
        document_chain = create_stuff_documents_chain(self.llm, prompt)
        
        # Configure retriever to fetch top 3 most relevant documents
        # TODO(v2): dense-only top-3 with no scores and no threshold. Replace with
        #     retrieval/hybrid.py (BM25 + dense, RRF), retrieval/rerank.py (top 5) and
        #     retrieval/abstain.py. This exact setup becomes eval/configs/v1.yaml.
        retriever = self.vector_db.as_retriever(search_kwargs={"k": 3})
        self.retriever_chain = create_retrieval_chain(retriever, document_chain)
        
        # Initialize Redis cache
        self.use_cache = use_cache
        self.cache = None
        
        # Initialize logger
        print("    Setting up logging system...")
        self.logger = get_logger()
        
        if use_cache:
            try:
                print("    Connecting to Redis cache...")
                # TODO(v2): cache keys must include index_version + prompt_version (cache/keys.py).
                #     When Redis is down, fall back to cache/memory.py instead of disabling caching.
                self.cache = RedisCache(ttl_hours=24)
                health = self.cache.health_check()
                if health['connected']:
                    print(f"    Redis connected (latency: {health['latency_ms']}ms)")
                else:
                    print(f"    Redis unhealthy: {health.get('error', 'Unknown error')}")
                    print(f"    Cache disabled - queries will be slower")
                    self.cache = None
            except Exception as e:
                print(f"    Redis cache disabled: {e}")
                print(f"    Tip: Make sure Redis is running (docker-compose up -d)")
                self.cache = None
        
        print(" ReactDotChatbot ready!\n")

    # TODO(v2): becomes AnswerPipeline.answer() and .stream() in pipeline.py. Returns a
    #     typed Answer (schemas.py) with trace_id, citations, groundedness, confidence.
    def ask(self, query: str, bypass_cache: bool = False) -> Dict[str, Any]:
        """
        Ask a question and get an answer with sources.
        
        This is the main method for querying the chatbot. It handles:
        - Cache checking and management
        - Query embedding and vector search
        - LLM answer generation
        - Comprehensive logging
        - Error handling
        
        Args:
            query: User's question about React
            bypass_cache: If True, skip cache and always query RAG system
            
        Returns:
            Dictionary with:
                - answer: Bot's response text
                - sources: List of source file paths from React docs
                - cached: Whether response came from cache
                - response_time: Total time taken in seconds
                - cache_latency: Redis lookup time (if cached)
                - retrieval_time: RAG pipeline time (if not cached)
        
        Example:
            >>> bot = ReactDotChatbot()
            >>> response = bot.ask("How do I use useState?")
            >>> print(response['answer'])
            >>> print(f"Cached: {response['cached']}, Time: {response['response_time']}s")
        """
        start_time = time.time()
        
        # Check cache first (if enabled and not bypassed)
        if self.use_cache and self.cache and not bypass_cache:
            cache_start = time.time()
            cached_result = self.cache.get(query)
            cache_latency = time.time() - cache_start
            
            if cached_result:
                # Cache hit! Return immediately
                total_time = time.time() - start_time
                response = {
                    **cached_result,
                    "cached": True,
                    "response_time": round(total_time, 3),
                    "cache_latency": round(cache_latency * 1000, 2)  # Convert to ms
                }
                
                # Log cache hit
                self.logger.log_query(query, response, success=True)
                self.logger.log_cache_operation('get', query, hit=True)
                
                return response
            else:
                # Cache miss - log it
                self.logger.log_cache_operation('get', query, hit=False)
        
        # Cache miss or cache disabled - query RAG system
        try:
            retrieval_start = time.time()
            
            # Execute RAG pipeline:
            # 1. Embed query
            # 2. Search vector DB for similar docs
            # 3. Pass context to LLM
            # 4. Generate answer
            result = self.retriever_chain.invoke({'input': query})
            
            retrieval_time = time.time() - retrieval_start
            
            # Extract answer from result
            answer: str = result["answer"]
            
            # Extract unique source file paths from metadata
            # TODO(v2): set() throws away retrieval order, and these are local file paths. v2 returns
            #     ordered Citation objects with react.dev URLs (ingest/urls.py,
            #     generation/citations.py).
            sources: List[str] = list(set([
                doc.metadata.get("source", "Unknown") 
                for doc in result["context"]
            ]))
            
            # Prepare response
            response = {
                "answer": answer,
                "sources": sources
            }
            
            # Cache the response for future queries
            if self.use_cache and self.cache:
                self.cache.set(query, response)
                self.logger.log_cache_operation('set', query, hit=False)
            
            total_time = time.time() - start_time
            
            final_response = {
                **response,
                "cached": False,
                "response_time": round(total_time, 3),
                "retrieval_time": round(retrieval_time, 3)
            }
            
            # Log successful query
            self.logger.log_query(query, final_response, success=True)
            
            return final_response
            
        except Exception as e:
            # Handle errors gracefully
            error_time = time.time() - start_time
            error_response = {
                # TODO(v2): never send raw exception text to users. Raise CodeSearchError subclasses
                #     (errors.py); the API maps them to HTTP codes and includes a trace_id.
                "answer": f" Error processing your question: {str(e)}",
                "sources": [],
                "cached": False,
                "response_time": round(error_time, 3),
                "error": str(e)
            }
            
            # Log error for debugging
            self.logger.log_query(query, error_response, success=False, error=str(e))
            self.logger.log_error("query_execution", str(e), {"query": query[:50]})
            
            return error_response
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get Redis cache performance statistics.
        
        Returns:
            Dict with hits, misses, hit_rate, memory_usage, etc.
            Returns error dict if cache is disabled.
        """
        if self.cache:
            return self.cache.get_stats()
        return {"error": "Cache not enabled"}
    
    def clear_cache(self) -> Dict[str, int]:
        """
        Clear all cached queries.
        
        Returns:
            Dict with number of keys deleted
        """
        if self.cache:
            return self.cache.clear()
        return {"error": "Cache not enabled"}
    
    def get_logs_summary(self) -> Dict[str, Any]:
        """
        Get logging statistics and metrics.
        
        Returns:
            Dict with total queries, error rates, response times, etc.
        """
        return self.logger.get_statistics()
    
    def print_dashboard(self) -> None:
        """
        Print real-time metrics dashboard to console.
        
        Shows:
        - Total queries and error rate
        - Cache performance (hit rate, speedup)
        - Response time distribution
        """
        self.logger.print_dashboard()
    
    def export_metrics(self, output_file: str = "metrics_summary.json") -> Dict[str, Any]:
        """
        Export comprehensive metrics to JSON file.
        
        Args:
            output_file: Path to save metrics summary
            
        Returns:
            Complete metrics dictionary
        """
        return self.logger.export_metrics_summary(output_file)
    
    def get_popular_queries(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Get most frequently asked queries.
        
        Args:
            top_n: Number of top queries to return
            
        Returns:
            List of dicts with query text and count
        """
        return self.logger.get_popular_queries(top_n)
    
    def get_slow_queries(self, threshold_seconds: float = 5.0, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Get queries that took longer than threshold.
        
        Args:
            threshold_seconds: Minimum response time to include
            top_n: Number of queries to return
            
        Returns:
            List of slow queries with response times
        """
        return self.logger.get_slow_queries(threshold_seconds, top_n)
    
    # TODO(v2): moves to GET /healthz and GET /readyz in api/routes_admin.py.
    def health_check(self) -> Dict[str, Any]:
        """
        Check system health status.
        
        Returns:
            Dict with status of Redis, vector DB, and overall system
        """
        health = {
            "status": "healthy",
            "components": {}
        }
        
        # Check Redis
        if self.cache:
            redis_health = self.cache.health_check()
            health["components"]["redis"] = redis_health
            if not redis_health["connected"]:
                health["status"] = "degraded"
        else:
            health["components"]["redis"] = {"status": "disabled"}
        
        # Check Vector DB
        try:
            # Try a simple query to verify ChromaDB is working
            # TODO(v2): BUG: embeds a query on every Streamlit rerun, and document_count is always 1.
            #     Use the collection count from index/vector_store.py.
            test_results = self.vector_db.similarity_search("test", k=1)
            health["components"]["chromadb"] = {
                "status": "healthy",
                "document_count": len(test_results)
            }
        except Exception as e:
            health["components"]["chromadb"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health["status"] = "unhealthy"
        
        return health


# Example usage and demonstration
# TODO(v2): demo script goes away; use `codesearch ask "..."` from cli.py.
if __name__ == "__main__":
    print("="*70)
    print("ReactDotChatbot - Production Demo")
    print("="*70 + "\n")
    
    # Initialize bot
    bot = ReactDotChatbot(use_cache=True)
    
    # Health check
    print("\n🏥 System Health Check:")
    health = bot.health_check()
    print(f"   Status: {health['status']}")
    for component, status in health['components'].items():
        print(f"   {component}: {status.get('status', 'unknown')}")
    
    # Test queries
    test_queries = [
        "How do I use the useState hook?",
        "What's the difference between useState and useReducer?",
        "How do I use the useState hook?"  # Duplicate to test cache
    ]
    
    print("\n" + "="*70)
    print("Testing Queries")
    print("="*70)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"Query {i}: {query}")
        print('='*70)
        
        response = bot.ask(query)
        
        # Display response
        print(f"\n💬 Answer:")
        answer_preview = response['answer'][:200] + "..." if len(response['answer']) > 200 else response['answer']
        print(answer_preview)
        
        print(f"\n📚 Sources: {len(response['sources'])} found")
        for src in response['sources'][:2]:
            print(f"   - {src}")
        
        # Display metrics
        cache_icon = "💾" if response['cached'] else "🔍"
        cache_text = "CACHE HIT" if response['cached'] else "CACHE MISS"
        print(f"\n{cache_icon} {cache_text}")
        print(f"⏱️  Response Time: {response['response_time']}s")
        
        if response['cached']:
            print(f"   Cache Latency: {response.get('cache_latency', 'N/A')}ms")
        else:
            print(f"   Retrieval Time: {response.get('retrieval_time', 'N/A')}s")
        
        # Small delay between queries
        if i < len(test_queries):
            time.sleep(0.5)
    
    # Display dashboard
    print("\n" + "="*70)
    bot.print_dashboard()
    
    # Show popular queries
    print("\n Popular Queries:")
    popular = bot.get_popular_queries(top_n=3)
    for i, item in enumerate(popular, 1):
        print(f"   {i}. {item['query']} (asked {item['count']} times)")
    
    # Export metrics
    print("\n Exporting metrics to 'metrics_summary.json'...")
    bot.export_metrics()
    
    print("\n Demo complete! Check logs/ directory for detailed logs.")
    print("="*70)