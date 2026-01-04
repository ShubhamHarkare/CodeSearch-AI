from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain

import os
from typing import List, Tuple, Dict, Any
import re
import time
from redis_cache import RedisCache

load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

class ReactDotChatbot:
    def __init__(
            self,
            db_path: str = './react_db',
            model_name: str = 'llama-3.1-8b-instant',
            use_cache: bool = True,
            redis_host: str = 'localhost',
            redis_port: int = 6379
    ):
        """
        Initialize React Documentation Chatbot with Redis caching.
        
        Args:
            db_path: Path to ChromaDB vector store
            model_name: Groq model name
            use_cache: Enable/disable Redis caching
            redis_host: Redis server host
            redis_port: Redis server port
        """
        print("🚀 Initializing ReactDotChatbot...")
        
        # Initialize embeddings
        print("   Loading embeddings model...")
        self.embeddings = OllamaEmbeddings(model='nomic-embed-text')

        # Load vector database
        print("   Loading vector database...")
        self.vector_db = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings
        )

        # Initialize LLM
        print("   Connecting to Groq LLM...")
        self.llm = ChatGroq(
            model=model_name,
            temperature=0.3,
            max_tokens=500
        )
        
        # Create prompt template
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
        retriever = self.vector_db.as_retriever(search_kwargs={"k": 3})
        self.retriever_chain = create_retrieval_chain(retriever, document_chain)
        
        # Initialize Redis cache
        self.use_cache = use_cache
        self.cache = None
        
        if use_cache:
            try:
                print("   Connecting to Redis cache...")
                self.cache = RedisCache(
                    host=redis_host,
                    port=redis_port,
                    ttl_hours=24
                )
                health = self.cache.health_check()
                if health['connected']:
                    print(f"Redis connected (latency: {health['latency_ms']}ms)")
                else:
                    print(f"Redis unhealthy: {health.get('error', 'Unknown error')}")
                    self.cache = None
            except Exception as e:
                print(f"Redis cache disabled: {e}")
                self.cache = None
        
        print("✅ ReactDotChatbot ready!\n")

    def ask(self, query: str, bypass_cache: bool = False) -> Dict[str, Any]:
        """
        Ask a question and get an answer with sources.
        
        Args:
            query: User's question about React
            bypass_cache: If True, skip cache and always query RAG system
            
        Returns:
            Dictionary with:
                - answer: Bot's response
                - sources: List of source file paths
                - cached: Whether response came from cache
                - response_time: Time taken in seconds
                - cache_latency: Redis lookup time (if cached)
        """
        start_time = time.time()
        
        # Check cache first (if enabled and not bypassed)
        if self.use_cache and self.cache and not bypass_cache:
            cache_start = time.time()
            cached_result = self.cache.get(query)
            cache_latency = time.time() - cache_start
            
            if cached_result:
                total_time = time.time() - start_time
                return {
                    **cached_result,
                    "cached": True,
                    "response_time": round(total_time, 3),
                    "cache_latency": round(cache_latency * 1000, 2)  # ms
                }
        
        # Cache miss - query RAG system
        try:
            retrieval_start = time.time()
            result = self.retriever_chain.invoke({'input': query})
            retrieval_time = time.time() - retrieval_start
            
            answer: str = result["answer"]
            
            # Extract unique source file paths
            sources: List[str] = list(set([
                doc.metadata.get("source", "Unknown") 
                for doc in result["context"]
            ]))
            
            # Prepare response
            response = {
                "answer": answer,
                "sources": sources
            }
            
            # Cache the response
            if self.use_cache and self.cache:
                self.cache.set(query, response)
            
            total_time = time.time() - start_time
            
            return {
                **response,
                "cached": False,
                "response_time": round(total_time, 3),
                "retrieval_time": round(retrieval_time, 3)
            }
            
        except Exception as e:
            error_time = time.time() - start_time
            return {
                "answer": f"Error: {str(e)}",
                "sources": [],
                "cached": False,
                "response_time": round(error_time, 3),
                "error": str(e)
            }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Returns:
            Dict with hits, misses, hit_rate, etc.
        """
        if self.cache:
            return self.cache.get_stats()
        return {"error": "Cache not enabled"}
    
    def clear_cache(self) -> Dict[str, int]:
        """Clear all cached queries"""
        if self.cache:
            return self.cache.clear()
        return {"error": "Cache not enabled"}
    
    def reset_cache_stats(self) -> None:
        """Reset cache hit/miss counters"""
        if self.cache:
            self.cache.reset_stats()
        else:
            print("Cache not enabled")


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("ReactDotChatbot with Redis Cache - Demo")
    print("="*60 + "\n")
    
    # Initialize bot
    bot = ReactDotChatbot(use_cache=True)
    
    # Test queries
    test_queries = [
        "How do I use the useState hook?",
        "What's the difference between useState and useReducer?",
        "How do I use the useState hook?"  # Duplicate to test cache
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}: {query}")
        print('='*60)
        
        response = bot.ask(query)
        
        # Display response
        print(f"\n💬 Answer:")
        print(response['answer'][:200] + "..." if len(response['answer']) > 200 else response['answer'])
        
        print(f"\n📚 Sources: {len(response['sources'])} found")
        for src in response['sources'][:2]:
            print(f"   - {src}")
        
        # Display metrics
        print(f"\n📊 Metrics:")
        print(f"   Cached: {'✅ Yes' if response['cached'] else 'No'}")
        print(f"   Response Time: {response['response_time']}s")
        
        if response['cached']:
            print(f"   Cache Latency: {response.get('cache_latency', 'N/A')}ms")
        else:
            print(f"   Retrieval Time: {response.get('retrieval_time', 'N/A')}s")
    
    # Display cache statistics
    print(f"\n{'='*60}")
    print("CACHE STATISTICS")
    print('='*60)
    
    stats = bot.get_cache_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\nDemo complete!")