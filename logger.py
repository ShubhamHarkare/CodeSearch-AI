import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import time
from collections import deque

class CodeSearchLogger:
    """
    Production-grade logging system for CodeSearch AI.
    
    Tracks:
    - All queries and responses
    - Performance metrics
    - Cache statistics
    - Errors and exceptions
    - System health
    """
    
    def __init__(self, log_dir: str = "./logs"):
        """Initialize logging system"""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create separate log files
        self.query_log_file = self.log_dir / "queries.log"
        self.error_log_file = self.log_dir / "errors.log"
        self.metrics_log_file = self.log_dir / "metrics.log"
        
        # Set up loggers
        self._setup_query_logger()
        self._setup_error_logger()
        self._setup_metrics_logger()
        
        # In-memory metrics for real-time analytics
        self.recent_queries = deque(maxlen=100)  # Last 100 queries
        self.total_queries = 0
        self.total_errors = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        print(f"✅ Logging system initialized at {self.log_dir}")
    
    def _setup_query_logger(self):
        """Set up query logger"""
        self.query_logger = logging.getLogger("query_logger")
        self.query_logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(self.query_log_file, encoding='utf-8')
        fh.setLevel(logging.INFO)
        
        # JSON formatter for easy parsing
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", %(message)s}'
        )
        fh.setFormatter(formatter)
        
        self.query_logger.addHandler(fh)
        self.query_logger.propagate = False
    
    def _setup_error_logger(self):
        """Set up error logger"""
        self.error_logger = logging.getLogger("error_logger")
        self.error_logger.setLevel(logging.ERROR)
        
        # File handler
        fh = logging.FileHandler(self.error_log_file, encoding='utf-8')
        fh.setLevel(logging.ERROR)
        
        # Console handler for errors (see them immediately)
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        
        # Detailed formatter for errors
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.error_logger.addHandler(fh)
        self.error_logger.addHandler(ch)
        self.error_logger.propagate = False
    
    def _setup_metrics_logger(self):
        """Set up metrics logger"""
        self.metrics_logger = logging.getLogger("metrics_logger")
        self.metrics_logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(self.metrics_log_file, encoding='utf-8')
        fh.setLevel(logging.INFO)
        
        # JSON formatter
        formatter = logging.Formatter('%(message)s')
        fh.setFormatter(formatter)
        
        self.metrics_logger.addHandler(fh)
        self.metrics_logger.propagate = False
    
    def log_query(
        self,
        query: str,
        response: Dict[str, Any],
        success: bool = True,
        error: Optional[str] = None
    ):
        """
        Log a query and its response.
        
        Args:
            query: User's question
            response: Bot's response dict
            success: Whether query succeeded
            error: Error message if failed
        """
        self.total_queries += 1
        
        # Extract metrics from response
        cached = response.get('cached', False)
        response_time = response.get('response_time', 0)
        sources_count = len(response.get('sources', []))
        answer_length = len(response.get('answer', ''))
        
        # Update cache stats
        if cached:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        
        # Create log entry
        log_entry = {
            "query": query[:100],  # Truncate long queries
            "response_time": response_time,
            "cached": cached,
            "sources_count": sources_count,
            "answer_length": answer_length,
            "success": success,
            "error": error
        }
        
        # Add to recent queries for analytics
        self.recent_queries.append({
            **log_entry,
            "timestamp": datetime.now().isoformat(),
            "full_query": query
        })
        
        # Log to file
        log_msg = ", ".join([f'"{k}": {json.dumps(v)}' for k, v in log_entry.items()])
        self.query_logger.info(log_msg)
        
        # Log error if any
        if not success and error:
            self.total_errors += 1
            self.error_logger.error(f"Query failed: {query[:50]}... | Error: {error}")
    
    def log_cache_operation(self, operation: str, query: str, hit: bool):
        """
        Log cache operations for debugging.
        
        Args:
            operation: 'get' or 'set'
            query: User's question
            hit: Whether cache hit occurred (for 'get')
        """
        log_entry = {
            "operation": operation,
            "query": query[:50],
            "cache_hit": hit if operation == 'get' else None,
            "timestamp": datetime.now().isoformat()
        }
        
        # Log to metrics file
        self.metrics_logger.info(json.dumps(log_entry))
    
    def log_system_health(self, health_data: Dict[str, Any]):
        """
        Log system health metrics.
        
        Args:
            health_data: Dict with Redis status, memory usage, etc.
        """
        log_entry = {
            "type": "system_health",
            "timestamp": datetime.now().isoformat(),
            **health_data
        }
        
        self.metrics_logger.info(json.dumps(log_entry))
    
    def log_error(self, error_type: str, error_message: str, context: Optional[Dict] = None):
        """
        Log an error with context.
        
        Args:
            error_type: Type of error (e.g., 'redis_connection', 'llm_timeout')
            error_message: Error details
            context: Additional context
        """
        self.total_errors += 1
        
        log_msg = f"{error_type}: {error_message}"
        if context:
            log_msg += f" | Context: {json.dumps(context)}"
        
        self.error_logger.error(log_msg)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get real-time statistics from logged data.
        
        Returns:
            Dict with comprehensive statistics
        """
        if not self.recent_queries:
            return {
                "total_queries": 0,
                "message": "No queries logged yet"
            }
        
        # Calculate metrics from recent queries
        recent_list = list(self.recent_queries)
        response_times = [q['response_time'] for q in recent_list if q['success']]
        cached_queries = [q for q in recent_list if q['cached']]
        
        # Calculate averages
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        avg_sources = sum(q['sources_count'] for q in recent_list) / len(recent_list)
        
        # Cache statistics
        cache_hit_rate = (self.cache_hits / self.total_queries * 100) if self.total_queries > 0 else 0
        
        # Error rate
        error_rate = (self.total_errors / self.total_queries * 100) if self.total_queries > 0 else 0
        
        return {
            "total_queries": self.total_queries,
            "total_errors": self.total_errors,
            "error_rate_percent": round(error_rate, 2),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate_percent": round(cache_hit_rate, 2),
            "avg_response_time": round(avg_response_time, 3),
            "avg_sources_per_query": round(avg_sources, 2),
            "recent_queries_count": len(recent_list),
            "fastest_response": round(min(response_times), 3) if response_times else 0,
            "slowest_response": round(max(response_times), 3) if response_times else 0
        }
    
    def get_popular_queries(self, top_n: int = 10) -> list:
        """
        Get most popular queries from recent history.
        
        Args:
            top_n: Number of top queries to return
            
        Returns:
            List of popular queries with counts
        """
        from collections import Counter
        
        recent_list = list(self.recent_queries)
        queries = [q['full_query'] for q in recent_list]
        
        popular = Counter(queries).most_common(top_n)
        
        return [
            {"query": q, "count": count}
            for q, count in popular
        ]
    
    def get_slow_queries(self, threshold_seconds: float = 5.0, top_n: int = 10) -> list:
        """
        Get slowest queries above threshold.
        
        Args:
            threshold_seconds: Minimum response time to include
            top_n: Number of queries to return
            
        Returns:
            List of slow queries
        """
        recent_list = list(self.recent_queries)
        
        slow_queries = [
            {
                "query": q['full_query'],
                "response_time": q['response_time'],
                "timestamp": q['timestamp']
            }
            for q in recent_list
            if q['response_time'] > threshold_seconds
        ]
        
        # Sort by response time (slowest first)
        slow_queries.sort(key=lambda x: x['response_time'], reverse=True)
        
        return slow_queries[:top_n]
    
    def export_metrics_summary(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Export comprehensive metrics summary.
        
        Args:
            output_file: Optional file path to save JSON summary
            
        Returns:
            Complete metrics summary
        """
        stats = self.get_statistics()
        popular = self.get_popular_queries(top_n=5)
        slow = self.get_slow_queries(top_n=5)
        
        summary = {
            "export_timestamp": datetime.now().isoformat(),
            "overall_statistics": stats,
            "popular_queries": popular,
            "slow_queries": slow
        }
        
        # Save to file if specified
        if output_file:
            output_path = Path(output_file)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"📊 Metrics summary exported to {output_path}")
        
        return summary
    
    def print_dashboard(self):
        """Print a real-time dashboard to console"""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("CODESEARCH AI - REAL-TIME DASHBOARD")
        print("="*60)
        
        print(f"\n OVERALL METRICS:")
        print(f"   Total Queries: {stats['total_queries']}")
        print(f"   Total Errors: {stats['total_errors']} ({stats['error_rate_percent']}%)")
        print(f"   Avg Response Time: {stats['avg_response_time']}s")
        
        print(f"\n CACHE PERFORMANCE:")
        print(f"   Hits: {stats['cache_hits']}")
        print(f"   Misses: {stats['cache_misses']}")
        print(f"   Hit Rate: {stats['cache_hit_rate_percent']}%")
        
        print(f"\n⚡ RESPONSE TIMES:")
        print(f"   Fastest: {stats['fastest_response']}s")
        print(f"   Slowest: {stats['slowest_response']}s")
        print(f"   Average: {stats['avg_response_time']}s")
        
        print("\n" + "="*60 + "\n")


# Global logger instance
_logger_instance = None

def get_logger() -> CodeSearchLogger:
    """Get or create global logger instance"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = CodeSearchLogger()
    return _logger_instance


# Example usage and testing
if __name__ == "__main__":
    print("Testing CodeSearch AI Logging System\n")
    
    # Initialize logger
    logger = CodeSearchLogger()
    
    # Simulate some queries
    test_queries = [
        ("How do I use useState?", {"answer": "useState is...", "sources": ["doc1"], "cached": False, "response_time": 2.5}),
        ("What is JSX?", {"answer": "JSX is...", "sources": ["doc2", "doc3"], "cached": False, "response_time": 3.1}),
        ("How do I use useState?", {"answer": "useState is...", "sources": ["doc1"], "cached": True, "response_time": 0.003}),  # Cache hit
        ("Form validation?", {"answer": "To validate...", "sources": ["doc4"], "cached": False, "response_time": 2.8}),
    ]
    
    print("Logging sample queries...\n")
    for query, response in test_queries:
        logger.log_query(query, response, success=True)
        time.sleep(0.1)
    
    # Log an error
    logger.log_error("test_error", "This is a test error", {"context": "testing"})
    
    # Print dashboard
    logger.print_dashboard()
    
    # Get popular queries
    print("\n Popular Queries:")
    popular = logger.get_popular_queries(top_n=3)
    for i, item in enumerate(popular, 1):
        print(f"   {i}. {item['query']} (asked {item['count']} times)")
    
    # Export summary
    summary = logger.export_metrics_summary("test_metrics_summary.json")
    
    print("\n Logging system test complete!")
    print(f"   Check logs/ directory for generated log files")