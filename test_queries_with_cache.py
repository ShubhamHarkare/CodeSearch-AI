from chatbot import ReactDotChatbot
import json
from datetime import datetime
from typing import List, Dict, Any
import time

class ReactChatbotTesterWithCache:
    """Enhanced test suite with Redis cache performance tracking"""
    
    def __init__(self):
        self.bot = ReactDotChatbot(use_cache=True)
        
        # Clear cache for clean test (removed reset_cache_stats - doesn't exist)
        print("🗑️ Clearing cache for fresh test...")
        result = self.bot.clear_cache()
        print(f"   Cleared {result.get('deleted', 0)} cached queries")
        
        self.test_questions = [
            # Beginner Level (1-5)
            "How do I create a simple React component?",
            "What is JSX and why do we use it?",
            "How do I handle button clicks in React?",
            "What's the difference between props and state?",
            "How do I render a list of items in React?",
            
            # Hooks Related (6-10)
            "How do I use the useState hook?",
            "What's the difference between useState and useReducer?",
            "How do I fetch data when a component mounts using useEffect?",
            "How do I optimize performance with useMemo?",
            "What is useCallback and when should I use it?",
            
            # Forms & Events (11-13)
            "How do I handle form validation in React?",
            "How do I create controlled components for forms?",
            "How do I prevent default form submission?",
            
            # Performance & Optimization (14-16)
            "Show me how to optimize performance with React.memo",
            "How do I use lazy loading for components?",
            "What causes unnecessary re-renders and how do I fix them?",
            
            # Advanced Concepts (17-20)
            "How do I use React Context API?",
            "What are React portals and when should I use them?",
            "How do I handle errors with Error Boundaries?",
            "What is React Server Components and how do they work?",
            
            # DUPLICATE queries to test cache (21-25)
            "How do I use the useState hook?",  # Duplicate of #6
            "What is JSX and why do we use it?",  # Duplicate of #2
            "How do I handle button clicks in React?",  # Duplicate of #3
            "How do I create controlled components for forms?",  # Duplicate of #12
            "How do I use React Context API?"  # Duplicate of #17
        ]
        
        self.results: List[Dict[str, Any]] = []
    
    def run_test(self, question: str, index: int) -> Dict[str, Any]:
        """Run a single test query and collect metrics"""
        print(f"\n{'='*80}")
        print(f"Question {index + 1}/{len(self.test_questions)}: {question}")
        print('='*80)
        
        try:
            response = self.bot.ask(question)
            
            answer = response['answer']
            sources = response['sources']
            cached = response['cached']
            response_time = response['response_time']
            
            # Display results
            print(f"\n💬 ANSWER:")
            print(f"{answer[:150]}..." if len(answer) > 150 else answer)
            
            print(f"\n📚 SOURCES: {len(sources)} found")
            
            # Cache indicator
            cache_emoji = "💾" if cached else "🔍"
            cache_status = "CACHE HIT" if cached else "CACHE MISS"
            print(f"\n{cache_emoji} {cache_status}")
            print(f"⏱️  Response Time: {response_time}s")
            
            if cached:
                print(f"   Cache Latency: {response.get('cache_latency', 'N/A')}ms")
            else:
                print(f"   Retrieval Time: {response.get('retrieval_time', 'N/A')}s")
            
            # Collect metrics
            result = {
                "question": question,
                "answer": answer,
                "sources_count": len(sources),
                "response_time": response_time,
                "cached": cached,
                "success": True,
                "answer_length": len(answer),
                "has_code": "```" in answer or "const " in answer or "function" in answer
            }
            
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            
            result = {
                "question": question,
                "answer": None,
                "sources_count": 0,
                "response_time": 0,
                "cached": False,
                "success": False,
                "error": str(e)
            }
        
        return result
    
    def run_all_tests(self) -> None:
        """Run all test questions and generate report"""
        print("\n" + "="*80)
        print("🚀 STARTING REACTDOTCHATBOT TEST SUITE WITH REDIS CACHE & LOGGING")
        print("="*80)
        
        for i, question in enumerate(self.test_questions):
            result = self.run_test(question, i)
            self.results.append(result)
            
            # Small delay to avoid overwhelming the system
            if not result.get('cached', False):
                time.sleep(0.5)
        
        self.generate_report()
    
    def generate_report(self) -> None:
        """Generate comprehensive test report with cache analysis"""
        print("\n" + "="*80)
        print("📊 TEST SUMMARY REPORT")
        print("="*80)
        
        successful_tests = [r for r in self.results if r['success']]
        failed_tests = [r for r in self.results if not r['success']]
        cached_responses = [r for r in successful_tests if r['cached']]
        uncached_responses = [r for r in successful_tests if not r['cached']]
        
        # Basic Metrics
        print(f"\n✅ Successful Queries: {len(successful_tests)}/{len(self.test_questions)}")
        print(f"❌ Failed Queries: {len(failed_tests)}/{len(self.test_questions)}")
        print(f"📈 Success Rate: {(len(successful_tests)/len(self.test_questions))*100:.1f}%")
        
        # Cache Performance
        print(f"\n💾 CACHE PERFORMANCE:")
        print(f"   Cache Hits: {len(cached_responses)}")
        print(f"   Cache Misses: {len(uncached_responses)}")
        if len(successful_tests) > 0:
            print(f"   Hit Rate: {(len(cached_responses)/len(successful_tests)*100):.1f}%")
        
        if successful_tests:
            # Overall metrics
            avg_response_time = sum(r['response_time'] for r in successful_tests) / len(successful_tests)
            
            print(f"\n⏱️  RESPONSE TIME ANALYSIS:")
            print(f"   Average (All): {avg_response_time:.2f}s")
            
            if cached_responses:
                avg_cached_time = sum(r['response_time'] for r in cached_responses) / len(cached_responses)
                print(f"   Average (Cached): {avg_cached_time:.3f}s ⚡")
            
            if uncached_responses:
                avg_uncached_time = sum(r['response_time'] for r in uncached_responses) / len(uncached_responses)
                print(f"   Average (Uncached): {avg_uncached_time:.2f}s")
            
            # Speed improvement
            if cached_responses and uncached_responses:
                speedup = avg_uncached_time / avg_cached_time
                print(f"   🚀 Cache Speedup: {speedup:.1f}x faster!")
            
            # Other metrics
            avg_sources = sum(r['sources_count'] for r in successful_tests) / len(successful_tests)
            avg_answer_length = sum(r['answer_length'] for r in successful_tests) / len(successful_tests)
            code_examples = sum(1 for r in successful_tests if r.get('has_code', False))
            
            print(f"\n📚 CONTENT METRICS:")
            print(f"   Average Sources: {avg_sources:.1f}")
            print(f"   Average Answer Length: {avg_answer_length:.0f} chars")
            print(f"   Answers with Code: {code_examples}/{len(successful_tests)}")
            
            # Response Time Distribution
            response_times = [r['response_time'] for r in successful_tests]
            max_time = max(response_times)
            min_time = min(response_times)
            
            print(f"\n⚡ RESPONSE TIME RANGE:")
            print(f"   Fastest: {min_time:.3f}s")
            print(f"   Slowest: {max_time:.2f}s")
        
        # Redis Cache Stats
        print(f"\n💾 REDIS CACHE STATISTICS:")
        cache_stats = self.bot.get_cache_stats()
        for key, value in cache_stats.items():
            print(f"   {key}: {value}")
        
        # Logging System Stats
        print(f"\n📝 LOGGING STATISTICS:")
        log_stats = self.bot.get_logs_summary()
        for key, value in log_stats.items():
            print(f"   {key}: {value}")
        
        # Get popular queries
        print(f"\n🔥 MOST POPULAR QUERIES:")
        popular = self.bot.get_popular_queries(top_n=5)
        for i, item in enumerate(popular, 1):
            print(f"   {i}. {item['query'][:50]}... (asked {item['count']} times)")
        
        # Get slow queries
        slow_queries = self.bot.get_slow_queries(threshold_seconds=5.0, top_n=3)
        if slow_queries:
            print(f"\n🐌 SLOWEST QUERIES (>5s):")
            for i, item in enumerate(slow_queries, 1):
                print(f"   {i}. {item['query'][:50]}... ({item['response_time']}s)")
        
        # Export comprehensive metrics
        print(f"\n📊 Exporting comprehensive metrics...")
        self.bot.export_metrics("test_metrics_summary.json")
        
        # Failed Tests
        if failed_tests:
            print(f"\n❌ FAILED TESTS:")
            for i, test in enumerate(failed_tests, 1):
                print(f"  {i}. {test['question']}")
                print(f"     Error: {test.get('error', 'Unknown')}")
        
        # Save results
        self.save_results_to_file()
        
        print("\n" + "="*80)
        print("✅ Testing Complete!")
        print("📄 Results saved to: 'test_results_cached.json'")
        print("📊 Metrics saved to: 'test_metrics_summary.json'")
        print("📝 Logs available in: 'logs/' directory")
        print("="*80)
    
    def save_results_to_file(self) -> None:
        """Save test results to JSON file"""
        cache_stats = self.bot.get_cache_stats()
        log_stats = self.bot.get_logs_summary()
        
        output = {
            "test_date": datetime.now().isoformat(),
            "total_questions": len(self.test_questions),
            "successful": len([r for r in self.results if r['success']]),
            "failed": len([r for r in self.results if not r['success']]),
            "cache_stats": cache_stats,
            "logging_stats": log_stats,
            "results": self.results
        }
        
        with open('test_results_cached.json', 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

def main():
    """Main execution function"""
    tester = ReactChatbotTesterWithCache()
    
    print("\n🔍 This enhanced test includes:")
    print("   • 20 unique React questions")
    print("   • 5 duplicate questions to test caching")
    print("   • Redis cache performance tracking")
    print("   • Comprehensive logging & metrics")
    print("   • Response time analysis (cached vs uncached)")
    print("   • Popular queries tracking")
    print("   • Slow query identification")
    
    input("\nPress Enter to start testing...")
    
    tester.run_all_tests()
    
    # Optional: Print dashboard at the end
    print("\n" + "="*80)
    print("📊 FINAL DASHBOARD")
    print("="*80)
    tester.bot.print_dashboard()

if __name__ == "__main__":
    main()
    