from chatbot import ReactDotChatbot
import json
from datetime import datetime
from typing import List, Dict, Any
import time

class ReactChatbotTester:
    """Test suite for ReactDotChatbot with 20 diverse questions"""
    
    def __init__(self):
        self.bot = ReactDotChatbot()
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
            "What is React Server Components and how do they work?"
        ]
        
        self.results: List[Dict[str, Any]] = []
    
    def run_test(self, question: str, index: int) -> Dict[str, Any]:
        """Run a single test query and collect metrics"""
        print(f"\n{'='*80}")
        print(f"Question {index + 1}/20: {question}")
        print('='*80)
        
        start_time = time.time()
        
        try:
            response = self.bot.ask(question)
            elapsed_time = time.time() - start_time
            
            answer = response['answer']
            sources = response['sources']
            
            # Display results
            print(f"\n📝 ANSWER:")
            print(f"{answer}\n")
            
            print(f"📚 SOURCES ({len(sources)} found):")
            for i, src in enumerate(sources[:3], 1):  # Show first 3 sources
                print(f"  {i}. {src[:100]}..." if len(src) > 100 else f"  {i}. {src}")
            
            print(f"\n⏱️  Response Time: {elapsed_time:.2f}s")
            
            # Collect metrics
            result = {
                "question": question,
                "answer": answer,
                "sources_count": len(sources),
                "response_time": round(elapsed_time, 2),
                "success": True,
                "answer_length": len(answer),
                "has_code": "```" in answer or "const " in answer or "function" in answer
            }
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"\n❌ ERROR: {str(e)}")
            print(f"⏱️  Failed after: {elapsed_time:.2f}s")
            
            result = {
                "question": question,
                "answer": None,
                "sources_count": 0,
                "response_time": round(elapsed_time, 2),
                "success": False,
                "error": str(e)
            }
        
        return result
    
    def run_all_tests(self) -> None:
        """Run all test questions and generate report"""
        print("\n" + "="*80)
        print("🚀 STARTING REACTDOTCHATBOT TEST SUITE")
        print("="*80)
        
        for i, question in enumerate(self.test_questions):
            result = self.run_test(question, i)
            self.results.append(result)
            
            # Small delay to avoid rate limiting
            time.sleep(1)
        
        self.generate_report()
    
    def generate_report(self) -> None:
        """Generate comprehensive test report"""
        print("\n" + "="*80)
        print("📊 TEST SUMMARY REPORT")
        print("="*80)
        
        successful_tests = [r for r in self.results if r['success']]
        failed_tests = [r for r in self.results if not r['success']]
        
        # Basic Metrics
        print(f"\n✅ Successful Queries: {len(successful_tests)}/20")
        print(f"❌ Failed Queries: {len(failed_tests)}/20")
        print(f"📈 Success Rate: {(len(successful_tests)/20)*100:.1f}%")
        
        if successful_tests:
            avg_response_time = sum(r['response_time'] for r in successful_tests) / len(successful_tests)
            avg_sources = sum(r['sources_count'] for r in successful_tests) / len(successful_tests)
            avg_answer_length = sum(r['answer_length'] for r in successful_tests) / len(successful_tests)
            code_examples = sum(1 for r in successful_tests if r.get('has_code', False))
            
            print(f"\n⏱️  Average Response Time: {avg_response_time:.2f}s")
            print(f"📚 Average Sources Retrieved: {avg_sources:.1f}")
            print(f"📝 Average Answer Length: {avg_answer_length:.0f} characters")
            print(f"💻 Answers with Code Examples: {code_examples}/{len(successful_tests)}")
            
            # Response Time Analysis
            response_times = [r['response_time'] for r in successful_tests]
            max_time = max(response_times)
            min_time = min(response_times)
            
            print(f"\n⚡ Fastest Response: {min_time:.2f}s")
            print(f"🐌 Slowest Response: {max_time:.2f}s")
        
        # Failed Tests Details
        if failed_tests:
            print(f"\n❌ FAILED TESTS:")
            for i, test in enumerate(failed_tests, 1):
                print(f"  {i}. {test['question']}")
                print(f"     Error: {test.get('error', 'Unknown')}")
        
        # Save detailed results to JSON
        self.save_results_to_file()
        
        print("\n" + "="*80)
        print("✅ Testing Complete! Results saved to 'test_results.json'")
        print("="*80)
    
    def save_results_to_file(self) -> None:
        """Save test results to JSON file"""
        output = {
            "test_date": datetime.now().isoformat(),
            "total_questions": len(self.test_questions),
            "successful": len([r for r in self.results if r['success']]),
            "failed": len([r for r in self.results if not r['success']]),
            "results": self.results
        }
        
        with open('test_results.json', 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

def main():
    """Main execution function"""
    tester = ReactChatbotTester()
    
    print("\n🔍 This test will ask 20 diverse React questions covering:")
    print("   • Beginner concepts (components, JSX, props)")
    print("   • React Hooks (useState, useEffect, useMemo)")
    print("   • Forms & Events")
    print("   • Performance optimization")
    print("   • Advanced topics (Context, Portals, Error Boundaries)")
    
    input("\nPress Enter to start testing...")
    
    tester.run_all_tests()

if __name__ == "__main__":
    main()
