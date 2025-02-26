"""
Patched Smart Router Test

This script uses the fix_imports module to properly test your smart router.
It handles the import issues and tests the router with various queries.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Apply import fixes
logger.info("Applying import fixes...")
try:
    from fix_imports import fix_imports
    fix_imports()
except ImportError:
    logger.error("Could not import fix_imports.py. Make sure it's in the same directory as this script.")
    sys.exit(1)

# Try to import required modules
logger.info("Importing modules...")
try:
    from serverRouter.core.datamodels import ChatMessage, SmartRouterRequest, ModelInfo, ModelProvider
    from serverRouter.core.models import CHAT_MODELS
    logger.info(f"Successfully imported models ({len(CHAT_MODELS)} models found)")
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    logger.error("Make sure the fix_imports.py script was executed properly and your files are in the right locations.")
    sys.exit(1)

# Define simple smart router
class SimpleRouter:
    def __init__(self):
        self.models = CHAT_MODELS
        
    def analyze_query(self, query):
        """Identify the task type from the query."""
        query = query.lower()
        
        # Check for coding queries
        if any(kw in query for kw in ["code", "function", "programming", "algorithm", "python", "javascript"]):
            return "Coding", "HumanEval"
            
        # Check for math queries
        if any(kw in query for kw in ["math", "calculate", "equation", "integral", "solve"]):
            return "Math", "MATH"
            
        # Check for science queries
        if any(kw in query for kw in ["physics", "chemistry", "biology", "science", "theory"]):
            return "Science", "GPQA"
            
        # Check for creative writing
        if any(kw in query for kw in ["write", "story", "essay", "article", "creative"]):
            return "Creative Writing", "MMLU"
            
        # Default to general knowledge
        return "General Knowledge", "MMLU"
    
    def recommend_models(self, query, top_k=3):
        """Recommend top models for a given query."""
        # Identify the query type
        task_type, primary_benchmark = self.analyze_query(query)
        
        # Score each model based on the primary benchmark
        model_scores = []
        for name, model in self.models.items():
            if not hasattr(model, 'benchmarks') or not model.benchmarks:
                continue
                
            # Get the benchmark score
            score = 0
            if hasattr(model.benchmarks, primary_benchmark):
                score = getattr(model.benchmarks, primary_benchmark) or 0
                
            model_scores.append((name, score, model))
        
        # Sort by score and return top k
        model_scores.sort(key=lambda x: x[1], reverse=True)
        return task_type, primary_benchmark, model_scores[:top_k]

def main():
    print("\n" + "="*80)
    print("SMART ROUTER TEST".center(80))
    print("="*80)
    
    # Initialize router
    router = SimpleRouter()
    
    # Test queries
    test_queries = [
        "Write a Python function to calculate the Fibonacci sequence",
        "Calculate the integral of x^2 * sin(x)",
        "Explain Einstein's theory of relativity",
        "Write a short story about a time traveler",
        "What was the significance of the Industrial Revolution?",
        "Debug this JavaScript code: function sortArray(arr) { return arr.soft((a,b) => a-b); }"
    ]
    
    # Ask user to select a query
    print("\nSelect a query to test:")
    for i, query in enumerate(test_queries, 1):
        print(f"{i}. {query}")
    print(f"{len(test_queries) + 1}. Custom query")
    
    try:
        choice = int(input("\nEnter selection (1-7): "))
        if 1 <= choice <= len(test_queries):
            query = test_queries[choice - 1]
        else:
            query = input("\nEnter your custom query: ")
    except (ValueError, IndexError):
        print("Invalid selection. Using default query.")
        query = test_queries[0]
    
    print(f"\nTesting with query: '{query}'")
    
    # Get model recommendations
    task_type, benchmark, top_models = router.recommend_models(query)
    
    print(f"\nIdentified task type: {task_type}")
    print(f"Primary benchmark: {benchmark}")
    
    print("\nTop recommended models:")
    for i, (name, score, model) in enumerate(top_models, 1):
        print(f"\n{i}. {name} ({model.provider.value})")
        print(f"   Benchmark score: {score:.3f}")
        print(f"   Provider: {model.provider.value}")
        print(f"   Max tokens: {model.max_tokens}")
        if hasattr(model, 'tokenCost') and model.tokenCost is not None:
            print(f"   Cost per 1M tokens: ${model.tokenCost:.2f}")
        if hasattr(model, 'latency') and model.latency is not None:
            print(f"   Average latency: {model.latency:.2f}s")
        print(f"   Description: {model.description[:100]}...")

if __name__ == "__main__":
    main()