"""
Standalone test script for Smart Router

This script uses direct imports and should be placed in the serverRouter directory.
It will test the router with a sample query and display the results.
"""

import os
import sys
import logging
from pathlib import Path
import pickle
import json
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to sys.path to allow importing core modules
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if parent_dir not in sys.path:
    sys.path.append(str(parent_dir))

# Direct imports
try:
    sys.path.append(str(current_dir))
    # Import core modules
    from core.datamodels import ChatMessage, SmartRouterRequest, ModelInfo
    from core.models import CHAT_MODELS
    
    # Import our own modules
    from smartRouter.embedding_model import OpenAIEmbeddings
    from smartRouter.smart_router import SmartRouter
    
    logger.info("Successfully imported all modules")
except ImportError as e:
    logger.error(f"Failed to import modules: {e}")
    sys.exit(1)

def test_with_query(query: str, top_k: int = 3) -> None:
    """Test the smart router with a specific query."""
    print(f"\nTesting with query: '{query}'")
    
    # Initialize the router
    try:
        router = SmartRouter()
        print(f"Router initialized with {len(router.models)} models")
    except Exception as e:
        logger.error(f"Error initializing router: {e}")
        return
    
    # Create the request
    messages = [ChatMessage(role="user", content=query)]
    request = SmartRouterRequest(
        messages=messages,
        rel_accuracy=0.6,
        rel_cost=0.3,
        rel_latency=0.1,
        k=top_k,
        verbose=True
    )
    
    # Get model recommendations
    try:
        result = router.select_models(request)
        
        # Display task analysis
        print("\n--- TASK ANALYSIS ---")
        for task, score in sorted(result["identified_tasks"].items(), key=lambda x: x[1], reverse=True):
            if score > 0.05:
                print(f"- {task.replace('_', ' ').title()}: {score:.2f}")
        
        # Display recommended models
        print("\n--- RECOMMENDED MODELS ---")
        if not result["selected_models"]:
            print("No models selected!")
            return
            
        for i, model_name in enumerate(result["selected_models"], 1):
            details = result["model_details"][model_name]
            model_info = router.models.get(model_name)
            provider = details.get("provider", "unknown")
            
            print(f"{i}. {model_name} ({provider})")
            print(f"   Score: {details['score']:.3f}")
            print(f"   Accuracy: {details['accuracy']:.3f}")
            print(f"   Cost-Efficiency: {details['cost_efficiency']:.3f}")
            print(f"   Speed: {details['speed']:.3f}")
            if model_info and model_info.max_tokens:
                print(f"   Context Length: {model_info.max_tokens} tokens")
            print()
        
    except Exception as e:
        logger.error(f"Error getting model recommendations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test with command line arg or default query
    query = sys.argv[1] if len(sys.argv) > 1 else "Write a Python function to calculate the Fibonacci sequence"
    test_with_query(query)
    
    # Optional: Add more test queries
    test_queries = [
        "Explain the theory of relativity",
        "Calculate the integral of x^2 * sin(x)",
        "Write a short story about time travel",
        "What's the capital of France?",
        "Debug this JavaScript code: function sortArray(arr) { return arr.soft((a,b) => a-b); }"
    ]
    
    print("\nWould you like to test with more queries? (y/n)")
    response = input().strip().lower()
    if response == 'y':
        for test_query in test_queries:
            test_with_query(test_query)