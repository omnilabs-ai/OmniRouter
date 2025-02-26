"""
Test script for the Smart Router

This script demonstrates how to use the SmartRouter to recommend models
based on different queries and preference settings.

Usage:
    python test_router.py "Your query here"
"""

import sys
import os
import logging
from pathlib import Path

# Add the project root to the path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent if current_dir.name == 'smartRouter' else current_dir
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    # Import required modules
    from serverRouter.core.datamodels import ChatMessage, SmartRouterRequest
    from serverRouter.smartRouter.smart_router import SmartRouter
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    sys.exit(1)

def main():
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python test_router.py \"Your query here\"")
        print("Testing with default query: What is the capital of France?")
        query = "What is the capital of France?"
    else:
        query = sys.argv[1]

    print("\n" + "="*80)
    print("SMART ROUTER TEST".center(80))
    print("="*80 + "\n")
    
    print(f"Query: {query}\n")
    
    # Initialize the router
    try:
        router = SmartRouter()
    except Exception as e:
        logger.error(f"Error initializing SmartRouter: {e}")
        sys.exit(1)
    
    # Create the request
    messages = [ChatMessage(role="user", content=query)]
    
    # Test with different preference settings
    preference_sets = [
        {"name": "Balanced", "accuracy": 0.33, "cost": 0.33, "latency": 0.33},
        {"name": "High Accuracy", "accuracy": 0.8, "cost": 0.1, "latency": 0.1},
        {"name": "Budget-Friendly", "accuracy": 0.2, "cost": 0.7, "latency": 0.1},
        {"name": "Speed-Focused", "accuracy": 0.2, "cost": 0.1, "latency": 0.7}
    ]
    
    for prefs in preference_sets:
        print("\n" + "-"*80)
        print(f"Preference Profile: {prefs['name']}")
        print("-"*80)
        
        request = SmartRouterRequest(
            messages=messages,
            rel_accuracy=prefs["accuracy"],
            rel_cost=prefs["cost"],
            rel_latency=prefs["latency"],
            k=3,  # Get top 3 models
            verbose=True  # Include detailed explanation
        )
        
        # Get model recommendations
        try:
            result = router.select_models(request)
            
            # Display task analysis
            print("\nTask Analysis:")
            for task, score in sorted(result["identified_tasks"].items(), key=lambda x: x[1], reverse=True):
                if score > 0.05:  # Only show relevant tasks
                    print(f"- {task.replace('_', ' ').title()}: {score:.2f}")
            
            # Display selected models
            print("\nRecommended Models:")
            for i, model_name in enumerate(result["selected_models"], 1):
                details = result["model_details"][model_name]
                print(f"{i}. {model_name} (Score: {details['score']:.3f}, Accuracy: {details['accuracy']:.3f}, "
                     f"Cost: {details['cost_efficiency']:.3f}, Speed: {details['speed']:.3f})")
                
        except Exception as e:
            logger.error(f"Error selecting models: {e}")
            continue

if __name__ == "__main__":
    main()