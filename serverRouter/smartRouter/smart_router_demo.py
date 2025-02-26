"""
Smart Router Demo Script

This script demonstrates the functionality of the Omni Project's SmartRouter.
It allows interactive testing of different queries and preferences.
"""

import os
import sys
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

# Ensure we can import from parent directory
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from serverRouter.core.datamodels import ChatMessage, SmartRouterRequest, ModelProvider
from serverRouter.smartRouter.smart_router import SmartRouter

def print_header(text: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80 + "\n")

def print_section(text: str) -> None:
    """Print a formatted section header."""
    print("\n" + "-" * 80)
    print(text)
    print("-" * 80)

def format_model_score(model_name: str, details: Dict[str, Any], max_name_len: int = 30) -> str:
    """Format model score information for display."""
    score = details["score"]
    accuracy = details["accuracy"]
    cost = details["cost_efficiency"]
    speed = details["speed"]
    provider = details["provider"].upper() if isinstance(details["provider"], str) else "UNKNOWN"
    
    # Format percentages
    acc_pct = f"{accuracy*100:.1f}%"
    cost_pct = f"{cost*100:.1f}%"
    speed_pct = f"{speed*100:.1f}%"
    
    # Pad model name for alignment
    padded_name = model_name.ljust(max_name_len)
    
    return f"{padded_name} | Score: {score:.3f} | Accuracy: {acc_pct} | Cost: {cost_pct} | Speed: {speed_pct} | {provider}"

def run_demo() -> None:
    """Run the interactive SmartRouter demo."""
    print_header("Omni Project Smart Router Demo")
    
    # Initialize the SmartRouter
    router = SmartRouter()
    
    # If benchmark embeddings file doesn't exist, create mock embeddings
    if not router.benchmark_embeddings:
        print("Warning: Benchmark embeddings not found. Using default weights.")
    
    while True:
        print_section("TEST OPTIONS")
        print("1. Test with custom query")
        print("2. Run with example queries")
        print("3. Compare preference settings")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            test_custom_query(router)
        elif choice == "2":
            test_example_queries(router)
        elif choice == "3":
            compare_preferences(router)
        elif choice == "4":
            print("\nThanks for using the Omni Smart Router Demo!")
            break
        else:
            print("Invalid choice, please try again.")

def test_custom_query(router: SmartRouter) -> None:
    """Test the router with a custom user query."""
    print_section("CUSTOM QUERY TEST")
    
    # Get user input
    query = input("Enter your query: ").strip()
    
    if not query:
        print("Query cannot be empty.")
        return
    
    # Get preference weights
    print("\nEnter preference weights (0.0-1.0):")
    try:
        rel_accuracy = float(input("Accuracy importance: ").strip() or "0.5")
        rel_cost = float(input("Cost importance: ").strip() or "0.3")
        rel_latency = float(input("Speed importance: ").strip() or "0.2")
    except ValueError:
        print("Invalid input. Using default weights.")
        rel_accuracy = 0.5
        rel_cost = 0.3
        rel_latency = 0.2
    
    # Create request
    messages = [ChatMessage(role="user", content=query)]
    request = SmartRouterRequest(
        messages=messages,
        rel_accuracy=rel_accuracy,
        rel_cost=rel_cost,
        rel_latency=rel_latency,
        k=5,
        verbose=True
    )
    
    # Process request
    result = router.select_models(request)
    
    # Display results
    display_results(result)

def test_example_queries(router: SmartRouter) -> None:
    """Test the router with predefined example queries."""
    example_queries = [
        "Write a Python function to calculate the Fibonacci sequence.",
        "Explain the theory of relativity and its implications for modern physics.",
        "Compare and contrast the economic policies of different countries in response to inflation.",
        "Help me debug this JavaScript code: function sortArray(arr) { return arr.soft((a,b) => a-b); }",
        "Solve this calculus problem: find the integral of x^2 * sin(x) dx.",
        "Write a creative story about a time traveler who visits ancient Rome.",
        "Create an analysis of recent climate data and visualize the trends."
    ]
    
    print_section("EXAMPLE QUERIES TEST")
    
    for i, query in enumerate(example_queries, 1):
        print(f"{i}. {query}")
    
    try:
        choice = int(input("\nSelect an example query (1-7): ").strip())
        if choice < 1 or choice > len(example_queries):
            raise ValueError()
        selected_query = example_queries[choice-1]
    except (ValueError, IndexError):
        print("Invalid choice. Using the first example.")
        selected_query = example_queries[0]
    
    # Create request with balanced preferences
    messages = [ChatMessage(role="user", content=selected_query)]
    request = SmartRouterRequest(
        messages=messages,
        rel_accuracy=0.5,
        rel_cost=0.3,
        rel_latency=0.2,
        k=5,
        verbose=True
    )
    
    # Process request
    result = router.select_models(request)
    
    # Display results
    print(f"\nQuery: {selected_query}")
    display_results(result)

def compare_preferences(router: SmartRouter) -> None:
    """Compare different preference settings with the same query."""
    print_section("PREFERENCE COMPARISON TEST")
    
    # Get user query
    query = input("Enter a query to test with different preferences: ").strip()
    if not query:
        query = "Create a machine learning model to classify images of dogs and cats."
        print(f"Using example query: {query}")
    
    # Define different preference profiles
    preference_profiles = [
        {"name": "Balanced", "accuracy": 0.33, "cost": 0.33, "latency": 0.33},
        {"name": "High Accuracy", "accuracy": 0.8, "cost": 0.1, "latency": 0.1},
        {"name": "Budget-Friendly", "accuracy": 0.2, "cost": 0.7, "latency": 0.1},
        {"name": "Speed-Focused", "accuracy": 0.2, "cost": 0.1, "latency": 0.7}
    ]
    
    messages = [ChatMessage(role="user", content=query)]
    
    # Test with each profile
    for profile in preference_profiles:
        print_section(f"Profile: {profile['name']}")
        
        request = SmartRouterRequest(
            messages=messages,
            rel_accuracy=profile["accuracy"],
            rel_cost=profile["cost"],
            rel_latency=profile["latency"],
            k=3,
            verbose=False
        )
        
        # Process request
        result = router.select_models(request)
        
        # Display top 3 models
        print(f"Top models for {profile['name']} preference profile:")
        
        if "model_details" in result:
            max_name_len = max(len(name) for name in result["model_details"].keys())
            
            for name, details in result["model_details"].items():
                print(format_model_score(name, details, max_name_len))
        else:
            print("No model details available.")
        
        print("\n")

def display_results(result: Dict[str, Any]) -> None:
    """Display the results of a router query."""
    print_section("TASK ANALYSIS")
    
    if "identified_tasks" in result:
        for task, score in sorted(result["identified_tasks"].items(), key=lambda x: x[1], reverse=True):
            print(f"- {task.replace('_', ' ').title()}: {score:.2f}")
    else:
        print("No task analysis available.")
    
    print_section("BENCHMARK WEIGHTS")
    
    if "benchmark_weights" in result:
        for benchmark, weight in sorted(result["benchmark_weights"].items(), key=lambda x: x[1], reverse=True):
            print(f"- {benchmark}: {weight:.2f}")
    else:
        print("No benchmark weights available.")
    
    print_section("SELECTED MODELS")
    
    if "model_details" in result:
        max_name_len = max(len(name) for name in result["model_details"].keys())
        
        for name, details in result["model_details"].items():
            print(format_model_score(name, details, max_name_len))
    else:
        print("No model details available.")
    
    print_section("DETAILED EXPLANATION")
    
    if result.get("explanation"):
        print(result["explanation"])
    else:
        print("No detailed explanation available.")

if __name__ == "__main__":
    run_demo()