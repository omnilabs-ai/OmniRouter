"""
OmniRouter Reasoning API Example

This example demonstrates how to use the OmniRouter Reasoning API
to access models with enhanced reasoning capabilities.
"""

import requests
import json
import os
import sseclient
import time

# Get API key from environment variable or set it directly
API_KEY = os.environ.get("OMNI_API_KEY", "your_api_key_here")

# Base URL for the API
API_BASE = "https://api.omnirouter.ai/v1"

# Headers for the API request
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

def list_reasoning_models():
    """Get available reasoning models"""
    response = requests.get(f"{API_BASE}/models/reasoning", headers=headers)
    
    if response.status_code == 200:
        models = response.json()["models"]
        print(f"Available reasoning models: {len(models)}")
        
        for model in models:
            print(f"- {model['id']}: {model['description'][:100]}...")
            print(f"  Provider: {model['provider']}, Max tokens: {model['max_tokens']}")
            print(f"  Thinking budget: {model.get('thinking_budget', 'N/A')}")
            print()
    else:
        print(f"Error listing models: {response.status_code}")
        print(response.text)

def reasoning_completion(problem, model_id, effort="medium"):
    """Get a reasoning completion for the given problem"""
    url = f"{API_BASE}/reason/completions"
    
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": problem
            }
        ],
        "reasoning_effort": effort,
        "temperature": 1.0,
        "max_tokens": 2000
    }
    
    print(f"Sending reasoning request with effort level: {effort}")
    start_time = time.time()
    
    response = requests.post(url, headers=headers, json=payload)
    
    elapsed = time.time() - start_time
    print(f"Response received in {elapsed:.2f} seconds")
    
    if response.status_code == 200:
        result = response.json()
        
        print("\n" + "="*50)
        print(f"MODEL: {result['model']}")
        print("="*50)
        print(f"{result['content']}")
        print("="*50)
        
        print("\nToken usage:")
        print(f"  Input tokens: {result['usage']['input_tokens']}")
        print(f"  Output tokens: {result['usage']['output_tokens']}")
        print(f"  Reasoning tokens: {result['usage']['reasoning_tokens']}")
        print(f"  Total tokens: {result['usage']['total_tokens']}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

def reasoning_streaming(problem, model_id, effort="medium"):
    """Stream a reasoning completion for the given problem"""
    url = f"{API_BASE}/reason/completions/stream"
    
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": problem
            }
        ],
        "reasoning_effort": effort,
        "temperature": 1.0,
        "max_tokens": 2000
    }
    
    print(f"Sending streaming reasoning request with effort level: {effort}")
    
    response = requests.post(url, headers=headers, json=payload, stream=True)
    
    if response.status_code == 200:
        # Create an SSE client
        client = sseclient.SSEClient(response)
        
        # Track timing
        start_time = time.time()
        last_event_time = start_time
        
        # Initialize content buffers
        reasoning_buffer = []
        content_buffer = []
        
        # Process events
        for event in client.events():
            current_time = time.time()
            elapsed = current_time - last_event_time
            last_event_time = current_time
            
            if event.event == "reasoning":
                # Process reasoning event
                data = json.loads(event.data)
                reasoning_buffer.append(data.get("content", ""))
                print(f"[REASONING] {data.get('content', '')}", end="", flush=True)
            
            elif event.event == "content":
                # Process content event
                data = json.loads(event.data)
                content_buffer.append(data.get("content", ""))
                print(f"\n[CONTENT] {data.get('content', '')}", end="", flush=True)
            
            elif event.event == "usage":
                # Process usage information
                data = json.loads(event.data)
                print("\n\nToken usage:")
                print(f"  Input tokens: {data.get('input_tokens', 0)}")
                print(f"  Output tokens: {data.get('output_tokens', 0)}")
                print(f"  Reasoning tokens: {data.get('reasoning_tokens', 0)}")
                print(f"  Total tokens: {data.get('total_tokens', 0)}")
                
            # Add small delay for readability in this example
            time.sleep(0.01)
        
        # Display total time
        total_time = time.time() - start_time
        print(f"\n\nResponse completed in {total_time:.2f} seconds")
        
        # Print the complete content
        print("\nFinal response content:")
        print("="*50)
        print("".join(content_buffer))
        print("="*50)
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    # List available reasoning models
    list_reasoning_models()
    
    # Define a problem that needs reasoning
    problem = "What is the sum of the first 10 prime numbers? Show your step-by-step reasoning."
    
    # Use the default model ID - change this to one from your available models
    model_id = "claude-3-7-sonnet-extended-thinking"
    
    # Example 1: Get a standard reasoning completion
    reasoning_completion(problem, model_id, effort="medium")
    
    # Example 2: Get a streaming reasoning completion
    another_problem = "Explain why the sky is blue. Step through your reasoning."
    reasoning_streaming(another_problem, model_id, effort="low")
    
    # Example 3: Try with high effort level for a more complex problem
    complex_problem = """
    A ball is thrown vertically upward with an initial velocity of 40 m/s. 
    Assuming air resistance is negligible and the acceleration due to gravity is 9.8 m/s², 
    calculate:
    1. The maximum height reached by the ball
    2. The time it takes to reach the maximum height
    3. The total time the ball is in the air
    
    Show your step-by-step reasoning and calculations.
    """
    reasoning_completion(complex_problem, model_id, effort="high") 