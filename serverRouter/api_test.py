import asyncio
import json
import requests
import time

# Base URL for the API
BASE_URL = "http://localhost:8000"
API_KEY = "test-sk1o83e"  # Your test API key from config.py

def test_list_agents():
    print("\n===== Testing List Agents API =====")
    
    # Call the API
    response = requests.get(
        f"{BASE_URL}/v1/agents/",
        headers={"Authorization": f"Bearer {API_KEY}"}
    )
    
    # Print the results
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_run_agent():
    print("\n===== Testing Run Agent API =====")
    
    # Create a run request
    request = {
        "agent_id": "web-search-agent",
        "inputs": {
            "prompt": "What is the capital of France?"
        },
        "stream": False
    }
    
    # Call the API
    response = requests.post(
        f"{BASE_URL}/v1/agents/run",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json=request
    )
    
    # Check if the request was successful
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.text}")
        return
    
    # Get the run ID
    run_id = response.json().get("run_id")
    print(f"Run created with ID: {run_id}")
    
    # Poll for results
    max_attempts = 10
    for attempt in range(max_attempts):
        print(f"Checking run status (attempt {attempt + 1}/{max_attempts})...")
        
        # Get the run
        run_response = requests.get(
            f"{BASE_URL}/v1/agents/runs/{run_id}",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if run_response.status_code != 200:
            print(f"Error: {run_response.status_code}")
            print(f"Response: {run_response.text}")
            break
        
        run_data = run_response.json()
        status = run_data.get("status")
        
        print(f"Status: {status}")
        
        if status in ["completed", "failed"]:
            print(f"Run {status}!")
            print(f"Output: {run_data.get('output')}")
            break
        
        time.sleep(1)  # Wait a bit before polling again

def main():
    test_list_agents()
    test_run_agent()

if __name__ == "__main__":
    main()