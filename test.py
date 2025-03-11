import requests
import json
import os
from dotenv import load_dotenv

API_KEY = "test-sk1o83e"  # Replace with your actual API key

# Base URL for the API
BASE_URL = "http://localhost:8000"  # Change this if your server is running on a different host/port

def test_chat_completion():
    """Test the chat completion endpoint with the API key."""
    endpoint = f"{BASE_URL}/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gpt-3.5-turbo",  # Replace with a model ID that your server supports
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "max_tokens": 100
    }
    
    try:
        response = requests.post(endpoint, headers=headers, json=payload)
        
        if response.status_code == 200:
            print("✅ Request successful!")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"❌ Request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    print("Testing protected routes with API key...")
    test_chat_completion()