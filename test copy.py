import requests
import sseclient
import json
import time
key = "omni-fnBjMyX738GAZMVIlyYhTXncoMqVkvAu"

url = "http://localhost:8000/v1/chat/completions"


# Prepare the request payload
payload = {
    "model": "gemini-2.0-flash-lite",
    "messages": [
        {"role": "user", "content": "Tell me a joke"}
    ],
    "temperature": 0.7,
    "max_tokens": 100,
}

# Set up headers
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {key}"
}

# Make the request
response = requests.post(url, json=payload, headers=headers, stream=True)


print(response.json())