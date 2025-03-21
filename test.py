import requests
import sseclient
import json
import time
key = "omni-fnBjMyX738GAZMVIlyYhTXncoMqVkvAu"

url = "http://localhost:8000/v1/chat/completions/stream"


# Prepare the request payload
payload = {
    "model": "gpt-4o-mini",
    "messages": [
        {"role": "user", "content": "Write a short story about a cat"}
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

# Create SSE client
client = sseclient.SSEClient(response)

start_time = time.time()
for event in client.events():
    elapsed_time = time.time() - start_time
    print(f"{elapsed_time:.2f}s [{event.event}]: {event.data}")