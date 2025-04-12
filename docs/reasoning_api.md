# OmniRouter Reasoning API Documentation

## Overview

The OmniRouter Reasoning API provides access to models with enhanced reasoning capabilities, allowing you to get detailed step-by-step thinking for complex problems. This feature is particularly valuable for tasks requiring transparent reasoning processes such as:

- Mathematical problem solving
- Logical analysis
- Complex decision making
- Step-by-step explanations
- Educational content

## Available Reasoning Models

You can retrieve a list of available reasoning models by making a GET request to:

```
GET /v1/models/reasoning
```

Currently supported models include:

- `claude-3-7-sonnet-extended-thinking` - Claude 3.7 Sonnet with extended thinking capabilities

## Making Reasoning Requests

### Non-Streaming Endpoint

```
POST /v1/reason/completions
```

This endpoint generates a response with detailed reasoning for your prompt.

#### Request Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | **Required**. The model to use (e.g., `claude-3-7-sonnet-extended-thinking`) |
| `messages` | array | **Required**. An array of message objects representing the conversation history |
| `reasoning_effort` | string | The level of reasoning effort: `low`, `medium` (default), or `high` |
| `max_tokens` | integer | Maximum number of tokens to generate in the response |
| `temperature` | float | Sampling temperature (0-2.0). Default is 1.0. Note: For Claude models, this must be 1.0 when using reasoning |

#### Example Request

```json
{
  "model": "claude-3-7-sonnet-extended-thinking",
  "messages": [
    {
      "role": "user",
      "content": "What is the sum of the first 10 prime numbers? Show your step-by-step reasoning."
    }
  ],
  "reasoning_effort": "medium",
  "temperature": 1.0,
  "max_tokens": 2000
}
```

#### Example Response

```json
{
  "model": "claude-3-7-sonnet-20250219",
  "content": "# Sum of the First 10 Prime Numbers\n\nTo find the sum of the first 10 prime numbers, I'll first identify them, then add them together.\n\n## Step 1: List the first 10 prime numbers\n1. 2\n2. 3\n3. 5\n4. 7\n5. 11\n6. 13\n7. 17\n8. 19\n9. 23\n10. 29\n\n## Step 2: Calculate their sum\n2 + 3 = 5\n5 + 5 = 10\n10 + 7 = 17\n17 + 11 = 28\n28 + 13 = 41\n41 + 17 = 58\n58 + 19 = 77\n77 + 23 = 100\n100 + 29 = 129\n\nTherefore, the sum of the first 10 prime numbers is 129.",
  "provider": "anthropic",
  "usage": {
    "input_tokens": 84,
    "output_tokens": 185,
    "reasoning_tokens": 430,
    "total_tokens": 699
  }
}
```

### Streaming Endpoint

```
POST /v1/reason/completions/stream
```

This endpoint streams both the reasoning process and final response using Server-Sent Events (SSE).

#### Request Parameters

The same parameters as the non-streaming endpoint, with `stream` automatically set to `true`.

#### Example Request

```json
{
  "model": "claude-3-7-sonnet-extended-thinking",
  "messages": [
    {
      "role": "user",
      "content": "Explain why the sky is blue. Step through your reasoning."
    }
  ],
  "reasoning_effort": "low",
  "temperature": 1.0,
  "max_tokens": 1000
}
```

#### Streaming Response Format

The streaming response contains several event types:

- `metadata`: Initial model information
- `thinking_start`: Indicates the beginning of the reasoning process
- `reasoning`: Contains chunks of the model's reasoning process
- `content`: The visible content chunks that form the final response
- `block_start` and `block_stop`: Indicate the beginning and end of content blocks
- `usage`: Final token usage statistics

#### Handling the Stream

Here's an example of how to handle the streaming response using JavaScript:

```javascript
const response = await fetch('https://api.example.com/v1/reason/completions/stream', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${apiKey}`
  },
  body: JSON.stringify({
    model: 'claude-3-7-sonnet-extended-thinking',
    messages: [{ role: 'user', content: 'Explain why the sky is blue. Step through your reasoning.' }],
    reasoning_effort: 'medium',
    temperature: 1.0
  })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

// Process the stream
while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  
  const chunk = decoder.decode(value);
  const lines = chunk.split('\n').filter(line => line.trim() !== '');
  
  for (const line of lines) {
    if (line.startsWith('event:')) {
      const eventType = line.replace('event:', '').trim();
      console.log(`Event type: ${eventType}`);
    } else if (line.startsWith('data:')) {
      const data = JSON.parse(line.replace('data:', '').trim());
      
      if (data.content) {
        // Handle content based on the event type
        console.log(`Content: ${data.content}`);
      }
    }
  }
}
```

## Python Client Example

Here's a complete example using Python to make a reasoning request:

```python
import requests
import json

API_KEY = "your_api_key"
API_URL = "https://api.example.com/v1/reason/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

payload = {
    "model": "claude-3-7-sonnet-extended-thinking",
    "messages": [
        {
            "role": "user",
            "content": "What is the sum of the first 10 prime numbers? Show your step-by-step reasoning."
        }
    ],
    "reasoning_effort": "medium",
    "temperature": 1.0,
    "max_tokens": 2000
}

response = requests.post(API_URL, headers=headers, json=payload)

if response.status_code == 200:
    result = response.json()
    print(f"Model: {result['model']}")
    print(f"Content:\n{result['content']}")
    print(f"\nToken usage:")
    print(f"  Input tokens: {result['usage']['input_tokens']}")
    print(f"  Output tokens: {result['usage']['output_tokens']}")
    print(f"  Reasoning tokens: {result['usage']['reasoning_tokens']}")
    print(f"  Total tokens: {result['usage']['total_tokens']}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
```

## Understanding Reasoning Effort Levels

The `reasoning_effort` parameter controls how much thinking the model puts into solving your problem:

- `low`: Uses fewer tokens for reasoning, resulting in a faster response but potentially less thorough analysis. Good for simpler questions.
- `medium` (default): Balanced approach with moderate reasoning depth.
- `high`: Allocates more tokens to the reasoning process, producing more thorough and detailed analysis. Best for complex problems requiring deep thinking.

## Token Usage and Billing

When using reasoning models, your token usage will include:

- `input_tokens`: The tokens in your prompt
- `output_tokens`: The tokens in the visible response
- `reasoning_tokens`: The tokens used in the thinking process
- `total_tokens`: The sum of all tokens used

The `reasoning_tokens` are billed as output tokens, so extended thinking will increase your overall token usage. 