#!/usr/bin/env python3
"""
Test client for OmniRouter streaming API
"""

import asyncio
import aiohttp
import json
import argparse
import sys
import os
from datetime import datetime

# Configuration class instead of global variables
class Config:
    API_KEY = "test-sk1o83e"  # Default test key
    API_URL = "http://localhost:8000/v1/chat/completions"  # Default API URL

async def test_streaming(config, model: str, prompt: str, verbose: bool = False):
    """Test streaming from the OmniRouter API"""
    
    print(f"\n--- Testing streaming with model: {model} ---")
    print(f"Prompt: {prompt}")
    print("-" * 60)
    
    # Prepare the request payload
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": True,
        "temperature": 0.7
    }
    
    headers = {
        "Authorization": f"Bearer {config.API_KEY}",
        "Content-Type": "application/json"
    }
    
    start_time = datetime.now()
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                config.API_URL,
                json=payload,
                headers=headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"Error: {response.status} - {error_text}")
                    return
                
                # Process the streaming response
                complete_content = ""
                first_token_received = False
                first_token_time = None
                chunk = None
                
                async for line in response.content:
                    line = line.decode('utf-8')
                    if verbose:
                        print(f"Raw data: {line}")
                        
                    if line.startswith('data: '):
                        data = line[6:].strip()
                        if data == "[DONE]":
                            print("\n" + "-" * 60)
                            break
                            
                        try:
                            chunk = json.loads(data)
                            if "error" in chunk:
                                print(f"Error: {chunk['error']['message']}")
                                break
                            
                            # Record time of first token
                            if not first_token_received:
                                first_token_time = datetime.now()
                                first_token_received = True
                                
                            # Print the content chunk
                            content = chunk.get("content", "")
                            sys.stdout.write(content)
                            sys.stdout.flush()
                            
                            complete_content += content
                            
                        except json.JSONDecodeError:
                            print(f"Error parsing JSON: {data}")
                
                end_time = datetime.now()
                
                # Calculate timings
                total_time = (end_time - start_time).total_seconds()
                ttft = None
                if first_token_time:
                    ttft = (first_token_time - start_time).total_seconds()
                
                print(f"\nModel: {model}")
                if chunk:
                    print(f"Provider: {chunk.get('provider', 'unknown')}")
                print(f"Total time: {total_time:.2f} seconds")
                if ttft:
                    print(f"Time to first token: {ttft:.2f} seconds")
                print("-" * 60)
    
    except Exception as e:
        print(f"Error: {str(e)}")

async def main():
    config = Config()
    
    parser = argparse.ArgumentParser(description="Test OmniRouter streaming API")
    parser.add_argument("--model", "-m", type=str, default="gpt-3.5-turbo", 
                        help="Model to use for streaming (default: gpt-3.5-turbo)")
    parser.add_argument("--prompt", "-p", type=str, 
                        default="Tell me a short story about a robot learning to feel emotions.",
                        help="Prompt to send to the API")
    parser.add_argument("--api-key", "-k", type=str, default=config.API_KEY,
                        help="API key to use for authentication")
    parser.add_argument("--api-url", "-u", type=str, default=config.API_URL,
                        help="API URL to send requests to")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show verbose output including raw SSE data")
    
    args = parser.parse_args()
    
    # Update config based on args
    config.API_KEY = args.api_key
    config.API_URL = args.api_url
    
    await test_streaming(config, args.model, args.prompt, args.verbose)

if __name__ == "__main__":
    asyncio.run(main())