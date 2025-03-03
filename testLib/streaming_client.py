"""
Streaming Client for testing the streaming API functionality.
This module provides utilities to test the streaming capabilities
of the OmniRouter API.
"""

import asyncio
import aiohttp
import json
import sys
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

# Add parent directory to path to allow imports
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

# Initialize logger
from testLib.test_utils import test_logger

class StreamingClient:
    """Client for testing the OmniRouter streaming API."""
    
    def __init__(self, api_key: str = "test-sk1o83e", api_url: str = "http://localhost:8000/v1/chat/completions"):
        """
        Initialize the streaming client.
        
        Args:
            api_key: API key for authentication
            api_url: URL for the streaming API endpoint
        """
        self.api_key = api_key
        self.api_url = api_url
    
    async def test_streaming(self, model: str, prompt: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Test streaming from the OmniRouter API.
        
        Args:
            model: Model to use for streaming
            prompt: Prompt to send to the API
            verbose: Whether to show verbose output
            
        Returns:
            Dictionary with test results
        """
        test_logger.info(f"Testing streaming with model: {model}")
        test_logger.info(f"Prompt: {prompt}")
        
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
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        start_time = datetime.now()
        complete_content = ""
        result = {
            "model": model,
            "prompt": prompt,
            "content": "",
            "provider": None,
            "total_time": None,
            "ttft": None,  # Time to first token
            "success": False,
            "error": None
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        test_logger.error(f"Error: {response.status} - {error_text}")
                        result["error"] = f"HTTP {response.status}: {error_text}"
                        return result
                    
                    # Process the streaming response
                    first_token_received = False
                    first_token_time = None
                    chunk = None
                    
                    async for line in response.content:
                        line = line.decode('utf-8')
                        if verbose:
                            test_logger.debug(f"Raw data: {line}")
                            
                        if line.startswith('data: '):
                            data = line[6:].strip()
                            if data == "[DONE]":
                                break
                                
                            try:
                                chunk = json.loads(data)
                                if "error" in chunk:
                                    test_logger.error(f"Error: {chunk['error']['message']}")
                                    result["error"] = chunk["error"]["message"]
                                    break
                                
                                # Record time of first token
                                if not first_token_received:
                                    first_token_time = datetime.now()
                                    first_token_received = True
                                    
                                # Get content
                                content = chunk.get("content", "")
                                sys.stdout.write(content)
                                sys.stdout.flush()
                                
                                complete_content += content
                                
                            except json.JSONDecodeError:
                                test_logger.error(f"Error parsing JSON: {data}")
                    
                    end_time = datetime.now()
                    
                    # Calculate timings
                    total_time = (end_time - start_time).total_seconds()
                    ttft = None
                    if first_token_time:
                        ttft = (first_token_time - start_time).total_seconds()
                    
                    # Update result
                    result["content"] = complete_content
                    result["total_time"] = total_time
                    result["ttft"] = ttft
                    result["success"] = True
                    
                    if chunk:
                        result["provider"] = chunk.get("provider", "unknown")
                    
                    # Print summary
                    print(f"\nModel: {model}")
                    if chunk:
                        print(f"Provider: {chunk.get('provider', 'unknown')}")
                    print(f"Total time: {total_time:.2f} seconds")
                    if ttft:
                        print(f"Time to first token: {ttft:.2f} seconds")
        
        except Exception as e:
            test_logger.error(f"Error: {str(e)}")
            result["error"] = str(e)
        
        return result

    async def compare_models(self, models: List[str], prompt: str) -> Dict[str, Dict[str, Any]]:
        """
        Compare streaming performance across multiple models.
        
        Args:
            models: List of models to test
            prompt: Prompt to send to each model
            
        Returns:
            Dictionary mapping model names to test results
        """
        test_logger.info(f"Comparing streaming for models: {', '.join(models)}")
        test_logger.info(f"Prompt: {prompt}")
        
        results = {}
        
        for model in models:
            test_logger.info(f"Testing model: {model}")
            result = await self.test_streaming(model, prompt)
            results[model] = result
            
            # Add a short delay between requests
            await asyncio.sleep(1)
        
        # Print comparison table
        print("\n===== Model Comparison =====")
        print(f"{'Model':<20} | {'Provider':<15} | {'TTFT (s)':<10} | {'Total Time (s)':<15} | {'Success':<7}")
        print("-" * 75)
        
        for model, result in results.items():
            ttft = f"{result['ttft']:.2f}" if result["ttft"] else "N/A"
            total_time = f"{result['total_time']:.2f}" if result["total_time"] else "N/A"
            print(f"{model:<20} | {result.get('provider', 'N/A'):<15} | {ttft:<10} | {total_time:<15} | {result['success']}")
        
        return results


async def main():
    """Example usage of the streaming client."""
    client = StreamingClient()
    
    # Test with a single model
    result = await client.test_streaming(
        "gpt-3.5-turbo",
        "Tell me a short story about a robot learning to feel emotions.",
        verbose=True
    )
    
    # Compare multiple models
    models = ["gpt-3.5-turbo", "claude-3-5-sonnet", "gemini-2.0-pro"]
    results = await client.compare_models(
        models,
        "Explain the concept of machine learning in simple terms."
    )
    
    # Print detailed results
    print("\n===== Detailed Results =====")
    for model, model_result in results.items():
        if model_result["success"]:
            content_preview = model_result["content"][:100] + "..." if len(model_result["content"]) > 100 else model_result["content"]
            print(f"\nModel: {model}")
            print(f"Content preview: {content_preview}")
        else:
            print(f"\nModel: {model}")
            print(f"Error: {model_result['error']}")

if __name__ == "__main__":
    asyncio.run(main())