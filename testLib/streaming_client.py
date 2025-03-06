"""
StreamingTest for testing the streaming API functionality.
This module provides utilities to test the streaming capabilities
of the OmniRouter API using the BaseTest framework.
"""

import time
import json
import sys
from datetime import datetime
from typing import Dict, Any, List

from .test_core import BaseTest
from .test_utils import test_logger

class StreamingTest(BaseTest):
    """Test class for the OmniRouter streaming API extending BaseTest."""
    
    def test_streaming(self, model: str, prompt: str, verbose: bool = False) -> Dict[str, Any]:
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
            # Using the client from BaseTest - note that the TestClient doesn't truly stream
            # This implementation processes the response as if it were streaming
            response = self.client.post(
                "/v1/chat/completions",
                json=payload
            )
            
            if response.status_code != 200:
                error_text = response.text
                test_logger.error(f"Error: {response.status_code} - {error_text}")
                result["error"] = f"HTTP {response.status_code}: {error_text}"
                return result
            
            # Process the response as if it were streaming
            first_token_received = False
            first_token_time = None
            received_provider = False
            
            if "data:" in response.text:
                for line in response.text.splitlines():
                    if line.startswith('data: ') and line != 'data: [DONE]':
                        data = line[6:].strip()
                        if not data:
                            continue
                            
                        try:
                            chunk = json.loads(data)
                            if "error" in chunk:
                                test_logger.error(f"Error: {chunk['error']['message']}")
                                result["error"] = chunk["error"]["message"]
                                break
                            
                            # Get the provider if available
                            if "provider" in chunk and chunk["provider"] and not received_provider:
                                result["provider"] = chunk["provider"]
                                received_provider = True
                            
                            # Record time of first token
                            if "content" in chunk and chunk["content"] and not first_token_received:
                                first_token_time = datetime.now()
                                first_token_received = True
                                
                            # Get content
                            content = chunk.get("content", "")
                            if content:
                                if verbose:
                                    sys.stdout.write(content)
                                    sys.stdout.flush()
                                complete_content += content
                            
                        except json.JSONDecodeError as e:
                            test_logger.error(f"Error parsing JSON: {data} - {str(e)}")
            
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
            
            # Print summary
            if verbose:
                print(f"\nModel: {model}")
                print(f"Provider: {result['provider'] or 'unknown'}")
                print(f"Total time: {total_time:.2f} seconds")
                if ttft:
                    print(f"Time to first token: {ttft:.2f} seconds")
                    
        except Exception as e:
            test_logger.error(f"Error: {str(e)}")
            result["error"] = str(e)
        
        return result

    def compare_models(self, models: List[str], prompt: str) -> Dict[str, Dict[str, Any]]:
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
            result = self.test_streaming(model, prompt)
            results[model] = result
            
            # Add a short delay between requests
            time.sleep(1)
        
        # Print comparison table
        print("\n===== Model Comparison =====")
        print(f"{'Model':<20} | {'Provider':<15} | {'TTFT (s)':<10} | {'Total Time (s)':<15} | {'Success':<7}")
        print("-" * 75)
        
        for model, result in results.items():
            ttft = f"{result['ttft']:.2f}" if result["ttft"] else "N/A"
            total_time = f"{result['total_time']:.2f}" if result["total_time"] else "N/A"
            print(f"{model:<20} | {result.get('provider', 'N/A'):<15} | {ttft:<10} | {total_time:<15} | {result['success']}")
        
        return results

    def test_all_providers(self, prompt: str = "Tell me a short story about a robot learning to feel emotions.") -> Dict[str, Dict[str, Any]]:
        """
        Test streaming for one model from each provider.
        
        Args:
            prompt: Prompt to send to each model
            
        Returns:
            Dictionary mapping provider names to test results
        """
        # Get available models and group by provider
        response = self.client.get("/v1/models/chat")
        models_data = response.json()["models"]
        
        # Group models by provider
        providers = {}
        for model in models_data:
            provider = model["provider"]
            if provider not in providers:
                providers[provider] = model["id"]
        
        # Get one model for each provider
        test_models = list(providers.values())
        test_logger.info(f"Testing one model from each provider: {test_models}")
        
        # Run the comparison
        return self.compare_models(test_models, prompt)


if __name__ == "__main__":
    """Example usage of the streaming test."""
    client = StreamingTest()
    client.setup_method()
    
    # Test with all providers
    client.test_all_providers()
    
    # Alternatively, test specific models
    models = ["gpt-3.5-turbo", "claude-3-5-sonnet", "gemini-2.0-pro"]
    results = client.compare_models(
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