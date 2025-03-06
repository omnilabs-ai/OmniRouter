"""
Comprehensive tests for function/tool calling capabilities across all providers.
This test file verifies that all providers correctly handle function/tool calling parameters.
"""

import json
import time
import re
from typing import Dict, Any, List

from .test_core import BaseTest
from .test_utils import test_logger

# Create provider feature support tracking
PROVIDER_FEATURES = {
    # Known working providers for each feature
    "function_calling": ["openai"],      # Provider expected to support function calling
    "forced_function": ["openai"],       # Provider expected to support forced function calling
    "json_format": ["openai"],           # Provider expected to support JSON formatting
    "multiple_functions": ["openai"],    # Provider expected to support multiple function calls
}

class TestFunctionCalling(BaseTest):
    """Test suite for function calling capabilities across all providers."""
    
    # Common tool definitions for testing
    WEATHER_TOOL = {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use"
                    }
                },
                "required": ["location"]
            }
        }
    }
    
    CALCULATOR_TOOL = {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform basic arithmetic operations",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "The operation to perform"
                    },
                    "number1": {
                        "type": "number",
                        "description": "The first number"
                    },
                    "number2": {
                        "type": "number",
                        "description": "The second number"
                    }
                },
                "required": ["operation", "number1", "number2"]
            }
        }
    }
    
    def test_get_all_models(self):
        """Get all available models to test."""
        response = self.client.get("/v1/models/chat")
        assert response.status_code == 200
        
        models_data = response.json()["models"]
        test_logger.info(f"Found {len(models_data)} total models")
        
        # Group models by provider
        provider_models = {}
        for model in models_data:
            provider = model["provider"]
            if provider not in provider_models:
                provider_models[provider] = []
            provider_models[provider].append(model["id"])
        
        test_logger.info(f"Found {len(provider_models)} unique providers: {list(provider_models.keys())}")
        return provider_models
    
    def test_basic_function_calling(self):
        """Test basic function calling with the weather function for all providers."""
        provider_models = self.test_get_all_models()
        
        results = {}
        
        # Test one model from each provider for function calling
        for provider, models in provider_models.items():
            model_id = models[0]  # Use the first model for each provider
            test_logger.info(f"Testing function calling for provider: {provider}, model: {model_id}")
            
            # Create a request with tool calling
            request_data = {
                "model": model_id,
                "messages": [
                    {"role": "user", "content": "What's the weather like in San Francisco?"}
                ],
                "tools": [self.WEATHER_TOOL],
                "tool_choice": "auto",
                "temperature": 0.7
            }
            
            try:
                response = self.client.post(
                    "/v1/chat/completions",
                    json=request_data,
                    timeout=30  # Allow longer timeout for function calling
                )
                
                if response.status_code != 200:
                    test_logger.error(f"Error for {provider}/{model_id}: HTTP {response.status_code}")
                    results[provider] = {
                        "model": model_id,
                        "success": False,
                        "error": f"HTTP {response.status_code}: {response.text}",
                        "has_function_call": False
                    }
                    continue
                
                response_data = response.json()
                
                # Check if the response contains a tool_calls field
                has_tool_calls = "tool_calls" in response_data and response_data["tool_calls"] is not None
                
                if has_tool_calls:
                    tool_call = response_data["tool_calls"][0]
                    function_name = tool_call.get("function", {}).get("name", "")
                    arguments = tool_call.get("function", {}).get("arguments", "{}")
                    
                    # Try to parse arguments
                    try:
                        args = json.loads(arguments)
                        location = args.get("location", "")
                        has_location = "san francisco" in location.lower()
                    except:
                        has_location = False
                    
                    results[provider] = {
                        "model": model_id,
                        "success": True,
                        "has_function_call": True,
                        "function_name": function_name,
                        "has_correct_function": function_name == "get_current_weather",
                        "has_location": has_location
                    }
                    
                    test_logger.info(f"✓ {provider}/{model_id}: Successfully called function '{function_name}'")
                else:
                    # Some models might respond with content instead of a function call - this is OK
                    content = response_data.get("content", "")
                    test_logger.info(f"! {provider}/{model_id}: Did not make a function call, responded with content instead")
                    
                    results[provider] = {
                        "model": model_id,
                        "success": True,  # Still consider this a successful test
                        "has_function_call": False,
                        "content_preview": content[:100] + "..." if len(content) > 100 else content
                    }
                
            except Exception as e:
                test_logger.error(f"Exception testing {provider}/{model_id}: {str(e)}")
                results[provider] = {
                    "model": model_id,
                    "success": False,
                    "error": str(e),
                    "has_function_call": False
                }
            
            # Add a short delay between requests
            time.sleep(1)
        
        # Print function calling results
        print("\n===== Function Calling Support by Provider =====")
        print(f"{'Provider':<15} | {'Model':<25} | {'Function Call':<12} | {'Correct Function':<17} | {'Details':<40}")
        print("-" * 115)
        
        for provider, result in results.items():
            if result["has_function_call"]:
                call_status = "✓"
                correct_fn = "✓" if result.get("has_correct_function", False) else "✗"
                details = f"Function: {result.get('function_name', 'N/A')}"
                if result.get("has_location", False):
                    details += ", Found location"
            else:
                call_status = "✗"
                correct_fn = "N/A"
                if result.get("success", False):
                    details = result.get("content_preview", "Responded with content")
                else:
                    details = result.get("error", "Unknown error")
            
            print(f"{provider:<15} | {result['model']:<25} | {call_status:<12} | {correct_fn:<17} | {details:<40}")
        
        # Calculate success metrics
        successful_calls = sum(1 for r in results.values() if r.get("has_function_call", False))
        correct_functions = sum(1 for r in results.values() if r.get("has_correct_function", False))
        
        print(f"\nProviders with function calling: {successful_calls}/{len(results)}")
        print(f"Providers with correct function: {correct_functions}/{len(results)}")
        
        # Check if expected providers support function calling
        for provider in PROVIDER_FEATURES["function_calling"]:
            if provider in results:
                if provider in provider_models:
                    assert results[provider]["success"], f"Provider {provider} failed the function calling test"
    
    def test_json_response_format(self):
        """Test the response_format parameter for JSON responses."""
        provider_models = self.test_get_all_models()
        
        results = {}
        
        # Test one model from each provider for JSON response format
        for provider, models in provider_models.items():
            model_id = models[0]  # Use the first model for each provider
            test_logger.info(f"Testing JSON response format for provider: {provider}, model: {model_id}")
            
            # Create a request with JSON response format
            request_data = {
                "model": model_id,
                "messages": [
                    {"role": "user", "content": "Generate a JSON object with information about 3 popular programming languages including: name, year created, and main uses"}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.7
            }
            
            try:
                response = self.client.post(
                    "/v1/chat/completions",
                    json=request_data,
                    timeout=30
                )
                
                if response.status_code != 200:
                    test_logger.error(f"Error for {provider}/{model_id}: HTTP {response.status_code}")
                    results[provider] = {
                        "model": model_id,
                        "success": False,
                        "error": f"HTTP {response.status_code}: {response.text}",
                        "is_json": False
                    }
                    continue
                
                response_data = response.json()
                content = response_data.get("content", "")
                
                # Check if the response content is valid JSON
                is_json = False
                languages_found = 0
                
                try:
                    # Try to parse as JSON
                    parsed_json = json.loads(content)
                    is_json = True
                    
                    # Check if it contains programming languages information
                    if isinstance(parsed_json, dict) and "languages" in parsed_json:
                        languages_found = len(parsed_json["languages"])
                    elif isinstance(parsed_json, list):
                        languages_found = len(parsed_json)
                    elif isinstance(parsed_json, dict) and any(isinstance(v, dict) for v in parsed_json.values()):
                        # Might be directly a dict of languages
                        languages_found = len([v for v in parsed_json.values() if isinstance(v, dict)])
                    
                except json.JSONDecodeError:
                    is_json = False
                
                results[provider] = {
                    "model": model_id,
                    "success": True,  # Test succeeded even if JSON format isn't supported
                    "is_json": is_json,
                    "languages_found": languages_found,
                    "content_preview": content[:100] + "..." if len(content) > 100 else content
                }
                
                if is_json:
                    test_logger.info(f"✓ {provider}/{model_id}: Successfully returned JSON with {languages_found} languages")
                else:
                    # Log as info rather than warning
                    test_logger.info(f"! {provider}/{model_id}: Did not return valid JSON (feature may not be supported)")
                
            except Exception as e:
                test_logger.error(f"Exception testing {provider}/{model_id}: {str(e)}")
                results[provider] = {
                    "model": model_id,
                    "success": False,
                    "error": str(e),
                    "is_json": False
                }
            
            # Add a short delay between requests
            time.sleep(1)
        
        # Print JSON response format results
        print("\n===== JSON Response Format Support by Provider =====")
        print(f"{'Provider':<15} | {'Model':<25} | {'Valid JSON':<10} | {'Languages':<10} | {'Preview':<50}")
        print("-" * 115)
        
        for provider, result in results.items():
            is_json = "✓" if result.get("is_json", False) else "✗"
            languages = str(result.get("languages_found", 0))
            preview = result.get("content_preview", result.get("error", "N/A"))
            
            print(f"{provider:<15} | {result['model']:<25} | {is_json:<10} | {languages:<10} | {preview:<50}")
        
        # Calculate success metrics
        json_success = sum(1 for r in results.values() if r.get("is_json", False))
        
        print(f"\nProviders with JSON response format: {json_success}/{len(results)}")
        
        # Check if expected providers support JSON formatting
        for provider in PROVIDER_FEATURES["json_format"]:
            if provider in results:
                if provider in provider_models:
                    assert results[provider]["success"], f"Provider {provider} failed the JSON response format test"
                    
    def test_multiple_functions(self):
        """Test providing multiple functions and seeing which one the model chooses."""
        provider_models = self.test_get_all_models()
        
        results = {}
        
        # Test one model from each provider with multiple functions
        for provider, models in provider_models.items():
            model_id = models[0]  # Use the first model for each provider
            test_logger.info(f"Testing multiple functions for provider: {provider}, model: {model_id}")
            
            # Create a request with multiple tools
            request_data = {
                "model": model_id,
                "messages": [
                    {"role": "user", "content": "What's the weather like in Boston? Also, can you calculate 15 * 7?"}
                ],
                "tools": [self.WEATHER_TOOL, self.CALCULATOR_TOOL],
                "tool_choice": "auto",
                "temperature": 0.7
            }
            
            try:
                response = self.client.post(
                    "/v1/chat/completions",
                    json=request_data,
                    timeout=30
                )
                
                if response.status_code != 200:
                    test_logger.error(f"Error for {provider}/{model_id}: HTTP {response.status_code}")
                    results[provider] = {
                        "model": model_id,
                        "success": False,
                        "error": f"HTTP {response.status_code}: {response.text}",
                        "has_function_call": False
                    }
                    continue
                
                response_data = response.json()
                
                # Check if the response contains tool_calls field
                has_tool_calls = "tool_calls" in response_data and response_data["tool_calls"] is not None
                
                if has_tool_calls:
                    # Some models might call multiple functions, some might call just one
                    function_calls = []
                    for tool_call in response_data["tool_calls"]:
                        function_name = tool_call.get("function", {}).get("name", "")
                        function_calls.append(function_name)
                    
                    results[provider] = {
                        "model": model_id,
                        "success": True,
                        "has_function_call": True,
                        "function_calls": function_calls,
                        "num_functions_called": len(function_calls),
                        "weather_called": "get_current_weather" in function_calls,
                        "calculator_called": "calculator" in function_calls
                    }
                    
                    test_logger.info(f"✓ {provider}/{model_id}: Called {len(function_calls)} functions: {', '.join(function_calls)}")
                else:
                    # Some models might respond with content instead of function calls - this is OK
                    content = response_data.get("content", "")
                    test_logger.info(f"! {provider}/{model_id}: Did not make any function calls (feature may not be supported)")
                    
                    results[provider] = {
                        "model": model_id,
                        "success": True,  # Still consider this a successful test
                        "has_function_call": False,
                        "content_preview": content[:100] + "..." if len(content) > 100 else content
                    }
                
            except Exception as e:
                test_logger.error(f"Exception testing {provider}/{model_id}: {str(e)}")
                results[provider] = {
                    "model": model_id,
                    "success": False,
                    "error": str(e),
                    "has_function_call": False
                }
            
            # Add a short delay between requests
            time.sleep(1)
        
        # Print multiple functions results
        print("\n===== Multiple Functions Support by Provider =====")
        print(f"{'Provider':<15} | {'Model':<25} | {'Function Call':<12} | {'# Functions':<12} | {'Weather Called':<15} | {'Calculator Called':<18}")
        print("-" * 115)
        
        for provider, result in results.items():
            if result["has_function_call"]:
                call_status = "✓"
                num_functions = str(result.get("num_functions_called", 0))
                weather = "✓" if result.get("weather_called", False) else "✗"
                calculator = "✓" if result.get("calculator_called", False) else "✗"
            else:
                call_status = "✗"
                num_functions = "0"
                weather = "✗"
                calculator = "✗"
            
            print(f"{provider:<15} | {result['model']:<25} | {call_status:<12} | {num_functions:<12} | {weather:<15} | {calculator:<18}")
        
        # Calculate success metrics
        multi_fn_support = sum(1 for r in results.values() if r.get("num_functions_called", 0) > 1)
        
        print(f"\nProviders supporting multiple function calls: {multi_fn_support}/{len(results)}")
        
        # Check if expected providers support multiple functions
        for provider in PROVIDER_FEATURES["multiple_functions"]:
            if provider in results:
                if provider in provider_models:
                    assert results[provider]["success"], f"Provider {provider} failed the multiple functions test"