"""
Integration tests for function calling with real provider APIs.

This test module makes real API calls to verify function calling works
with each provider. These tests will be skipped if API keys are not available.

Usage:
    - Set environment variables for the providers you want to test
    - Run with: python -m pytest testLib/test_function_registry_integration.py -v
"""

import unittest
import pytest
import os
import json
from serverRouter.core.function_registry import (
    function_registry, 
    ProviderType, 
    register_function,
    FunctionExecutionResult
)
from serverRouter.core.datamodels import (
    ChatCompletionRequest,
    ChatMessage,
    FunctionCall
)
from serverRouter.providers.openai.provider import OpenAIProvider
from serverRouter.providers.anthropic.provider import AnthropicProvider
from serverRouter.providers.gemini.provider import GeminiProvider
from serverRouter.providers.together.provider import TogetherAIProvider

# Function to check if API key is available
def has_api_key(env_var):
    return os.environ.get(env_var) is not None

# Register test functions
@register_function(
    name="get_weather",
    description="Get current weather in a given location",
    parameter_descriptions={
        "location": "The city and state, e.g., San Francisco, CA",
        "unit": "Temperature unit (celsius/fahrenheit)"
    }
)
def get_weather(location: str, unit: str = "celsius") -> dict:
    """Test weather function"""
    return {
        "location": location,
        "temperature": 22.5 if unit == "celsius" else 72.5,
        "unit": unit,
        "condition": "Sunny"
    }

@register_function(
    name="calculate",
    description="Perform a simple calculation",
    parameter_descriptions={
        "operation": "Mathematical operation to perform (add, subtract, multiply, divide)",
        "a": "First number",
        "b": "Second number"
    }
)
def calculate(operation: str, a: float, b: float) -> dict:
    """Simple calculator function"""
    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b
    else:
        raise ValueError(f"Unknown operation: {operation}")
    
    return {
        "operation": operation,
        "a": a,
        "b": b,
        "result": result
    }

# Base test case for provider testing
class ProviderFunctionCallingTestCase(unittest.TestCase):
    """Base class for provider-specific function calling tests"""
    
    async def run_function_calling_test(self, provider, model, prompt):
        """
        Run a function calling test with a specific provider
        
        Args:
            provider: The provider instance
            model: Model name to use
            prompt: Text prompt asking to use a function
        """
        # Create a request with functions
        request = ChatCompletionRequest(
            model=model,
            messages=[
                ChatMessage(role="user", content=prompt)
            ],
            temperature=0.7,
            max_tokens=1024,
            # Get all registered functions in provider-specific format
            functions=function_registry.get_for_provider(provider.provider_type)
        )
        
        # Call the provider
        response = await provider.chat_complete(request)
        
        # Verify a response was returned
        self.assertIsNotNone(response)
        
        # Check if function calls were detected
        self.assertIsNotNone(response.function_calls, 
                             f"No function calls detected from {provider.__class__.__name__}")
        self.assertGreater(len(response.function_calls), 0, 
                           f"Empty function calls list from {provider.__class__.__name__}")
        
        # Execute the function
        function_call = response.function_calls[0]
        result = function_registry.execute_function(
            function_call.name, 
            function_call.arguments
        )
        
        # Verify execution worked
        self.assertTrue(result.success, f"Function execution failed: {result.error}")
        self.assertIsNotNone(result.result)
        
        # Create a response with the function result
        function_response = await provider.create_function_response([result])
        
        # Verify function response is valid
        self.assertIsNotNone(function_response)
        
        return response, function_call, result, function_response

# Test cases for each provider

@pytest.mark.skipif(not has_api_key("OPENAI_API_KEY"), 
                    reason="OPENAI_API_KEY not set")
class TestOpenAIFunctionCalling(ProviderFunctionCallingTestCase):
    """Test function calling with OpenAI"""
    
    @pytest.mark.asyncio
    async def test_function_calling(self):
        provider = OpenAIProvider()
        response, function_call, result, function_response = await self.run_function_calling_test(
            provider=provider,
            model="gpt-3.5-turbo",
            prompt="What's the weather like in Seattle? Use the get_weather function."
        )
        
        # OpenAI-specific assertions
        self.assertEqual(function_call.name, "get_weather")
        self.assertIn("location", function_call.arguments)
        self.assertIsInstance(function_response, list)

@pytest.mark.skipif(not has_api_key("ANTHROPIC_API_KEY"), 
                    reason="ANTHROPIC_API_KEY not set")
class TestAnthropicFunctionCalling(ProviderFunctionCallingTestCase):
    """Test function calling with Anthropic"""
    
    @pytest.mark.asyncio
    async def test_function_calling(self):
        provider = AnthropicProvider()
        response, function_call, result, function_response = await self.run_function_calling_test(
            provider=provider,
            model="claude-3-haiku-20240307",
            prompt="What's the weather like in Miami? Use the get_weather function."
        )
        
        # Anthropic-specific assertions
        self.assertEqual(function_call.name, "get_weather")
        self.assertIn("location", function_call.arguments)
        self.assertEqual(function_response["role"], "user")
        self.assertIn("content", function_response)

@pytest.mark.skipif(not has_api_key("GEMINI_API_KEY"), 
                    reason="GEMINI_API_KEY not set")
class TestGeminiFunctionCalling(ProviderFunctionCallingTestCase):
    """Test function calling with Gemini"""
    
    @pytest.mark.asyncio
    async def test_function_calling(self):
        provider = GeminiProvider()
        response, function_call, result, function_response = await self.run_function_calling_test(
            provider=provider,
            model="gemini-1.5-pro-latest",
            prompt="Calculate 24 multiplied by 8. Use the calculate function."
        )
        
        # Gemini-specific assertions
        self.assertEqual(function_call.name, "calculate")
        self.assertIn("operation", function_call.arguments)
        self.assertIn("a", function_call.arguments)
        self.assertIn("b", function_call.arguments)
        self.assertEqual(function_response["role"], "function")
        self.assertIn("parts", function_response)

@pytest.mark.skipif(not has_api_key("TOGETHER_API_KEY"), 
                    reason="TOGETHER_API_KEY not set")
class TestTogetherFunctionCalling(ProviderFunctionCallingTestCase):
    """Test function calling with Together AI"""
    
    @pytest.mark.asyncio
    async def test_function_calling(self):
        provider = TogetherAIProvider()
        response, function_call, result, function_response = await self.run_function_calling_test(
            provider=provider,
            model="meta-llama/Llama-3.1-8B-Instruct-Function-Calling",  # Function-calling specific model
            prompt="What's the result of 15 divided by 3? Use the calculate function."
        )
        
        # Together AI-specific assertions
        self.assertEqual(function_call.name, "calculate")
        self.assertIn("operation", function_call.arguments)
        self.assertIsInstance(function_response, list)

if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 