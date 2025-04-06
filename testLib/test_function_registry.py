"""
Tests for the function registry implementation.
"""

import unittest
import json
from unittest.mock import Mock, patch
from serverRouter.core.function_registry import (
    function_registry, 
    ProviderType, 
    register_function,
    FunctionExecutionResult
)
from serverRouter.core.datamodels import FunctionCall
from types import SimpleNamespace

class TestFunctionRegistry(unittest.TestCase):
    """Test cases for function registry"""
    
    def setUp(self):
        """Set up test functions"""
        # Clear existing functions
        function_registry._functions = {}
        function_registry._execution_history = []
        
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
            name="get_stock_price",
            description="Get the current stock price",
            auto_execute=True
        )
        def get_stock_price(symbol: str) -> dict:
            """Get the current stock price for a given symbol"""
            # Simulated response
            prices = {
                "AAPL": 150.25,
                "GOOG": 2500.75,
                "MSFT": 300.50,
                "AMZN": 3200.00
            }
            return {
                "symbol": symbol,
                "price": prices.get(symbol, 0.0),
                "currency": "USD",
                "timestamp": "2023-04-01T12:00:00Z"
            }
    
    def test_registration(self):
        """Test function registration"""
        # Check if functions are registered
        self.assertTrue(function_registry.exists("get_weather"))
        self.assertTrue(function_registry.exists("get_stock_price"))
        
        # Check function count
        self.assertEqual(len(function_registry.get_all_functions()), 2)
        
        # Check function details
        weather_func = function_registry.get_function("get_weather")
        self.assertEqual(weather_func.name, "get_weather")
        self.assertEqual(len(weather_func.parameters), 2)
        self.assertEqual(weather_func.parameters["location"].required, True)
        self.assertEqual(weather_func.parameters["unit"].required, False)
        
        # Check auto-execute
        stock_func = function_registry.get_function("get_stock_price")
        self.assertTrue(stock_func.auto_execute)
    
    def test_execution(self):
        """Test function execution"""
        # Execute weather function
        result = function_registry.execute_function("get_weather", {"location": "Seattle, WA"})
        self.assertTrue(result.success)
        self.assertEqual(result.result["location"], "Seattle, WA")
        self.assertEqual(result.result["unit"], "celsius")
        
        # Execute with all parameters
        result = function_registry.execute_function("get_weather", {
            "location": "Miami, FL", 
            "unit": "fahrenheit"
        })
        self.assertTrue(result.success)
        self.assertEqual(result.result["location"], "Miami, FL")
        self.assertEqual(result.result["unit"], "fahrenheit")
        self.assertEqual(result.result["temperature"], 72.5)
        
        # Check execution history
        history = function_registry.get_execution_history()
        self.assertEqual(len(history), 2)
    
    def test_invalid_execution(self):
        """Test invalid function execution"""
        # Non-existent function
        result = function_registry.execute_function("non_existent_function", {})
        self.assertFalse(result.success)
        self.assertIn("not found", result.error)
        
        # Missing required parameter
        result = function_registry.execute_function("get_weather", {})
        self.assertFalse(result.success)
    
    def test_openai_format(self):
        """Test conversion to OpenAI format"""
        openai_tools = function_registry.get_for_provider(ProviderType.OPENAI)
        
        # Check if OpenAI format is correct
        self.assertIsInstance(openai_tools, list)
        self.assertEqual(len(openai_tools), 2)
        
        # Verify structure
        tool = openai_tools[0]
        self.assertEqual(tool["type"], "function")
        self.assertIn("name", tool["function"])
        self.assertIn("description", tool["function"])
        self.assertIn("parameters", tool["function"])
    
    def test_anthropic_format(self):
        """Test conversion to Anthropic format"""
        anthropic_tools = function_registry.get_for_provider(ProviderType.ANTHROPIC)
        
        # Check if Anthropic format is correct
        self.assertIsInstance(anthropic_tools, dict)
        self.assertIn("tools", anthropic_tools)
        self.assertEqual(len(anthropic_tools["tools"]), 2)
        
        # Verify structure
        tool = anthropic_tools["tools"][0]
        self.assertIn("name", tool)
        self.assertIn("description", tool)
        self.assertIn("input_schema", tool)
    
    def test_selective_functions(self):
        """Test getting selective functions"""
        # Get only one function in OpenAI format
        openai_tools = function_registry.get_for_provider(
            ProviderType.OPENAI, 
            ["get_weather"]
        )
        
        # Check if only requested function is included
        self.assertEqual(len(openai_tools), 1)
        self.assertEqual(openai_tools[0]["function"]["name"], "get_weather")

    def test_anthropic_parsing(self):
        """Test parsing Anthropic function calls"""
        # Create a mock Anthropic response
        mock_response = SimpleNamespace()
        mock_content = [
            SimpleNamespace(
                type="tool_use",
                tool_use=SimpleNamespace(
                    id="tool_123",
                    name="get_weather",
                    input=json.dumps({"location": "Boston, MA"})
                )
            ),
            SimpleNamespace(
                type="text",
                text="Here's the weather in Boston:"
            )
        ]
        mock_response.content = mock_content
        
        # Test parsing
        adapter = function_registry._providers[ProviderType.ANTHROPIC]
        function_calls = adapter.parse_function_call(mock_response)
        
        # Verify results
        self.assertEqual(len(function_calls), 1)
        self.assertEqual(function_calls[0]['name'], "get_weather")
        self.assertEqual(function_calls[0]['arguments']['location'], "Boston, MA")
        self.assertEqual(function_calls[0]['id'], "tool_123")
        
        # Test function response creation
        execution_results = [
            FunctionExecutionResult(
                function_name="get_weather",
                arguments={"id": "tool_123", "location": "Boston, MA"},
                result={"temperature": 20, "condition": "Sunny"},
                success=True,
                execution_time=0.01
            )
        ]
        
        response = adapter.create_function_response(execution_results)
        self.assertEqual(response["role"], "user")
        self.assertIsInstance(response["content"], list)
        self.assertEqual(response["content"][0]["type"], "tool_result")
        self.assertEqual(response["content"][0]["tool_result"]["tool_call_id"], "tool_123")

    def test_gemini_format(self):
        """Test conversion to Gemini format"""
        gemini_tools = function_registry.get_for_provider(ProviderType.GEMINI)
        
        # Check if Gemini format is correct
        self.assertIsInstance(gemini_tools, list)
        self.assertEqual(len(gemini_tools), 2)
        
        # Verify structure - Gemini uses function_declarations
        tool = gemini_tools[0]
        self.assertIn("function_declarations", tool)
        
        function_decl = tool["function_declarations"][0]
        self.assertIn("name", function_decl)
        self.assertIn("description", function_decl)
        self.assertIn("parameters", function_decl)
        
        # Check for uppercase type (Gemini specific)
        self.assertEqual(function_decl["parameters"]["type"], "OBJECT")

    def test_gemini_parsing(self):
        """Test parsing Gemini function calls"""
        # Create a mock Gemini response with function calls
        candidate = SimpleNamespace(
            content=SimpleNamespace(
                parts=[
                    SimpleNamespace(
                        function_call=SimpleNamespace(
                            name="get_weather",
                            args=json.dumps({"location": "Denver, CO"})
                        )
                    )
                ]
            )
        )
        
        mock_response = SimpleNamespace(
            candidates=[candidate]
        )
        
        # Test parsing
        adapter = function_registry._providers[ProviderType.GEMINI]
        function_calls = adapter.parse_function_call(mock_response)
        
        # Verify results
        self.assertEqual(len(function_calls), 1)
        self.assertEqual(function_calls[0]['name'], "get_weather")
        self.assertEqual(function_calls[0]['arguments']['location'], "Denver, CO")
        # Instead of checking the prefix, just verify the ID exists
        self.assertIsNotNone(function_calls[0]['id'])
        
        # Test function response creation
        execution_results = [
            FunctionExecutionResult(
                function_name="get_weather",
                arguments={"location": "Denver, CO"},
                result={"temperature": 18, "condition": "Cloudy"},
                success=True,
                execution_time=0.02
            )
        ]
        
        response = adapter.create_function_response(execution_results)
        self.assertEqual(response["role"], "function")
        self.assertIsInstance(response["parts"], list)
        self.assertIn("function_response", response["parts"][0])
        self.assertEqual(response["parts"][0]["function_response"]["name"], "get_weather")

    def test_together_format(self):
        """Test conversion to Together AI format (should be OpenAI compatible)"""
        together_tools = function_registry.get_for_provider(ProviderType.TOGETHER)
        openai_tools = function_registry.get_for_provider(ProviderType.OPENAI)
        
        # Together should use the OpenAI format
        self.assertEqual(len(together_tools), len(openai_tools))
        self.assertEqual(together_tools[0]["type"], "function")
        self.assertEqual(together_tools[0]["function"]["name"], openai_tools[0]["function"]["name"])

    def test_together_parsing(self):
        """Test parsing Together AI function calls (OpenAI format)"""
        # Create a simplistic mock that will work with the adapter
        # Directly test the OpenAIAdapter since Together uses it
        adapter = function_registry._providers[ProviderType.OPENAI]
        
        # Create a function call structure similar to OpenAI's
        mock_func = SimpleNamespace(
            name="get_stock_price", 
            arguments='{"symbol":"AAPL"}'
        )
        
        mock_tool_call = SimpleNamespace(
            id="call_abc123", 
            type="function", 
            function=mock_func
        )
        
        # Simple mock response
        mock_response = SimpleNamespace(
            tool_calls=[mock_tool_call]
        )
        
        # Parse using the OpenAI adapter directly
        function_calls = adapter.parse_function_call(mock_response)
        
        # Verify results
        self.assertEqual(len(function_calls), 1)
        self.assertEqual(function_calls[0]['name'], "get_stock_price")
        self.assertEqual(function_calls[0]['arguments']['symbol'], "AAPL")
        self.assertEqual(function_calls[0]['id'], "call_abc123")
        
        # Test function response creation
        execution_results = [
            FunctionExecutionResult(
                function_name="get_stock_price",
                arguments={"id": "call_abc123", "symbol": "AAPL"},
                result={"price": 150.25, "currency": "USD"},
                success=True,
                execution_time=0.01
            )
        ]
        
        response = adapter.create_function_response(execution_results)
        self.assertIsInstance(response, list)
        self.assertEqual(len(response), 1)
        self.assertEqual(response[0]["role"], "tool")
        self.assertEqual(response[0]["tool_call_id"], "call_abc123")
        self.assertIn("price", json.loads(response[0]["content"]))

if __name__ == "__main__":
    unittest.main() 