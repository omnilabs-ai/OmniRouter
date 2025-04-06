"""Universal Function Registry for Omni Labs AI.

This module provides a unified interface for registering functions that can be used
with different LLM providers (OpenAI, Anthropic, Gemini, Together AI, etc.).

Key features:
- Register functions once, use them with any provider
- Automatic conversion between provider-specific formats
- Type hint inspection for automatic parameter schema generation
- Support for parallel function calling where available
- Validation of function arguments before execution
- Comprehensive logging and debugging
"""

from typing import Any, Callable, Dict, List, Optional, Type, Union, get_type_hints
import inspect
import json
import enum
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pydantic import BaseModel, Field, create_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("omni.function_registry")

class ParameterType(str, enum.Enum):
    """Supported parameter types across providers."""
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    NULL = "null"
    ANY = "any"

class ProviderType(str, enum.Enum):
    """Supported providers for function calling."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    TOGETHER = "together"
    GENERIC = "generic"

class FunctionParameter(BaseModel):
    """Definition of a function parameter."""
    name: str
    type: ParameterType
    description: Optional[str] = None
    required: bool = True
    enum: Optional[List[Any]] = None
    default: Optional[Any] = None
    items: Optional[Dict[str, Any]] = None
    properties: Optional[Dict[str, Any]] = None

class FunctionDefinition(BaseModel):
    """Definition of a function that can be exposed to LLMs."""
    name: str
    description: str
    parameters: Dict[str, FunctionParameter]
    implementation: Optional[Callable] = None
    auto_execute: bool = False

    class Config:
        arbitrary_types_allowed = True

class FunctionExecutionResult(BaseModel):
    """Result of a function execution."""
    function_name: str
    arguments: Dict[str, Any]
    result: Any
    success: bool
    error: Optional[str] = None
    execution_time: float

class ToolChoice(str, enum.Enum):
    """Tool choice options for controlling function calling behavior."""
    AUTO = "auto"
    REQUIRED = "required"
    NONE = "none"

class ProviderAdapter(ABC):
    """Base adapter interface for provider-specific function calling."""
    
    @abstractmethod
    def convert_functions(self, functions: List[FunctionDefinition]) -> Any:
        """Convert function definitions to provider-specific format."""
        pass
    
    @abstractmethod
    def parse_function_call(self, response: Any) -> List[Dict[str, Any]]:
        """Parse function call from provider response."""
        pass
    
    @abstractmethod
    def create_function_response(self, function_results: List[FunctionExecutionResult]) -> Any:
        """Create provider-specific response with function results."""
        pass

class OpenAIAdapter(ProviderAdapter):
    """Adapter for OpenAI's function calling API."""
    
    def convert_functions(self, functions: List[FunctionDefinition]) -> List[Dict[str, Any]]:
        """Convert function definitions to OpenAI tools format."""
        tools = []
        
        for func in functions:
            # Build OpenAI parameter schema
            properties = {}
            required = []
            
            for param_name, param in func.parameters.items():
                prop = {
                    "type": param.type.value,
                    "description": param.description or f"Parameter {param_name}"
                }
                
                if param.enum:
                    prop["enum"] = param.enum
                
                if param.items:
                    prop["items"] = param.items
                
                if param.properties:
                    prop["properties"] = param.properties
                
                properties[param_name] = prop
                
                if param.required:
                    required.append(param_name)
            
            # Create OpenAI function schema
            function_schema = {
                "name": func.name,
                "description": func.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
            
            tools.append({
                "type": "function",
                "function": function_schema
            })
        
        return tools
    
    def parse_function_call(self, response: Any) -> List[Dict[str, Any]]:
        """Parse function calls from OpenAI response."""
        function_calls = []
        
        # Check for modern tool_calls format
        if hasattr(response, 'tool_calls') and response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call.type == 'function':
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                        function_calls.append({
                            'name': tool_call.function.name,
                            'arguments': arguments,
                            'id': tool_call.id  # Preserve the ID for response
                        })
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse arguments: {tool_call.function.arguments}")
        
        # Check for legacy function_call format (older models)
        elif hasattr(response, 'function_call') and response.function_call:
            try:
                arguments = json.loads(response.function_call.arguments)
                function_calls.append({
                    'name': response.function_call.name,
                    'arguments': arguments,
                    'id': 'function_call_0'  # Assign a default ID for backward compatibility
                })
            except json.JSONDecodeError:
                logger.error(f"Failed to parse arguments: {response.function_call.arguments}")
        
        return function_calls
    
    def create_function_response(self, function_results: List[FunctionExecutionResult]) -> List[Dict[str, Any]]:
        """Create response with function results in OpenAI format."""
        messages = []
        
        for result in function_results:
            message = {
                "role": "tool",
                "tool_call_id": result.arguments.get("id", "function_call_0"),
                "content": json.dumps(result.result) if result.success else f"Error: {result.error}"
            }
            messages.append(message)
        
        return messages

class AnthropicAdapter(ProviderAdapter):
    """Adapter for Anthropic's tool use API."""
    
    def convert_functions(self, functions: List[FunctionDefinition]) -> Dict[str, Any]:
        """Convert function definitions to Anthropic tools format."""
        tools = []
        
        for func in functions:
            # Build Anthropic parameter schema
            properties = {}
            required = []
            
            for param_name, param in func.parameters.items():
                prop = {
                    "type": param.type.value,
                    "description": param.description or f"Parameter {param_name}"
                }
                
                if param.enum:
                    prop["enum"] = param.enum
                
                if param.items:
                    prop["items"] = param.items
                
                if param.properties:
                    prop["properties"] = param.properties
                
                properties[param_name] = prop
                
                if param.required:
                    required.append(param_name)
            
            # Create Anthropic tool schema
            tools.append({
                "name": func.name,
                "description": func.description,
                "input_schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            })
        
        return {"tools": tools}
    
    def parse_function_call(self, response: Any) -> List[Dict[str, Any]]:
        """Parse function calls from Anthropic response."""
        function_calls = []
        
        # Check for tool_use in content blocks
        if hasattr(response, 'content') and response.content:
            for block in response.content:
                if block.type == 'tool_use':
                    try:
                        tool_use = block.tool_use
                        arguments = json.loads(tool_use.input) if isinstance(tool_use.input, str) else tool_use.input
                        function_calls.append({
                            'name': tool_use.name,
                            'arguments': arguments,
                            'id': tool_use.id  # Preserve the ID for response
                        })
                    except (json.JSONDecodeError, AttributeError) as e:
                        logger.error(f"Failed to parse Anthropic tool use: {e}")
        
        return function_calls
    
    def create_function_response(self, function_results: List[FunctionExecutionResult]) -> Dict[str, Any]:
        """Create response with function results in Anthropic format."""
        content_blocks = []
        
        for result in function_results:
            content = {
                "type": "tool_result",
                "tool_result": {
                    "tool_call_id": result.arguments.get("id", ""),
                    "content": json.dumps(result.result) if result.success else f"Error: {result.error}"
                }
            }
            content_blocks.append(content)
        
        return {"role": "user", "content": content_blocks}

class GeminiAdapter(ProviderAdapter):
    """Adapter for Google's Gemini function calling API."""
    
    def convert_functions(self, functions: List[FunctionDefinition]) -> List[Dict[str, Any]]:
        """Convert function definitions to Gemini tools format."""
        tools = []
        
        for func in functions:
            # Build Gemini parameter schema
            properties = {}
            required = []
            
            for param_name, param in func.parameters.items():
                # Convert parameter type to Gemini format (uppercase)
                param_type = param.type.value.upper()
                
                prop = {
                    "type": param_type,
                    "description": param.description or f"Parameter {param_name}"
                }
                
                if param.enum:
                    prop["enum"] = param.enum
                
                properties[param_name] = prop
                
                if param.required:
                    required.append(param_name)
            
            # Create Gemini function schema
            function_declaration = {
                "name": func.name,
                "description": func.description,
                "parameters": {
                    "type": "OBJECT",
                    "properties": properties
                }
            }
            
            if required:
                function_declaration["parameters"]["required"] = required
            
            tools.append({
                "function_declarations": [function_declaration]
            })
        
        return tools
    
    def parse_function_call(self, response: Any) -> List[Dict[str, Any]]:
        """Parse function calls from Gemini response."""
        function_calls = []
        
        # Gemini's function calling structure varies between SDK versions
        if hasattr(response, 'candidates') and response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, 'content') and candidate.content:
                    for part in candidate.content.parts:
                        if hasattr(part, 'function_call'):
                            try:
                                function_call = part.function_call
                                arguments = json.loads(function_call.args) if isinstance(function_call.args, str) else function_call.args
                                function_calls.append({
                                    'name': function_call.name,
                                    'arguments': arguments,
                                    'id': f"function_call_{len(function_calls)}"  # Generate an ID
                                })
                            except (json.JSONDecodeError, AttributeError) as e:
                                logger.error(f"Failed to parse Gemini function call: {e}")
        
        # Try alternate response format
        elif hasattr(response, 'function_call'):
            try:
                arguments = json.loads(response.function_call.args) if isinstance(response.function_call.args, str) else response.function_call.args
                function_calls.append({
                    'name': response.function_call.name,
                    'arguments': arguments,
                    'id': 'function_call_0'
                })
            except (json.JSONDecodeError, AttributeError) as e:
                logger.error(f"Failed to parse Gemini function call: {e}")
        
        return function_calls
    
    def create_function_response(self, function_results: List[FunctionExecutionResult]) -> Dict[str, Any]:
        """Create response with function results in Gemini format."""
        parts = []
        
        for result in function_results:
            part = {
                "function_response": {
                    "name": result.function_name,
                    "response": {
                        "result": json.dumps(result.result) if result.success else f"Error: {result.error}"
                    }
                }
            }
            parts.append(part)
        
        return {"role": "function", "parts": parts}

class TogetherAdapter(ProviderAdapter):
    """Adapter for Together AI's function calling API (OpenAI compatible)."""
    
    def convert_functions(self, functions: List[FunctionDefinition]) -> List[Dict[str, Any]]:
        """Convert function definitions to Together AI tools format."""
        # Together AI uses the OpenAI format
        openai_adapter = OpenAIAdapter()
        return openai_adapter.convert_functions(functions)
    
    def parse_function_call(self, response: Any) -> List[Dict[str, Any]]:
        """Parse function calls from Together AI response."""
        # Together AI uses the OpenAI format
        openai_adapter = OpenAIAdapter()
        return openai_adapter.parse_function_call(response)
    
    def create_function_response(self, function_results: List[FunctionExecutionResult]) -> List[Dict[str, Any]]:
        """Create response with function results in Together AI format."""
        # Together AI uses the OpenAI format
        openai_adapter = OpenAIAdapter()
        return openai_adapter.create_function_response(function_results)

class FunctionRegistry:
    """
    Universal Function Registry for Omni Labs AI.
    
    This registry manages functions that can be called by LLMs via different providers.
    Functions are registered once and can be used with any supported provider.
    """
    
    _instance = None  # Singleton instance
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FunctionRegistry, cls).__new__(cls)
            cls._instance._functions = {}
            cls._instance._providers = {
                ProviderType.OPENAI: OpenAIAdapter(),
                ProviderType.ANTHROPIC: AnthropicAdapter(),
                ProviderType.GEMINI: GeminiAdapter(),
                ProviderType.TOGETHER: TogetherAdapter()
            }
            cls._instance._execution_history = []
        return cls._instance
    
    def register(self, func=None, *, name=None, description=None, parameter_descriptions=None, auto_execute=False):
        """
        Register a function with the registry.
        
        Can be used as a decorator or a direct function call.
        
        Args:
            func (callable, optional): The function to register
            name (str, optional): Override the function name
            description (str, optional): Description of what the function does
            parameter_descriptions (dict, optional): Descriptions for parameters
            auto_execute (bool, optional): Whether to auto-execute this function
                when called by an LLM
            
        Returns:
            callable: The original function or a decorator
        """
        if func is None:
            # Used as a decorator with parameters
            return lambda f: self.register(
                f, 
                name=name, 
                description=description, 
                parameter_descriptions=parameter_descriptions,
                auto_execute=auto_execute
            )
            
        func_name = name or func.__name__
        func_doc = description or inspect.getdoc(func) or f"Execute {func_name}"
        param_desc = parameter_descriptions or {}
        
        # Inspect function signature to extract parameters
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)
        
        parameters = {}
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self' or param_name == 'cls':
                continue
                
            # Get parameter type
            param_type = type_hints.get(param_name, Any)
            param_type_str = self._get_parameter_type(param_type)
            
            # Check if parameter has a default value
            has_default = param.default is not param.empty
            default_value = param.default if has_default else None
            
            # Create parameter schema
            parameters[param_name] = FunctionParameter(
                name=param_name,
                type=param_type_str,
                description=param_desc.get(param_name, f"Parameter {param_name}"),
                required=not has_default,
                default=default_value
            )
                
            # Handle enum types
            if hasattr(param_type, '__origin__') and param_type.__origin__ is Union:
                # For Optional types (Union[Type, None])
                if type(None) in param_type.__args__:
                    non_none_types = [t for t in param_type.__args__ if t is not type(None)]
                    if len(non_none_types) == 1:
                        parameters[param_name].type = self._get_parameter_type(non_none_types[0])
                        parameters[param_name].required = False
            
            # Handle array types
            if param_type_str == ParameterType.ARRAY:
                if hasattr(param_type, '__origin__') and param_type.__origin__ is list:
                    if hasattr(param_type, '__args__') and param_type.__args__:
                        item_type = param_type.__args__[0]
                        parameters[param_name].items = {
                            "type": self._get_parameter_type(item_type).value
                        }
            
            # Handle enum classes for restricted values
            if isinstance(param_type, type) and issubclass(param_type, enum.Enum):
                parameters[param_name].enum = [e.value for e in param_type]
        
        # Create and store function definition
        function_def = FunctionDefinition(
            name=func_name,
            description=func_doc,
            parameters=parameters,
            implementation=func,
            auto_execute=auto_execute
        )
        
        self._functions[func_name] = function_def
        logger.info(f"Registered function: {func_name}")
        
        return func
    
    def _get_parameter_type(self, type_hint):
        """Convert Python type hints to parameter types."""
        type_map = {
            str: ParameterType.STRING,
            int: ParameterType.INTEGER,
            float: ParameterType.NUMBER,
            bool: ParameterType.BOOLEAN,
            list: ParameterType.ARRAY,
            dict: ParameterType.OBJECT,
            type(None): ParameterType.NULL,
            Any: ParameterType.ANY,
        }
        
        if type_hint in type_map:
            return type_map[type_hint]
        
        # Handle more complex types (Union, List, etc.)
        if hasattr(type_hint, '__origin__'):
            if type_hint.__origin__ is list:
                return ParameterType.ARRAY
            elif type_hint.__origin__ is dict:
                return ParameterType.OBJECT
            elif type_hint.__origin__ is Union:
                # For Optional types (Union[Type, None])
                if type(None) in type_hint.__args__:
                    non_none_types = [t for t in type_hint.__args__ if t is not type(None)]
                    if len(non_none_types) == 1:
                        return self._get_parameter_type(non_none_types[0])
                # Default for Union - use string (could be improved)
                return ParameterType.STRING
        
        # Default fallback
        return ParameterType.STRING
    
    def get_function(self, name: str) -> Optional[FunctionDefinition]:
        """Get a function definition by name."""
        return self._functions.get(name)
    
    def get_all_functions(self) -> List[FunctionDefinition]:
        """Get all registered function definitions."""
        return list(self._functions.values())
    
    def exists(self, name: str) -> bool:
        """Check if a function exists in the registry."""
        return name in self._functions
    
    def get_for_provider(self, provider: ProviderType, functions=None) -> Any:
        """
        Convert function definitions to provider-specific format.
        
        Args:
            provider (ProviderType): The provider to convert for
            functions (list, optional): List of function names to convert.
                If None, all functions are converted.
                
        Returns:
            Any: Functions in provider-specific format
        """
        # Get the specified adapter
        adapter = self._providers.get(provider)
        if not adapter:
            raise ValueError(f"Unsupported provider: {provider}")
        
        # Filter functions if specified
        func_defs = []
        if functions:
            for name in functions:
                if name in self._functions:
                    func_defs.append(self._functions[name])
                else:
                    logger.warning(f"Function not found: {name}")
        else:
            func_defs = list(self._functions.values())
        
        # Convert functions to provider format
        return adapter.convert_functions(func_defs)
    
    def parse_function_calls(self, provider: ProviderType, response: Any) -> List[Dict[str, Any]]:
        """
        Parse function calls from a provider's response.
        
        Args:
            provider (ProviderType): The provider that generated the response
            response (Any): The response from the provider
            
        Returns:
            List[Dict[str, Any]]: List of function calls with name, arguments and ID
        """
        adapter = self._providers.get(provider)
        if not adapter:
            raise ValueError(f"Unsupported provider: {provider}")
        
        return adapter.parse_function_call(response)
    
    def create_function_response(self, provider: ProviderType, function_results: List[FunctionExecutionResult]) -> Any:
        """
        Create a provider-specific response with function results.
        
        Args:
            provider (ProviderType): The provider to format the response for
            function_results (List[FunctionExecutionResult]): Results of function executions
            
        Returns:
            Any: Response in provider-specific format
        """
        adapter = self._providers.get(provider)
        if not adapter:
            raise ValueError(f"Unsupported provider: {provider}")
        
        return adapter.create_function_response(function_results)
    
    def execute_function(self, function_name: str, arguments: Dict[str, Any]) -> FunctionExecutionResult:
        """
        Execute a registered function with the given arguments.
        
        Args:
            function_name (str): Name of the function to execute
            arguments (dict): Arguments to pass to the function
            
        Returns:
            FunctionExecutionResult: Result of the function execution
            
        Raises:
            ValueError: If the function doesn't exist
        """
        if function_name not in self._functions:
            return FunctionExecutionResult(
                function_name=function_name,
                arguments=arguments,
                result=None,
                success=False,
                error=f"Function '{function_name}' not found in registry",
                execution_time=0.0
            )
        
        function_def = self._functions[function_name]
        
        # Start timer for execution performance
        start_time = datetime.now()
        
        try:
            # Filter arguments to only include those expected by the function
            filtered_args = {}
            sig = inspect.signature(function_def.implementation)
            
            for param_name in sig.parameters:
                if param_name == 'self' or param_name == 'cls':
                    continue
                
                # If argument is provided, use it
                if param_name in arguments:
                    filtered_args[param_name] = arguments[param_name]
                # If not provided but has default, we're good
                elif param_name in function_def.parameters and not function_def.parameters[param_name].required:
                    # Only pass default if explicitly provided in arguments
                    pass
            
            # Execute the function
            result = function_def.implementation(**filtered_args)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Create execution result
            execution_result = FunctionExecutionResult(
                function_name=function_name,
                arguments=arguments,
                result=result,
                success=True,
                execution_time=execution_time
            )
            
            # Log execution
            self._execution_history.append(execution_result)
            
            return execution_result
            
        except Exception as e:
            # Calculate execution time even for errors
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Create error result
            error_result = FunctionExecutionResult(
                function_name=function_name,
                arguments=arguments,
                result=None,
                success=False,
                error=str(e),
                execution_time=execution_time
            )
            
            # Log execution error
            self._execution_history.append(error_result)
            logger.error(f"Error executing function {function_name}: {str(e)}")
            
            return error_result
    
    def get_execution_history(self) -> List[FunctionExecutionResult]:
        """Get the history of function executions."""
        return self._execution_history
    
    def clear_execution_history(self):
        """Clear the execution history."""
        self._execution_history = []

# Global registry instance
function_registry = FunctionRegistry()

def register_function(func=None, *, name=None, description=None, parameter_descriptions=None, auto_execute=False):
    """
    Decorator to register a function with the global registry.
    
    Args:
        func (callable, optional): The function to register
        name (str, optional): Override the function name
        description (str, optional): Description of what the function does
        parameter_descriptions (dict, optional): Descriptions for parameters
        auto_execute (bool, optional): Whether to auto-execute this function
            when called by an LLM
        
    Returns:
        callable: The original function
    """
    return function_registry.register(
        func, 
        name=name, 
        description=description, 
        parameter_descriptions=parameter_descriptions,
        auto_execute=auto_execute
    ) 