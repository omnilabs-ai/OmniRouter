from typing import Dict, Any, List
import anthropic
from serverRouter.core.interfaces import ChatProvider
from serverRouter.core.datamodels import (
    ChatCompletionRequest, 
    ChatCompletionResponse, 
    ChatMessage, 
    ChatCompletionGenerator,
    FunctionCall,
    FunctionExecutionResult
)
from serverRouter.core.exceptions import ProviderError
from serverRouter.core.function_registry import ProviderType, function_registry
from dotenv import load_dotenv
import json
from sse_starlette.sse import EventSourceResponse
from serverRouter.core.models import MODELS

load_dotenv()

class AnthropicProvider(ChatProvider):
    """Anthropic chat completion provider"""
    
    def __init__(self):
        """Initialize the Anthropic provider with API key from environment"""
        try:
            self.client = anthropic.AsyncAnthropic()
        except Exception as e:
            raise ProviderError(f"Failed to initialize Anthropic client: {str(e)}")

    @property
    def provider_type(self) -> ProviderType:
        """Get the provider type for this implementation"""
        return ProviderType.ANTHROPIC
    
    async def chat_complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """
        Generate a chat completion using Anthropic's API
        
        Args:
            request: ChatCompletionRequest containing the input parameters
            
        Returns:
            ChatCompletionResponse containing the generated response
        """
        try:
            # Check if this is an extended thinking model
            model_params = {}
            model_key = next((k for k in MODELS if MODELS[k].name == request.model), None)
            
            # If this is a model with extended thinking enabled
            if model_key and hasattr(MODELS[model_key], 'extended_thinking') and MODELS[model_key].extended_thinking:
                model_params["thinking"] = {
                    "type": "extended",
                    "threshold": getattr(MODELS[model_key], 'thinking_threshold', 0.5),
                    "budget": getattr(MODELS[model_key], 'thinking_budget', 20000)
                }
            
            # Add function calling if provided
            if request.functions or request.tools:
                # Anthropic uses "tools" parameter
                if request.functions:
                    # Convert functions to Anthropic format using our registry
                    tools_data = function_registry.get_for_provider(
                        ProviderType.ANTHROPIC, 
                        [f["name"] for f in request.functions] if isinstance(request.functions, list) else None
                    )
                    model_params["tools"] = tools_data["tools"]
                elif request.tools:
                    # Tools already in Anthropic format
                    model_params["tools"] = request.tools

                # Handle tool_choice parameter if specified
                if request.tool_choice:
                    if isinstance(request.tool_choice, str) and request.tool_choice == "auto":
                        # Auto is the default for Anthropic, no need to specify
                        pass
                    elif isinstance(request.tool_choice, str) and request.tool_choice == "required":
                        model_params["tool_choice"] = "any"  # Anthropic equivalent
                    elif isinstance(request.tool_choice, dict) and "name" in request.tool_choice:
                        # Select a specific tool
                        model_params["tool_choice"] = {
                            "type": "tool",
                            "name": request.tool_choice["name"]
                        }
                
            # Create the completion with potential extended thinking parameters
            response = await self.client.messages.create(
                model=request.model,
                messages=[
                    {"role": msg.role, "content": msg.content}
                    for msg in request.messages
                ],
                max_tokens=request.max_tokens or 4092,
                temperature=request.temperature or 1.0,
                **model_params
            )
            
            # Parse function calls if any
            function_calls = await self.parse_function_calls(response)
            
            # Convert Anthropic response to our generic format
            content = ""
            if hasattr(response, 'content') and response.content:
                # For responses with tool calls, we need to get text content separately
                for content_block in response.content:
                    if content_block.type == 'text':
                        content += content_block.text
            
            return ChatCompletionResponse(
                model=response.model,
                content=content,
                provider="anthropic",
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                },
                function_calls=function_calls if function_calls else None
            )
            
        except anthropic.APIError as e:
            raise ProviderError(f"Anthropic API error: {str(e)}")
        except Exception as e:
            raise ProviderError(f"Unexpected error: {str(e)}")
    
    async def chat_complete_stream(self, request: ChatCompletionRequest) -> EventSourceResponse:
        async def event_generator():
            try:
                # Send initial metadata event
                yield {
                    "event": "metadata", 
                    "data": json.dumps({
                        "model": request.model,
                        "provider": "anthropic"
                    })
                }
                
                # Check if this is an extended thinking model
                model_params = {}
                model_key = next((k for k in MODELS if MODELS[k].name == request.model), None)
                
                # If this is a model with extended thinking enabled
                if model_key and hasattr(MODELS[model_key], 'extended_thinking') and MODELS[model_key].extended_thinking:
                    model_params["thinking"] = {
                        "type": "extended",
                        "threshold": getattr(MODELS[model_key], 'thinking_threshold', 0.5),
                        "budget": getattr(MODELS[model_key], 'thinking_budget', 20000)
                    }
                
                # Add function calling if provided
                if request.functions or request.tools:
                    # Anthropic uses "tools" parameter
                    if request.functions:
                        # Convert functions to Anthropic format using our registry
                        tools_data = function_registry.get_for_provider(
                            ProviderType.ANTHROPIC, 
                            [f["name"] for f in request.functions] if isinstance(request.functions, list) else None
                        )
                        model_params["tools"] = tools_data["tools"]
                    elif request.tools:
                        # Tools already in Anthropic format
                        model_params["tools"] = request.tools

                    # Handle tool_choice parameter if specified
                    if request.tool_choice:
                        if isinstance(request.tool_choice, str) and request.tool_choice == "auto":
                            # Auto is the default for Anthropic, no need to specify
                            pass
                        elif isinstance(request.tool_choice, str) and request.tool_choice == "required":
                            model_params["tool_choice"] = "any"  # Anthropic equivalent
                        elif isinstance(request.tool_choice, dict) and "name" in request.tool_choice:
                            # Select a specific tool
                            model_params["tool_choice"] = {
                                "type": "tool",
                                "name": request.tool_choice["name"]
                            }
                
                # Track tool calls during streaming
                tool_calls = []
                current_tool_call = None
                
                async with self.client.messages.stream(
                    model=request.model,
                    messages=[
                        {"role": msg.role, "content": msg.content}
                        for msg in request.messages
                    ],
                    max_tokens=request.max_tokens or 4092,
                    temperature=request.temperature or 1.0,
                    **model_params
                ) as stream:
                    async for chunk in stream:
                        if chunk.type == "text":
                            yield {
                                "event": "content",
                                "data": json.dumps({"content": chunk.text})
                            }
                        elif chunk.type == "message_stop":
                            yield {
                                "event": "usage",
                                "data": json.dumps({
                                    "input_tokens": chunk.message.usage.input_tokens,
                                    "output_tokens": chunk.message.usage.output_tokens,
                                    "total_tokens": chunk.message.usage.input_tokens + chunk.message.usage.output_tokens
                                })
                            }
                        # Handle tool use (function calls)
                        elif chunk.type == "tool_use":
                            # Get the tool call
                            tool_use = chunk.tool_use
                            current_tool_call = {
                                "id": tool_use.id,
                                "name": tool_use.name,
                                "input": {}
                            }
                            
                            # Parse input if available
                            if hasattr(tool_use, 'input') and tool_use.input:
                                try:
                                    if isinstance(tool_use.input, str):
                                        current_tool_call["input"] = json.loads(tool_use.input)
                                    else:
                                        current_tool_call["input"] = tool_use.input
                                except json.JSONDecodeError:
                                    current_tool_call["input"] = tool_use.input
                            
                            tool_calls.append(current_tool_call)
                            
                            # Send function call event
                            yield {
                                "event": "function_call",
                                "data": json.dumps({
                                    "id": current_tool_call["id"],
                                    "name": current_tool_call["name"],
                                    "arguments": current_tool_call["input"]
                                })
                            }
                            
                            # Send function call complete event
                            yield {
                                "event": "function_call_complete",
                                "data": json.dumps({
                                    "id": current_tool_call["id"],
                                    "name": current_tool_call["name"],
                                    "arguments": current_tool_call["input"]
                                })
                            }
                            
            except Exception as e:
                error_message = str(e)
                yield {
                    "event": "error",
                    "data": json.dumps({"error": error_message})
                }
                raise ProviderError(f"Unexpected error: {str(e)}")
        
        return EventSourceResponse(event_generator())
    
    async def parse_function_calls(self, raw_response: Any) -> List[FunctionCall]:
        """Parse function calls from Anthropic response"""
        function_calls = []
        
        if hasattr(raw_response, 'content') and raw_response.content:
            for content_block in raw_response.content:
                if content_block.type == 'tool_use':
                    try:
                        tool_use = content_block.tool_use
                        # Parse input, which could be a string or object
                        arguments = {}
                        if hasattr(tool_use, 'input'):
                            if isinstance(tool_use.input, str):
                                try:
                                    arguments = json.loads(tool_use.input)
                                except json.JSONDecodeError:
                                    arguments = {"raw_input": tool_use.input}
                            else:
                                arguments = tool_use.input
                                
                        function_calls.append(FunctionCall(
                            name=tool_use.name,
                            arguments=arguments,
                            id=tool_use.id
                        ))
                    except Exception as e:
                        pass  # Skip problematic tool_use blocks
        
        return function_calls
    
    async def create_function_response(self, function_results: List[FunctionExecutionResult]) -> Dict[str, Any]:
        """Create Anthropic-compatible function response messages"""
        content_blocks = []
        
        for result in function_results:
            tool_result = {
                "type": "tool_result",
                "tool_call_id": result.arguments.get("id", ""),
            }
            
            # Format the content based on success or failure
            if result.success:
                if isinstance(result.result, (dict, list)):
                    tool_result["content"] = json.dumps(result.result)
                else:
                    tool_result["content"] = str(result.result)
            else:
                tool_result["content"] = f"Error: {result.error}"
            
            content_blocks.append(tool_result)
        
        return {"role": "user", "content": content_blocks}


