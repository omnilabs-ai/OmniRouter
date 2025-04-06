from typing import Dict, Any, List
from openai import AsyncOpenAI
import os
import asyncio
from serverRouter.core.interfaces import ChatProvider, ImageProvider
from serverRouter.core.datamodels import (
    ChatCompletionRequest, 
    ChatCompletionResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ChatCompletionGenerator,
    FunctionCall,
    FunctionExecutionResult
)
from serverRouter.core.exceptions import ProviderError
from serverRouter.core.function_registry import ProviderType, function_registry
from dotenv import load_dotenv
import json
from sse_starlette.sse import EventSourceResponse
load_dotenv()

class OpenAIProvider(ChatProvider, ImageProvider):
    """OpenAI provider supporting both chat and image generation"""
    
    def __init__(self, api_key: str = None):
        """Initialize the OpenAI provider with API key from environment"""
        try:
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ProviderError("OPENAI_API_KEY not set in environment.")
            self.client = AsyncOpenAI(api_key=api_key)

        except Exception as e:
            raise ProviderError(f"Failed to initialize OpenAI client: {str(e)}")
    
    @property
    def provider_type(self) -> ProviderType:
        """Get the provider type for this implementation"""
        return ProviderType.OPENAI
          
    async def chat_complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        try:
            # Prepare API parameters
            params = {
                "model": request.model,
                "messages": [
                    {"role": msg.role, "content": msg.content}
                    for msg in request.messages
                ],
                "temperature": request.temperature,
                "max_tokens": request.max_tokens
            }

            # Add function calling parameters if provided
            if request.functions or request.tools:
                # If functions provided directly
                if request.functions:
                    params["functions"] = request.functions
                # If tools provided (OpenAI format)
                elif request.tools:
                    params["tools"] = request.tools
                # Add tool_choice if specified
                if request.tool_choice:
                    params["tool_choice"] = request.tool_choice
                elif request.function_call:  # legacy
                    params["function_call"] = request.function_call
            
            # Call OpenAI API
            response = await self.client.chat.completions.create(**params)
            
            # Parse function calls if any
            function_calls = []
            if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    if tool_call.type == 'function':
                        try:
                            arguments = json.loads(tool_call.function.arguments)
                            function_calls.append(FunctionCall(
                                name=tool_call.function.name,
                                arguments=arguments,
                                id=tool_call.id
                            ))
                        except json.JSONDecodeError:
                            pass
            elif hasattr(response.choices[0].message, 'function_call') and response.choices[0].message.function_call:
                try:
                    arguments = json.loads(response.choices[0].message.function_call.arguments)
                    function_calls.append(FunctionCall(
                        name=response.choices[0].message.function_call.name,
                        arguments=arguments,
                        id='function_call_0'
                    ))
                except json.JSONDecodeError:
                    pass
            
            # Create response
            return ChatCompletionResponse(
                model=response.model,
                content=response.choices[0].message.content or "",
                provider="openai",
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                function_calls=function_calls if function_calls else None
            )
        except Exception as e:
            raise ProviderError(f"OpenAI API error: {str(e)}")
    
    async def chat_complete_stream(self, request: ChatCompletionRequest) -> ChatCompletionGenerator:
        async def event_generator():
            try:
                # Prepare API parameters
                params = {
                    "model": request.model,
                    "messages": [
                        {"role": msg.role, "content": msg.content}
                        for msg in request.messages
                    ],
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "stream_options": {"include_usage": True},
                    "stream": True
                }

                # Add function calling parameters if provided
                if request.functions or request.tools:
                    # If functions provided directly
                    if request.functions:
                        params["functions"] = request.functions
                    # If tools provided (OpenAI format)
                    elif request.tools:
                        params["tools"] = request.tools
                    # Add tool_choice if specified
                    if request.tool_choice:
                        params["tool_choice"] = request.tool_choice
                    elif request.function_call:  # legacy
                        params["function_call"] = request.function_call
                
                # Call OpenAI API
                stream = await self.client.chat.completions.create(**params)
                
                # Send metadata
                yield {
                    "event": "metadata",
                    "data": json.dumps({
                        "model": request.model,
                        "provider": "openai"
                    })
                }
                
                # Track function calls for streaming response
                function_calls = []
                
                # Process stream chunks
                async for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        choice = chunk.choices[0]
                        
                        # Process content
                        if hasattr(choice.delta, 'content') and choice.delta.content is not None:
                            yield {
                                "event": "content",
                                "data": json.dumps({"content": choice.delta.content})
                            }
                        
                        # Process tool calls
                        if hasattr(choice.delta, 'tool_calls') and choice.delta.tool_calls:
                            for tool_call in choice.delta.tool_calls:
                                if tool_call.type == 'function':
                                    # Find or create function call entry
                                    func_call = next(
                                        (fc for fc in function_calls if fc.get('id') == tool_call.id), 
                                        None
                                    )
                                    
                                    if not func_call:
                                        func_call = {
                                            'id': tool_call.id,
                                            'name': '',
                                            'arguments': ''
                                        }
                                        function_calls.append(func_call)
                                    
                                    # Update function information
                                    if hasattr(tool_call.function, 'name') and tool_call.function.name:
                                        func_call['name'] = tool_call.function.name
                                    
                                    if hasattr(tool_call.function, 'arguments') and tool_call.function.arguments:
                                        func_call['arguments'] += tool_call.function.arguments
                                        
                                    # Send partial function call updates
                                    yield {
                                        "event": "function_call",
                                        "data": json.dumps({
                                            "id": tool_call.id,
                                            "name": func_call['name'],
                                            "arguments": func_call['arguments']
                                        })
                                    }
                    
                    # Send usage information
                    elif hasattr(chunk, 'usage') and chunk.usage is not None:
                        yield {
                            "event": "usage",
                            "data": json.dumps({
                                "prompt_tokens": chunk.usage.prompt_tokens,
                                "completion_tokens": chunk.usage.completion_tokens,
                                "total_tokens": chunk.usage.total_tokens
                            })
                        }
                
                # Send complete function calls at the end
                if function_calls:
                    for func_call in function_calls:
                        try:
                            # Try to parse the arguments JSON
                            arguments = json.loads(func_call['arguments'])
                            yield {
                                "event": "function_call_complete",
                                "data": json.dumps({
                                    "id": func_call['id'],
                                    "name": func_call['name'],
                                    "arguments": arguments
                                })
                            }
                        except json.JSONDecodeError:
                            # In case of invalid JSON
                            yield {
                                "event": "function_call_error",
                                "data": json.dumps({
                                    "id": func_call['id'],
                                    "name": func_call['name'],
                                    "error": "Failed to parse function arguments JSON"
                                })
                            }
                        
            except Exception as e:
                error_message = str(e)
                yield {
                    "event": "error",
                    "data": json.dumps({"error": error_message})
                }
                raise ProviderError(f"OpenAI API error during streaming: {error_message}")
        
        return EventSourceResponse(event_generator())
    
    async def parse_function_calls(self, raw_response: Any) -> List[FunctionCall]:
        """Parse function calls from OpenAI response"""
        function_calls = []
        
        # Check for modern tool_calls format
        if hasattr(raw_response, 'tool_calls') and raw_response.tool_calls:
            for tool_call in raw_response.tool_calls:
                if tool_call.type == 'function':
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                        function_calls.append(FunctionCall(
                            name=tool_call.function.name,
                            arguments=arguments,
                            id=tool_call.id
                        ))
                    except json.JSONDecodeError:
                        pass
        
        # Check for legacy function_call format
        elif hasattr(raw_response, 'function_call') and raw_response.function_call:
            try:
                arguments = json.loads(raw_response.function_call.arguments)
                function_calls.append(FunctionCall(
                    name=raw_response.function_call.name,
                    arguments=arguments,
                    id='function_call_0'
                ))
            except json.JSONDecodeError:
                pass
        
        return function_calls
    
    async def create_function_response(self, function_results: List[FunctionExecutionResult]) -> List[Dict[str, Any]]:
        """Create OpenAI-compatible function response messages"""
        messages = []
        
        for result in function_results:
            message = {
                "role": "tool",
                "tool_call_id": result.arguments.get("id", "function_call_0"),
                "content": json.dumps(result.result) if result.success else f"Error: {result.error}"
            }
            messages.append(message)
        
        return messages

    async def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        try:
            response = await self.client.images.generate(
                model=request.model,
                prompt=request.prompt,
                size=request.size.value,
                quality=request.quality,
                n=request.n,
                response_format="b64_json"
            )
            
            data_urls = [
                f"data:image/png;base64,{image.b64_json}" 
                for image in response.data
            ]
            
            return ImageGenerationResponse(
                urls=data_urls,
                model=request.model,
                provider="openai"
            )
        except Exception as e:
            raise ProviderError(f"OpenAI API error: {str(e)}")