# serverRouter/providers/together/provider.py
import os
import asyncio
from typing import List, Dict, Any
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
import json
from sse_starlette.sse import EventSourceResponse

class TogetherAIProvider(ChatProvider, ImageProvider):
    """
    Provider for Together AI.
    Uses the official Together library to call chat completions and generate images.
    """

    def __init__(self, api_key: str = None):
        # Get the Together API key from the environment variable TOGETHER_API_KEY
        api_key = api_key or os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise ProviderError("TOGETHER_API_KEY not set in environment.")
        try:
            # Make sure the together library is installed (pip install together)
            from together import Together
        except ImportError:
            raise ProviderError("together package not installed. Run 'pip install together'")
        # Instantiate the Together client with the key and base URL
        self.client = Together(api_key=api_key, base_url="https://api.together.xyz/v1")

    @property
    def provider_type(self) -> ProviderType:
        """Get the provider type for this implementation"""
        return ProviderType.TOGETHER

    async def chat_complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """
        Call Together AI's chat completion endpoint.
        Since the Together API is synchronous, we run it in a thread to avoid blocking.
        """
        try:
            # Prepare parameters
            params = {
                "model": request.model,
                "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
            }

            # Add function calling support
            if request.functions or request.tools:
                # Together AI uses OpenAI-compatible format
                if request.functions:
                    # Convert functions to the OpenAI format
                    tools = function_registry.get_for_provider(
                        ProviderType.OPENAI, 
                        [f["name"] for f in request.functions] if isinstance(request.functions, list) else None
                    )
                    params["tools"] = tools
                elif request.tools:
                    # Tools already in OpenAI format
                    params["tools"] = request.tools

                # Add tool_choice if specified
                if request.tool_choice:
                    params["tool_choice"] = request.tool_choice
                elif request.function_call:  # legacy
                    params["function_call"] = request.function_call
            
            # Run the synchronous API call in a thread pool to avoid blocking
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                **params
            )
            
            # Parse function calls if any
            function_calls = await self.parse_function_calls(response)
            
            return ChatCompletionResponse(
                model=response.model,
                content=response.choices[0].message.content or "",  # Handle None content
                provider="together",
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                function_calls=function_calls if function_calls else None
            )
        except Exception as e:
            raise ProviderError(f"Together AI API error: {str(e)}")
        
    async def chat_complete_stream(self, request: ChatCompletionRequest) -> ChatCompletionGenerator:
        async def event_generator():
            try:
                # Send initial metadata event
                yield {
                    "event": "metadata", 
                    "data": json.dumps({
                        "model": request.model,
                        "provider": "together"
                    })
                }
                
                # Prepare parameters
                params = {
                    "model": request.model,
                    "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "stream": True
                }

                # Add function calling support
                if request.functions or request.tools:
                    # Together AI uses OpenAI-compatible format
                    if request.functions:
                        # Convert functions to the OpenAI format
                        tools = function_registry.get_for_provider(
                            ProviderType.OPENAI, 
                            [f["name"] for f in request.functions] if isinstance(request.functions, list) else None
                        )
                        params["tools"] = tools
                    elif request.tools:
                        # Tools already in OpenAI format
                        params["tools"] = request.tools

                    # Add tool_choice if specified
                    if request.tool_choice:
                        params["tool_choice"] = request.tool_choice
                    elif request.function_call:  # legacy
                        params["function_call"] = request.function_call
                
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    **params
                )

                total_completion_tokens = 0
                function_calls = []
                
                for chunk in response:
                    # Process content
                    if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        total_completion_tokens += 1
                        yield {
                            "event": "content",
                            "data": json.dumps({"content": content})
                        }
                    
                    # Process tool calls (function calls)
                    if hasattr(chunk.choices[0].delta, 'tool_calls') and chunk.choices[0].delta.tool_calls:
                        for tool_call in chunk.choices[0].delta.tool_calls:
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
                    
                    # Process usage information
                    if hasattr(chunk, 'usage') and chunk.usage is not None:
                        yield {
                            "event": "usage",
                            "data": json.dumps({
                                "prompt_tokens": chunk.usage.prompt_tokens,
                                "completion_tokens": chunk.usage.completion_tokens,
                                "total_tokens": chunk.usage.total_tokens
                            })
                        }
                
                # Process complete function calls at the end
                for func_call in function_calls:
                    try:
                        # Try to parse as JSON
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
                        # Handle invalid JSON
                        yield {
                            "event": "function_call_error",
                            "data": json.dumps({
                                "id": func_call['id'],
                                "name": func_call['name'],
                                "error": "Failed to parse function arguments JSON"
                            })
                        }
                
            except Exception as e:
                # Send error event in case of exception
                yield {
                    "event": "error",
                    "data": json.dumps({"error": str(e)})
                }
                raise ProviderError(f"Together AI API error (stream): {str(e)}")
        
        return EventSourceResponse(event_generator())

    async def parse_function_calls(self, raw_response: Any) -> List[FunctionCall]:
        """Parse function calls from Together AI response (OpenAI-compatible format)"""
        function_calls = []
        
        # Check for modern tool_calls format
        if hasattr(raw_response.choices[0].message, 'tool_calls') and raw_response.choices[0].message.tool_calls:
            for tool_call in raw_response.choices[0].message.tool_calls:
                if tool_call.type == 'function':
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                        function_calls.append(FunctionCall(
                            name=tool_call.function.name,
                            arguments=arguments,
                            id=tool_call.id
                        ))
                    except json.JSONDecodeError:
                        # Handle invalid JSON
                        function_calls.append(FunctionCall(
                            name=tool_call.function.name,
                            arguments={"raw_input": tool_call.function.arguments},
                            id=tool_call.id
                        ))
        
        # Check for legacy function_call format
        elif hasattr(raw_response.choices[0].message, 'function_call') and raw_response.choices[0].message.function_call:
            try:
                arguments = json.loads(raw_response.choices[0].message.function_call.arguments)
                function_calls.append(FunctionCall(
                    name=raw_response.choices[0].message.function_call.name,
                    arguments=arguments,
                    id='function_call_0'
                ))
            except json.JSONDecodeError:
                # Handle invalid JSON
                function_calls.append(FunctionCall(
                    name=raw_response.choices[0].message.function_call.name,
                    arguments={"raw_input": raw_response.choices[0].message.function_call.arguments},
                    id='function_call_0'
                ))
        
        return function_calls
    
    async def create_function_response(self, function_results: List[FunctionExecutionResult]) -> List[Dict[str, Any]]:
        """Create Together AI-compatible function response messages (OpenAI format)"""
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
        """
        Call Together AI's image generation endpoint.
        Since the Together API is synchronous, we run it in a thread to avoid blocking.
        """
        try:
            # Parse size to get width and height
            if request.size.value == "1024x1024":
                width, height = 1024, 1024
            elif request.size.value == "512x512":
                width, height = 512, 512
            elif request.size.value == "256x256":
                width, height = 256, 256
            else:
                width, height = 1024, 1024

            # Custom parameter for FLUX models
            steps = 4  # Default value for FLUX, can be adjusted if needed
            
            # Run the synchronous API call in a thread pool to avoid blocking
            response = await asyncio.to_thread(
                self.client.images.generate,
                prompt=request.prompt,
                model=request.model,
                width=width,
                height=height,
                steps=steps,
                n=request.n,
                response_format="b64_json"
            )
            
            # Convert b64_json responses to data URLs
            image_urls = []
            for image_data in response.data:
                # Create a data URL from the base64 data
                image_urls.append(f"data:image/png;base64,{image_data.b64_json}")
            
            return ImageGenerationResponse(
                urls=image_urls,
                model=request.model,
                provider="together"
            )
        except Exception as e:
            raise ProviderError(f"Together AI Image API error: {str(e)}")


