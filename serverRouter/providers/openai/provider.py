from typing import Dict, Any, AsyncGenerator, List
from openai import AsyncOpenAI
import os
from serverRouter.core.interfaces import ChatProvider, ImageProvider
from serverRouter.core.datamodels import (
    ChatCompletionRequest, 
    ChatCompletionResponse,
    ChatCompletionChunk,
    ImageGenerationRequest,
    ImageGenerationResponse
)
from serverRouter.core.exceptions import ProviderError
from dotenv import load_dotenv
import logging
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

            
    async def chat_complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        try:
            # Base parameters
            params = {
                "model": request.model,
                "messages": [
                    {"role": msg.role, "content": msg.content}
                    for msg in request.messages
                ],
                "temperature": request.temperature,
                "max_tokens": request.max_tokens
            }
            
            # Add tools if specified
            if hasattr(request, "tools") and request.tools:
                params["tools"] = request.tools
                
            # Add tool choice if specified
            if hasattr(request, "tool_choice") and request.tool_choice:
                params["tool_choice"] = request.tool_choice
                
            # Add response format if specified
            if hasattr(request, "response_format") and request.response_format:
                params["response_format"] = request.response_format

            # API call
            response = await self.client.chat.completions.create(**params)
            
            # Extract tool calls if present
            tool_calls = None
            if response.choices[0].message.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in response.choices[0].message.tool_calls
                ]
            
            return ChatCompletionResponse(
                model=response.model,
                content=response.choices[0].message.content or "",  # Handle None content when only tool calls
                provider="openai",
                tool_calls=tool_calls,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens
                }
            )
        except Exception as e:
            logging.exception("OpenAI API error")
            raise ProviderError(f"OpenAI API error: {str(e)}")
        
    async def chat_complete_stream(self, request: ChatCompletionRequest) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Stream a chat completion using OpenAI's API"""
        try:
            # Convert messages to API format
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
            
            # Base parameters
            params = {
                "model": request.model,
                "messages": messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "stream": True
            }
            
            # Add tools if specified
            if hasattr(request, "tools") and request.tools:
                params["tools"] = request.tools
                
            # Add tool choice if specified
            if hasattr(request, "tool_choice") and request.tool_choice:
                params["tool_choice"] = request.tool_choice
                
            # Add response format if specified
            if hasattr(request, "response_format") and request.response_format:
                params["response_format"] = request.response_format
            
            # Create the streaming completion
            stream = await self.client.chat.completions.create(**params)
            
            # Variables to track tool calls
            current_tool_calls = []
            tool_call_chunks = {}  # Map from tool_call_id to accumulated chunks
            
            # Stream the chunks
            async for chunk in stream:
                if not chunk.choices:
                    continue
                    
                choice = chunk.choices[0]
                delta = choice.delta
                
                # Handle tool call chunks
                if delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        # Initialize tool call if new
                        tool_call_id = tool_call.id
                        if tool_call_id not in tool_call_chunks:
                            tool_call_chunks[tool_call_id] = {
                                "id": tool_call_id,
                                "type": tool_call.type or "function",
                                "function": {
                                    "name": tool_call.function.name or "",
                                    "arguments": tool_call.function.arguments or ""
                                }
                            }
                        else:
                            # Update existing tool call
                            if tool_call.function.name:
                                tool_call_chunks[tool_call_id]["function"]["name"] = tool_call.function.name
                            if tool_call.function.arguments:
                                tool_call_chunks[tool_call_id]["function"]["arguments"] += tool_call.function.arguments
                        
                        # Add to current batch of tool calls
                        current_tool_calls = list(tool_call_chunks.values())
                        
                        # Yield a chunk with the updated tool calls
                        yield ChatCompletionChunk(
                            model=chunk.model,
                            content="",
                            provider="openai",
                            finish_reason=None,
                            tool_calls=current_tool_calls
                        )
                        continue
                
                # Handle regular content chunks
                if delta.content:
                    yield ChatCompletionChunk(
                        model=chunk.model,
                        content=delta.content,
                        provider="openai",
                        finish_reason=choice.finish_reason
                    )
                elif choice.finish_reason:
                    # Final chunk without content, just the finish reason
                    yield ChatCompletionChunk(
                        model=chunk.model,
                        content="",
                        provider="openai",
                        finish_reason=choice.finish_reason,
                        tool_calls=current_tool_calls if current_tool_calls else None
                    )
                
        except Exception as e:
            logging.exception("OpenAI streaming API error")
            raise ProviderError(f"OpenAI streaming API error: {str(e)}")

    async def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        try:
            response = await self.client.images.generate(
                model=request.model,
                prompt=request.prompt,
                size=request.size.value,
                quality=request.quality,
                n=request.n
            )
            
            return ImageGenerationResponse(
                urls=[image.url for image in response.data],
                model=request.model,
                provider="openai"
            )
        except Exception as e:
            raise ProviderError(f"OpenAI API error: {str(e)}")