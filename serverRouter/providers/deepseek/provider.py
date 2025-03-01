from typing import Dict, Any, List, Optional, AsyncGenerator
from openai import AsyncOpenAI
from serverRouter.core.interfaces import ChatProvider
from serverRouter.core.datamodels import (
    ChatCompletionRequest, 
    ChatCompletionResponse,
    ChatCompletionChunk
)
from serverRouter.core.exceptions import ProviderError
from dotenv import load_dotenv
import os
import logging

load_dotenv()

class DeepSeekProvider(ChatProvider):
    """DeepSeek R1 provider with tool calling support"""
    
    def __init__(self, api_key: str = None, base_url: str = None):
        api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ProviderError("Missing DEEPSEEK_API_KEY")
            
        base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        
        try:
            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            self.supported_models = ["deepseek-r1"]
        except Exception as e:
            logging.exception("Client init error")
            raise ProviderError(f"Client init failed: {str(e)}")

    async def chat_complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        try:
            # Convert messages to API format
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
            
            # Base parameters
            params = {
                "model": request.model,
                "messages": messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "stream": False
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
            tool_calls = []
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
                content=response.choices[0].message.content,
                provider="deepseek",
                tool_calls=tool_calls,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            )
            
        except Exception as e:
            raise ProviderError(f"API request failed: {str(e)}")
            
    async def chat_complete_stream(self, request: ChatCompletionRequest) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Stream a chat completion using DeepSeek's API (which uses OpenAI's format)"""
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
            
            # Stream the chunks
            async for chunk in stream:
                if not chunk.choices:
                    continue
                    
                choice = chunk.choices[0]
                
                if not choice.delta or not choice.delta.content:
                    if choice.finish_reason:
                        # Final chunk without content, just the finish reason
                        yield ChatCompletionChunk(
                            model=chunk.model,
                            content="",
                            provider="deepseek",
                            finish_reason=choice.finish_reason
                        )
                    continue
                
                # Yield the chunk
                yield ChatCompletionChunk(
                    model=chunk.model,
                    content=choice.delta.content,
                    provider="deepseek",
                    finish_reason=choice.finish_reason
                )
                
        except Exception as e:
            logging.exception("DeepSeek streaming error")
            raise ProviderError(f"DeepSeek streaming API error: {str(e)}")