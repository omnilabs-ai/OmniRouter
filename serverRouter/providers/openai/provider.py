from typing import Dict, Any
from openai import AsyncOpenAI
import os
import asyncio
from serverRouter.core.interfaces import ChatProvider, ImageProvider
from serverRouter.core.datamodels import (
    ChatCompletionRequest, 
    ChatCompletionResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ChatCompletionGenerator
)
from serverRouter.core.exceptions import ProviderError
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
          
    async def chat_complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        try:
            response = await self.client.chat.completions.create(
                model=request.model,
                messages=[
                    {"role": msg.role, "content": msg.content}
                    for msg in request.messages
                ],
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            return ChatCompletionResponse(
                model=response.model,
                content=response.choices[0].message.content,
                provider="openai",
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.prompt_tokens + response.usage.completion_tokens
                }
            )
        except Exception as e:
            raise ProviderError(f"OpenAI API error: {str(e)}")
    
    async def chat_complete_stream(self, request: ChatCompletionRequest) -> ChatCompletionGenerator:
        async def event_generator():
            try:
                stream = await self.client.chat.completions.create(
                    model=request.model,
                    messages=[
                        {"role": msg.role, "content": msg.content}
                        for msg in request.messages
                    ],
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    stream_options={"include_usage": True},
                    stream=True
                )
                
                yield {
                    "event": "metadata",
                    "data": json.dumps({
                        "model": request.model,
                        "provider": "openai"
                    })
                }
                
                async for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        choice = chunk.choices[0]
                        if hasattr(choice.delta, 'content') and choice.delta.content is not None:
                            yield {
                                "event": "content",
                                "data": json.dumps({"content": choice.delta.content})
                            }
                    
                    elif hasattr(chunk, 'usage') and chunk.usage is not None:
                        yield {
                            "event": "usage",
                            "data": json.dumps({
                                "prompt_tokens": chunk.usage.prompt_tokens,
                                "completion_tokens": chunk.usage.completion_tokens,
                                "total_tokens": chunk.usage.total_tokens
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