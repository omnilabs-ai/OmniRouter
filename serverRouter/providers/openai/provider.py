from typing import Dict, Any, AsyncGenerator
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
                    "completion_tokens": response.usage.completion_tokens
                }
            )
        except Exception as e:
            raise ProviderError(f"OpenAI API error: {str(e)}")
        
    async def chat_complete_stream(self, request: ChatCompletionRequest) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Stream a chat completion using OpenAI's API"""
        try:
            # Convert messages to API format
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
            
            # Create the streaming completion
            stream = await self.client.chat.completions.create(
                model=request.model,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=True
            )
            
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
                            provider="openai",
                            finish_reason=choice.finish_reason
                        )
                    continue
                
                # Yield the chunk
                yield ChatCompletionChunk(
                    model=chunk.model,
                    content=choice.delta.content,
                    provider="openai",
                    finish_reason=choice.finish_reason
                )
                
        except Exception as e:
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