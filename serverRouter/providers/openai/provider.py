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
    
    async def chat_complete_stream(self, request: ChatCompletionRequest) -> ChatCompletionGenerator:
        try:
            stream = await self.client.chat.completions.create(
                model=request.model,
                messages=[
                    {"role": msg.role, "content": msg.content}
                    for msg in request.messages
                ],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    # Format as proper SSE
                    content = chunk.choices[0].delta.content
                    yield f"data: {json.dumps({'content': content})}\n\n"
            
            # Signal end of stream
            yield "data: [DONE]\n\n"
                    
        except Exception as e:
            raise ProviderError(f"OpenAI API error during streaming: {str(e)}")

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