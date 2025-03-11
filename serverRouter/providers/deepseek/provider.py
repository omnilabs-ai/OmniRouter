from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
from serverRouter.core.interfaces import ChatProvider
from serverRouter.core.datamodels import ChatCompletionRequest, ChatCompletionResponse, ChatCompletionGenerator
from serverRouter.core.exceptions import ProviderError
from dotenv import load_dotenv
import os
import logging
import json
load_dotenv()

class DeepSeekProvider(ChatProvider):
    """DeepSeek R1 provider with tool calling support"""
    
    def __init__(self, api_key: str = None):
        api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ProviderError("Missing DEEPSEEK_API_KEY")
        
        base_url = "https://api.deepseek.com"
        
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

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
                provider="deepseek",
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens
                }
            )
        except Exception as e:
            raise ProviderError(f"DeepSeek API error: {str(e)}")
    
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
            raise ProviderError(f"DeepSeek API error during streaming: {str(e)}")
