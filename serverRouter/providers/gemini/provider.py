# serverRouter/providers/gemini/provider.py
from typing import Dict, Any, List, Union, AsyncGenerator
from google import generativeai as genai
import os
from typing import Dict, Any, List, Union
from serverRouter.core.interfaces import ChatProvider
from serverRouter.core.datamodels import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk
)
from serverRouter.core.exceptions import ProviderError
import asyncio
from dotenv import load_dotenv
load_dotenv()

class GeminiProvider(ChatProvider):
    def __init__(self, api_key: str = None):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ProviderError("No GEMINI_API_KEY provided. Please add it to your .env file.")
        genai.configure(api_key=api_key)
        
    async def chat_complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        try:
            messages = []
            for msg in request.messages:
                role = "model" if msg.role == "assistant" else msg.role
                messages.append({"role": role, "parts": [msg.content]})

            model = genai.GenerativeModel(model_name=request.model)

            # Use synchronous version for simpler operation
            response = model.generate_content(
                contents=messages,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=request.max_tokens or 2048,
                    temperature=request.temperature or 1.0
                )
            )

            if response and response.text:
                return ChatCompletionResponse(
                    model=request.model,
                    content=response.text,
                    provider="gemini",
                    usage={}  # Gemini doesn't directly provide usage stats
                )
            else:
                raise ProviderError("Empty response from Gemini API")
                
        except Exception as e:
            raise ProviderError(f"Gemini API error (chat): {str(e)}")
    async def chat_complete_stream(self, request: ChatCompletionRequest) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Stream chat completions from Gemini API"""
        try:
            messages = []
            for msg in request.messages:
                role = "model" if msg.role == "assistant" else msg.role
                messages.append({"role": role, "parts": [msg.content]})

            model = genai.GenerativeModel(model_name=request.model)
            
            # Create a streaming response
            loop = asyncio.get_event_loop()
            response_iterator = await loop.run_in_executor(
                None,
                lambda: model.generate_content(
                    contents=messages,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=request.max_tokens or 2048,
                        temperature=request.temperature or 1.0
                    ),
                    stream=True
                )
            )
            
            # Process the chunks one by one
            while True:
                try:
                    # Get next chunk in a non-blocking way
                    chunk = await loop.run_in_executor(None, next, response_iterator, None)
                    if chunk is None:  # End of iterator
                        break
                        
                    if hasattr(chunk, 'text') and chunk.text:
                        yield ChatCompletionChunk(
                            model=request.model,
                            content=chunk.text,
                            provider="gemini",
                            finish_reason=None
                        )
                except StopIteration:
                    break
            
            # Send final chunk with finish reason
            yield ChatCompletionChunk(
                model=request.model,
                content="",
                provider="gemini",
                finish_reason="stop"
            )
            
        except Exception as e:
            raise ProviderError(f"Gemini streaming API error: {str(e)}")