# serverRouter/providers/gemini/provider.py
from typing import Dict, Any, List, Union
from google import generativeai as genai
import os
from typing import Dict, Any, List, Union
from serverRouter.core.interfaces import ChatProvider
from serverRouter.core.datamodels import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionGenerator
)
from serverRouter.core.exceptions import ProviderError
from dotenv import load_dotenv
import json
from sse_starlette.sse import EventSourceResponse

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
                    usage={
                        "prompt_tokens": response.usage_metadata.prompt_token_count,
                        "completion_tokens": response.usage_metadata.candidates_token_count,
                        "total_tokens": response.usage_metadata.prompt_token_count + response.usage_metadata.candidates_token_count
                    }
                )
            else:
                raise ProviderError("Empty response from Gemini API")
                
        except Exception as e:
            raise ProviderError(f"Gemini API error (chat): {str(e)}")
        
    async def chat_complete_stream(self, request: ChatCompletionRequest) -> ChatCompletionGenerator:
        async def event_generator():
            try:
                messages = []
                for msg in request.messages:
                    role = "model" if msg.role == "assistant" else msg.role
                    messages.append({"role": role, "parts": [msg.content]})

                model = genai.GenerativeModel(model_name=request.model)
                
                # Send metadata event at the beginning
                yield {
                    "event": "metadata",
                    "data": json.dumps({
                        "model": request.model,
                        "provider": "gemini"
                    })
                }
                
                response = model.generate_content(
                    contents=messages,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=request.max_tokens or 2048,
                        temperature=request.temperature or 1.0
                    ),
                    stream=True
                )
                
                total_prompt_tokens = 0
                total_completion_tokens = 0

                for chunk in response:
                    if chunk.text:
                        total_prompt_tokens = chunk.usage_metadata.prompt_token_count
                        total_completion_tokens = chunk.usage_metadata.candidates_token_count
                        yield {
                            "event": "content",
                            "data": json.dumps({"content": chunk.text})
                        }

                yield {
                    "event": "usage",
                    "data": json.dumps({
                        "prompt_tokens": total_prompt_tokens,
                        "completion_tokens": total_completion_tokens,
                        "total_tokens": total_prompt_tokens + total_completion_tokens
                    })
                }
                
            except Exception as e:
                error_message = str(e)
                yield {
                    "event": "error",
                    "data": json.dumps({"error": error_message})
                }
                raise ProviderError(f"Gemini API error (stream): {str(e)}")
        
        return EventSourceResponse(event_generator())
        
        
