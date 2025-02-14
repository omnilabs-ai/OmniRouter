# serverRouter/providers/gemini/provider.py
from typing import Dict, Any, List, Union
from google import generativeai as genai
import os
from typing import Dict, Any, List, Union
from serverRouter.core.interfaces import ChatProvider
from serverRouter.core.datamodels import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from serverRouter.core.exceptions import ProviderError
from dotenv import load_dotenv
load_dotenv()

class GeminiProvider(ChatProvider):
    def __init__(self, api_key: str = None):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ProviderError("No GEMINI_API_KEY provided. Please add it to your .env file.")
        
    async def chat_complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        try:
            messages = []
            for msg in request.messages:
                messages.append({"role": msg.role, "parts": [msg.content]})

            model = genai.GenerativeModel(model_name=request.model)

            response = await model.generate_content_async(
                contents=messages,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=request.max_tokens or 2048,  # Increased default
                    temperature=request.temperature
                )
            )

            if response and response.text:
                return ChatCompletionResponse(
                    model=request.model,
                    content=response.text,
                    provider="gemini",
                    usage={}  #  Gemini doesn't directly provide usage in the same way
                )
        except Exception as e:
            raise ProviderError(f"Gemini API error (chat): {str(e)}")
        
