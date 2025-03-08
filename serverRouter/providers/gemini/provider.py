# serverRouter/providers/gemini/provider.py
from typing import Dict, Any, List, Union
from google import generativeai as genai
import os
from typing import Dict, Any, List, Union
from serverRouter.core.interfaces import ChatProvider, ImageProvider
from serverRouter.core.datamodels import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
)
from serverRouter.core.exceptions import ProviderError
from dotenv import load_dotenv
import base64
from io import BytesIO
load_dotenv()

class GeminiProvider(ChatProvider, ImageProvider):
    def __init__(self, api_key: str = None):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ProviderError("No GEMINI_API_KEY provided. Please add it to your .env file.")
        genai.configure(api_key=api_key)
        self.client = genai.Client(api_key=api_key)
        
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
    
    async def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        """Generate images using Gemini's Imagen model"""
        try:
            # Default to 1 image if not specified
            num_images = request.n if request.n else 1
            
            # Call the Imagen API using the correct import and method
            from google.genai import types
            
            response = self.client.models.generate_images(
                model='imagen-3.0-generate-002',  # Using the specific model
                prompt=request.prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=num_images,
                )
            )
            
            # Convert images to base64 data URLs
            data_urls = []
            for generated_image in response.generated_images:
                # Convert image bytes to base64
                img_base64 = base64.b64encode(generated_image.image.image_bytes).decode('utf-8')
                data_urls.append(f"data:image/png;base64,{img_base64}")
            
            return ImageGenerationResponse(
                urls=data_urls,
                model=request.model,
                provider="gemini"
            )
            
        except Exception as e:
            raise ProviderError(f"Gemini API error (image): {str(e)}")
        
