# serverRouter/providers/gemini/provider.py
from typing import Dict, Any, List, Union
from google import genai
from google.genai import types
import os
import base64
from io import BytesIO
from PIL import Image
from serverRouter.core.interfaces import ChatProvider, ImageProvider
from serverRouter.core.datamodels import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
)
from serverRouter.core.exceptions import ProviderError
from dotenv import load_dotenv
load_dotenv()

class GeminiProvider(ChatProvider, ImageProvider):
    """Provider for Google Gemini models with flexible API support."""
    
    def __init__(self, api_key: str = None):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ProviderError("No GEMINI_API_KEY provided. Please add it to your .env file.")
            
        # Create the client using the new API approach
        try:
            self.client = genai.Client(api_key=api_key)
        except AttributeError:
            raise ProviderError(
                "Your google-generativeai package is outdated. "
                "Please update with: pip install --upgrade google-generativeai"
            )
        
    async def chat_complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Generate chat completions with Gemini models using client API."""
        try:
            # Check if we have a single message or a conversation
            if len(request.messages) == 1:
                # For single messages, use the simpler format
                content = request.messages[0].content
                response = self.client.models.generate_content(
                    model=request.model,
                    contents=content  # Just the string content
                )
            else:
                # For conversations, create a properly formatted history
                formatted_messages = []
                for msg in request.messages:
                    role = "model" if msg.role == "assistant" else msg.role
                    content_part = {"text": msg.content}
                    message = {"role": role, "parts": [content_part]}
                    formatted_messages.append(message)
                
                response = self.client.models.generate_content(
                    model=request.model,
                    contents=formatted_messages
                )

            if response and response.text:
                return ChatCompletionResponse(
                    model=request.model,
                    content=response.text,
                    provider="gemini",
                    usage={}  
                )
            else:
                raise ProviderError("Empty response from Gemini API")
                
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            raise ProviderError(f"Gemini API error (chat): {str(e)}\n{error_details}")
    
    async def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        """Generate images using Gemini's Imagen model."""
        try:
            # Default to 1 image if not specified
            num_images = request.n if request.n else 1
            
            # Call the Imagen API using client.models.generate_images
            response = self.client.models.generate_images(
                model='imagen-3.0-generate-002',
                prompt=request.prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=num_images,
                )
            )
            
            # Process images to base64 data URLs
            data_urls = []
            for generated_image in response.generated_images:
                image = Image.open(BytesIO(generated_image.image.image_bytes))
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                data_urls.append(f"data:image/png;base64,{img_str}")
            
            if not data_urls:
                raise ProviderError("No images generated")
                
            return ImageGenerationResponse(
                urls=data_urls,
                model=request.model,
                provider="gemini"
            )
            
        except Exception as e:
            raise ProviderError(f"Gemini API error (image): {str(e)}")
        
