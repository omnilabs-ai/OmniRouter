from typing import Dict, Any, List
import os
import aiohttp
from serverRouter.core.interfaces import ImageProvider
from serverRouter.core.datamodels import ImageGenerationRequest, ImageGenerationResponse, ImageSize
from serverRouter.core.exceptions import ProviderError
import base64
import json

class DeepAIProvider(ImageProvider):
    """Provider for DeepAI image generation"""
    
    def __init__(self, api_key: str = None, base_url: str = None):
        """Initialize the DeepAI provider with API key from environment"""
        api_key = api_key or os.getenv("DEEPAI_API_KEY")
        if not api_key:
            raise ProviderError("Missing DEEPAI_API_KEY")
            
        self.api_key = api_key
        self.base_url = base_url or "https://api.deepai.org/api/text2img"
        
    async def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        """Generate an image using DeepAI API"""
        try:
            # Map ImageSize to DeepAI dimensions
            size_mapping = {
                ImageSize.SMALL: "256",  # 256x256
                ImageSize.MEDIUM: "512", # 512x512
                ImageSize.LARGE: "1024"  # 1024x1024
            }
            
            # Set up image size/dimensions
            image_size = size_mapping.get(request.size, "512")
            
            # Parse model name to determine generator version and preferences
            generator_version = "standard"
            genius_preference = None
            
            if request.model == "deepai-hd":
                generator_version = "hd"
            elif request.model == "deepai-genius":
                generator_version = "genius"
                # Default to cinematic style for genius model
                genius_preference = "cinematic"  
                
                # Check if a specific style is specified in request
                # This would require extending the ImageGenerationRequest class
                # For now, we'll use a default
            elif request.model == "deepai-genius-anime":
                generator_version = "genius"
                genius_preference = "anime"
            elif request.model == "deepai-genius-photography":
                generator_version = "genius" 
                genius_preference = "photography"
            elif request.model == "deepai-genius-graphic":
                generator_version = "genius"
                genius_preference = "graphic"
            
            # Set up headers with API key
            headers = {
                "api-key": self.api_key
            }
            
            # Create form data for the API request
            data = {
                'text': request.prompt,
                'width': image_size,
                'height': image_size,
                'image_generator_version': generator_version
            }
            
            # Add genius preference if applicable
            if genius_preference:
                data['genius_preference'] = genius_preference
                
            # Add negative prompt if needed 
            # This could be added to ImageGenerationRequest in the future
            # data['negative_prompt'] = "..."
            
            # Make the API request
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url,
                    headers=headers,
                    data=data
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ProviderError(f"DeepAI API error: {error_text}")
                    
                    # Parse the response
                    response_json = await response.json()
                    
                    # DeepAI returns an output_url that we need to fetch
                    if "output_url" not in response_json:
                        raise ProviderError("DeepAI API did not return an image URL")
                    
                    image_url = response_json["output_url"]
                    
                    # Fetch the actual image
                    async with session.get(image_url) as img_response:
                        if img_response.status != 200:
                            raise ProviderError(f"Failed to fetch generated image: {img_response.status}")
                        
                        # Get the binary image data
                        image_data = await img_response.read()
                        
                        # Convert to base64 for transport
                        base64_image = base64.b64encode(image_data).decode('utf-8')
                        data_url = f"data:image/jpeg;base64,{base64_image}"
                        
                        return ImageGenerationResponse(
                            urls=[data_url],
                            model=request.model,
                            provider="deepai"
                        )
        
        except Exception as e:
            raise ProviderError(f"DeepAI API error: {str(e)}") 