from typing import Dict, Any, List
import os
import aiohttp
from serverRouter.core.interfaces import ImageProvider
from serverRouter.core.datamodels import ImageGenerationRequest, ImageGenerationResponse, ImageSize
from serverRouter.core.exceptions import ProviderError
import base64
from io import BytesIO

class StableDiffusionProvider(ImageProvider):
    """Provider for Stable Diffusion image generation"""
    
    def __init__(self, api_key: str = None, base_url: str = None):
        api_key = api_key or os.getenv("STABILITY_API_KEY")
        if not api_key:
            raise ProviderError("Missing STABILITY_API_KEY")
            
        self.api_key = api_key
        self.base_url = base_url or "https://api.stability.ai/v2beta/stable-image/generate/sd3"
        
    async def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        """Generate an image using Stable Diffusion API"""
        try:
            # Map size to aspect ratio
            size_mapping = {
                ImageSize.SMALL: "1:1",
                ImageSize.MEDIUM: "1:1",
                ImageSize.LARGE: "1:1"
            }
            aspect_ratio = size_mapping.get(request.size, "1:1")
            
            # Parse model name
            model_name = "sd3.5-large"  # default
            if request.model == "stable-diffusion-3.5-turbo":
                model_name = "sd3.5-large-turbo"
            elif request.model == "stable-diffusion-3.5-large":
                model_name = "sd3.5-large"
            
            headers = {
                "Accept": "image/*",
                "Authorization": f"Bearer {self.api_key}"
                # Let aiohttp set the Content-Type header for multipart/form-data
            }
            
            # Create form data
            form = aiohttp.FormData(quote_fields=False)
            form.add_field('prompt', request.prompt)
            form.add_field('negative_prompt', '')
            form.add_field('aspect_ratio', aspect_ratio)
            form.add_field('seed', '0')
            form.add_field('output_format', 'jpeg')
            form.add_field('model', model_name)
            form.add_field('mode', 'text-to-image')
            
            # Add a dummy file field to ensure multipart/form-data
            form.add_field('file', b'', filename='', content_type='application/octet-stream')
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url, 
                    headers=headers,
                    data=form
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ProviderError(f"Stable Diffusion API error: {error_text}")
                    
                    # Check for content filtering
                    finish_reason = response.headers.get("finish-reason")
                    if finish_reason == 'CONTENT_FILTERED':
                        raise ProviderError("Generation failed NSFW classifier")
                    
                    # Get the binary image data
                    image_data = await response.read()
                    # Convert to base64 for transport
                    base64_image = base64.b64encode(image_data).decode('utf-8')
                    image_url = f"data:image/jpeg;base64,{base64_image}"
                    
                    return ImageGenerationResponse(
                        urls=[image_url],
                        model=request.model,
                        provider="stablediffusion"
                    )
                    
        except Exception as e:
            raise ProviderError(f"Stable Diffusion API error: {str(e)}") 