# serverRouter/providers/together/provider.py
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
import json

class TogetherAIProvider(ChatProvider, ImageProvider):
    """
    Provider for Together AI.
    Uses the official Together library to call chat completions and generate images.
    """

    def __init__(self, api_key: str = None):
        # Get the Together API key from the environment variable TOGETHER_API_KEY
        api_key = api_key or os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise ProviderError("TOGETHER_API_KEY not set in environment.")
        try:
            # Make sure the together library is installed (pip install together)
            from together import Together
        except ImportError:
            raise ProviderError("together package not installed. Run 'pip install together'")
        # Instantiate the Together client with the key and base URL
        self.client = Together(api_key=api_key, base_url="https://api.together.xyz/v1")

    async def chat_complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """
        Call Together AI's chat completion endpoint.
        Since the Together API is synchronous, we run it in a thread to avoid blocking.
        """
        try:
            # Run the synchronous API call in a thread pool to avoid blocking
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=request.model,
                messages=[{"role": msg.role, "content": msg.content} for msg in request.messages],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
            
            return ChatCompletionResponse(
                model=response.model,
                content=response.choices[0].message.content,
                provider="together",
                usage={
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(response.usage, "completion_tokens", None)
                }
            )
        except Exception as e:
            raise ProviderError(f"Together AI API error: {str(e)}")
        
    async def chat_complete_stream(self, request: ChatCompletionRequest) -> ChatCompletionGenerator:
        try:
            messages = []
            for msg in request.messages:
                role = "model" if msg.role == "assistant" else msg.role
                messages.append({"role": role, "parts": [msg.content]})

            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=request.model,
                messages=[{"role": msg.role, "content": msg.content} for msg in request.messages],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True
            )

            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    # Format as proper SSE
                    content = chunk.choices[0].delta.content
                    yield f"data: {json.dumps({'content': content})}\n\n"
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            raise ProviderError(f"Together AI API error (stream): {str(e)}")

    async def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        """
        Call Together AI's image generation endpoint.
        Since the Together API is synchronous, we run it in a thread to avoid blocking.
        """
        try:
            # Parse size to get width and height
            if request.size.value == "1024x1024":
                width, height = 1024, 1024
            elif request.size.value == "512x512":
                width, height = 512, 512
            elif request.size.value == "256x256":
                width, height = 256, 256
            else:
                width, height = 1024, 1024

            # Custom parameter for FLUX models
            steps = 4  # Default value for FLUX, can be adjusted if needed
            
            # Run the synchronous API call in a thread pool to avoid blocking
            response = await asyncio.to_thread(
                self.client.images.generate,
                prompt=request.prompt,
                model=request.model,
                width=width,
                height=height,
                steps=steps,
                n=request.n,
                response_format="b64_json"
            )
            
            # Convert b64_json responses to data URLs
            image_urls = []
            for image_data in response.data:
                # Create a data URL from the base64 data
                image_urls.append(f"data:image/png;base64,{image_data.b64_json}")
            
            return ImageGenerationResponse(
                urls=image_urls,
                model=request.model,
                provider="together"
            )
        except Exception as e:
            raise ProviderError(f"Together AI Image API error: {str(e)}")


