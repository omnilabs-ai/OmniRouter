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
from sse_starlette.sse import EventSourceResponse

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
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            )
        except Exception as e:
            raise ProviderError(f"Together AI API error: {str(e)}")
        
    async def chat_complete_stream(self, request: ChatCompletionRequest) -> ChatCompletionGenerator:
        async def event_generator():
            try:
                # Send initial metadata event
                yield {
                    "event": "metadata", 
                    "data": {
                        "model": request.model,
                        "provider": "together"
                    }
                }
                
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=request.model,
                    messages=[{"role": msg.role, "content": msg.content} for msg in request.messages],
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    stream=True
                )

                total_completion_tokens = 0
                
                for chunk in response:
                    print(chunk)
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        total_completion_tokens += 1
                        yield {
                            "event": "content",
                            "data": {"content": content}
                        }
                    if chunk.usage is not None:
                        yield {
                            "event": "usage",
                            "data": {"usage": {
                                "prompt_tokens": chunk.usage.prompt_tokens,
                                "completion_tokens": chunk.usage.completion_tokens,
                                "total_tokens": chunk.usage.total_tokens
                            }}
                        }
                
            except Exception as e:
                # Send error event in case of exception
                yield {
                    "event": "error",
                    "data": {"error": str(e)}
                }
                raise ProviderError(f"Together AI API error (stream): {str(e)}")
        
        return EventSourceResponse(event_generator())

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


