from typing import Dict, Any, List
from openai import AsyncOpenAI
import os
import asyncio
from serverRouter.core.interfaces import ChatProvider, ImageProvider
from serverRouter.core.datamodels import (
    ChatCompletionRequest, 
    ChatCompletionResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ChatCompletionGenerator,
    FunctionCall,
    FunctionExecutionResult
)
from serverRouter.core.exceptions import ProviderError
from serverRouter.core.function_registry import ProviderType
from dotenv import load_dotenv
import json
from sse_starlette.sse import EventSourceResponse
load_dotenv()

class XAIProvider(ChatProvider, ImageProvider):
    """xAI provider supporting Grok models for chat and image generation"""
    
    def __init__(self, api_key: str = None):
        """Initialize the xAI provider with API key from environment"""
        try:
            api_key = api_key or os.getenv("XAI_API_KEY")
            if not api_key:
                raise ProviderError("XAI_API_KEY not set in environment.")
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.x.ai/v1"
            )
        except Exception as e:
            raise ProviderError(f"Failed to initialize xAI client: {str(e)}")

    async def chat_complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Generate a chat completion response using xAI's Grok model"""
        try:
            # Map OpenRouter model names to Grok model names if needed
            model_name = self._map_model_name(request.model)
            
            # Create the OpenAI-compatible request
            response = await self.client.chat.completions.create(
                model=model_name,
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
                provider="xai",
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            )
        except Exception as e:
            raise ProviderError(f"xAI chat completion error: {str(e)}")

    async def chat_complete_stream(self, request: ChatCompletionRequest) -> ChatCompletionGenerator:
        """Stream a chat completion response using xAI's Grok model"""
        async def generate():
            try:
                # Map OpenRouter model names to Grok model names if needed
                model_name = self._map_model_name(request.model)
                
                # Create the OpenAI-compatible streaming request
                stream = await self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": msg.role, "content": msg.content}
                        for msg in request.messages
                    ],
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    stream=True
                )
                
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        # Format as SSE event
                        yield {
                            "event": "message",
                            "data": {
                                "content": chunk.choices[0].delta.content,
                                "provider": "xai"
                            }
                        }
                
                yield {
                    "event": "done",
                    "data": {"provider": "xai"}
                }
            except Exception as e:
                error_msg = f"xAI chat completion stream error: {str(e)}"
                yield {"event": "error", "data": error_msg}

        return EventSourceResponse(generate())

    async def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        """Generate an image using xAI's Grok image model"""
        try:
            # Map OpenRouter model names to Grok model names if needed
            model_name = "grok-2-image"
            if request.model == "grok-2-image-1212" or request.model == "grok-2-image":
                model_name = "grok-2-image"
            
            # Create the OpenAI-compatible image generation request
            response = await self.client.images.generate(
                model=model_name,
                prompt=request.prompt,
                n=request.n,
                response_format="b64_json"  # Get base64 encoded images
            )
            
            # Extract URLs from response
            urls = []
            for data in response.data:
                if hasattr(data, "b64_json"):
                    # Convert base64 to data URL
                    urls.append(f"data:image/jpeg;base64,{data.b64_json}")
            
            return ImageGenerationResponse(
                urls=urls,
                model=model_name,
                provider="xai"
            )
        except Exception as e:
            raise ProviderError(f"xAI image generation error: {str(e)}")
    
    @property
    def provider_type(self) -> ProviderType:
        """Get the provider type (required by ChatProvider). Not implemented for XAI."""
        # TODO: Implement if XAI supports function calling via this interface
        # or assign a specific ProviderType if needed for registration.
        raise NotImplementedError("Provider type not specified for XAIProvider.")

    async def parse_function_calls(self, raw_response: Any) -> List[FunctionCall]:
        """Parse function calls (required by ChatProvider). Not supported by XAIProvider."""
        # XAI via OpenAI client might support tools/functions, but not implemented here.
        return []

    async def create_function_response(self, function_results: List[FunctionExecutionResult]) -> Any:
        """Create function response (required by ChatProvider). Not supported by XAIProvider."""
        # XAI via OpenAI client might support tools/functions, but not implemented here.
        raise NotImplementedError("Function call response creation not implemented for XAIProvider.")

    def _map_model_name(self, model_name: str) -> str:
        """Map OpenRouter model names to xAI model names"""
        model_mapping = {
            "grok-2-1212": "grok-2-latest",
            "grok-2-vision-1212": "grok-2-vision-latest",
            "grok-2-image-1212": "grok-2-image"
        }
        return model_mapping.get(model_name, model_name)
