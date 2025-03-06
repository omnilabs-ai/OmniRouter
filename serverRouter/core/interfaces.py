from abc import ABC, abstractmethod
from typing import AsyncGenerator  # This import is missing
from .datamodels import (
    ChatCompletionRequest, 
    ChatCompletionResponse, 
    ChatCompletionChunk, 
    ImageGenerationRequest, 
    ImageGenerationResponse
)

class ChatProvider(ABC):
    """Abstract base class for chat completion providers"""
    
    @abstractmethod
    async def chat_complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """
        Generate a chat completion response for the given request
        """
        pass
    
    @abstractmethod
    async def chat_complete_stream(self, request: ChatCompletionRequest) -> AsyncGenerator[ChatCompletionChunk, None]:
        """
        Stream a chat completion response for the given request
        
        Args:
            request: ChatCompletionRequest containing the input parameters
            
        Returns:
            AsyncGenerator yielding ChatCompletionChunk objects
        """
        pass

class ImageProvider(ABC):
    """Abstract base class for image generation providers"""
    
    @abstractmethod
    async def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        """
        Generate an image based on the given request
        """
        pass