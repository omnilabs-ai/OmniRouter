from abc import ABC, abstractmethod
from .datamodels import ChatCompletionRequest, ChatCompletionResponse, ImageGenerationRequest, ImageGenerationResponse, ChatCompletionGenerator, ChatReasoningRequest, ChatReasoningResponse, ChatReasoningGenerator

class ChatProvider(ABC):
    """Abstract base class for chat completion providers"""
    
    @abstractmethod
    async def chat_complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """
        Generate a chat completion response for the given request
        """
        pass

    @abstractmethod
    async def chat_complete_stream(self, request: ChatCompletionRequest) -> ChatCompletionGenerator:
        """
        Stream a chat completion response for the given request
        """
        pass

class ReasoningProvider(ABC):
    """Abstract base class for reasoning-enabled chat providers"""
    
    @abstractmethod
    async def chat_reason_complete(self, request: ChatReasoningRequest) -> ChatReasoningResponse:
        """
        Generate a reasoning chat completion response for the given request.
        Includes detailed reasoning process before providing the final answer.
        """
        pass
        
    @abstractmethod
    async def chat_reason_complete_stream(self, request: ChatReasoningRequest) -> ChatReasoningGenerator:
        """
        Stream a reasoning chat completion response for the given request.
        Includes streamed reasoning process before providing the final answer.
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





