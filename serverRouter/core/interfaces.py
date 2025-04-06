from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict

from .datamodels import (
    ChatCompletionRequest, 
    ChatCompletionResponse, 
    ImageGenerationRequest, 
    ImageGenerationResponse, 
    ChatCompletionGenerator,
    FunctionCall,
    FunctionExecutionResult
)
from .function_registry import ProviderType

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
    
    @abstractmethod
    async def parse_function_calls(self, raw_response: Any) -> List[FunctionCall]:
        """
        Parse function calls from the provider's raw response
        """
        pass
    
    @abstractmethod
    async def create_function_response(self, function_results: List[FunctionExecutionResult]) -> Any:
        """
        Create a provider-specific response with function results
        """
        pass
    
    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        """
        Get the provider type for this implementation
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

