from typing import List, Optional, Dict, Literal, Union, Any, AsyncGenerator
from pydantic import BaseModel, Field
from enum import Enum
from collections.abc import Mapping
from sse_starlette.sse import EventSourceResponse
from .function_registry import ToolChoice, FunctionDefinition, FunctionExecutionResult

# Type alias for streaming responses
ChatCompletionGenerator = EventSourceResponse

class ModelProvider(str, Enum):
    """Supported model providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"
    TOGETHER = "together"
    STABLEDIFFUSION = "stablediffusion"
    XAI = "xai"

## Chat Completion Models
class ChatMessage(BaseModel):
    """Represents a single message in a chat conversation"""
    role: str = Field(..., description="Role of the message sender (e.g. 'user', 'assistant', 'system')")
    content: str = Field(..., description="Content of the message")

class FunctionCall(BaseModel):
    """Function call information in a response"""
    name: str = Field(..., description="Name of the function to call")
    arguments: Dict[str, Any] = Field(..., description="Arguments for the function call")
    id: Optional[str] = Field(None, description="ID for the function call (for response tracking)")

class ToolResponseMessage(BaseModel):
    """Message containing a tool/function response"""
    role: Literal["tool"] = "tool"
    tool_call_id: str = Field(..., description="ID of the tool call this is responding to")
    content: str = Field(..., description="Content of the tool response")

class ChatCompletionRequest(BaseModel):
    """Input parameters for a chat completion request"""
    model: str = Field(..., description="Name of the model to use")
    messages: List[ChatMessage] = Field(..., description="List of messages in the conversation")
    temperature: float = Field(default=1.0, ge=0.0, le=2.0, description="Sampling temperature (0-2)")
    max_tokens: Optional[int] = Field(default=None, ge=1, description="Maximum number of tokens to generate")
    stream: bool = Field(default=False, description="Whether to stream the response")
    # Function calling parameters
    functions: Optional[List[Dict[str, Any]]] = Field(default=None, description="List of functions available to the model")
    tool_choice: Optional[Union[ToolChoice, str, Dict[str, Any]]] = Field(default=None, description="Controls function calling behavior")
    # Backward compatibility with OpenAI
    tools: Optional[List[Dict[str, Any]]] = Field(default=None, description="List of tools (OpenAI format)")
    function_call: Optional[Union[str, Dict[str, Any]]] = Field(default=None, description="Legacy function call option")

class ChatCompletionResponse(BaseModel):
    """Response from a chat completion request"""
    model: str = Field(..., description="Name of the model used")
    content: str = Field(..., description="Generated content")
    provider: str = Field(..., description="Provider that generated the response")
    usage: Dict[str, int] = Field(
        default_factory=lambda: {"total_tokens": 0},
        description="Token usage statistics"
    )
    # Function calling response
    function_calls: Optional[List[FunctionCall]] = Field(default=None, description="Function calls made by the model")
    function_results: Optional[List[FunctionExecutionResult]] = Field(default=None, description="Results of executed functions")

## Image Generation Models
class ImageSize(str, Enum):
    """Supported image sizes"""
    SMALL = "256x256"
    MEDIUM = "512x512"
    LARGE = "1024x1024"

class ImageGenerationRequest(BaseModel):
    """Input parameters for an image generation request"""
    prompt: str = Field(..., description="Text description of the desired image")
    model: str = Field(default="dall-e-3", description="Name of the model to use")
    size: ImageSize = Field(default=ImageSize.LARGE, description="Size of the generated image")
    quality: Literal["standard", "hd"] = Field(default="standard", description="Quality of the generated image")
    n: int = Field(default=1, ge=1, le=10, description="Number of images to generate")

class ImageGenerationResponse(BaseModel):
    """Response from an image generation request"""
    urls: List[str] = Field(..., description="URLs of the generated images")
    model: str = Field(..., description="Name of the model used")
    provider: str = Field(..., description="Provider that generated the images")


class BenchmarkScores(BaseModel, Mapping):
    MMLU: Optional[float] = Field(None, ge=0.0, le=1.0)
    GPQA: Optional[float] = Field(None, ge=0.0, le=1.0)
    HumanEval: Optional[float] = Field(None, ge=0.0, le=1.0)
    MATH: Optional[float] = Field(None, ge=0.0, le=1.0)
    BFCL: Optional[float] = Field(None, ge=0.0, le=1.0)
    MGSM: Optional[float] = Field(None, ge=0.0, le=1.0)

    class Config:
        validate_assignment = True

    def __getitem__(self, key: str) -> Optional[float]:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Optional[float]) -> None:
        setattr(self, key, value)

    def __iter__(self):
        return iter(self.__fields__)

    def __len__(self) -> int:
        return len(self.__fields__)

    def update(self, other: Dict[str, Optional[float]]) -> None:
        for key, value in other.items():
            self[key] = value

class ModelInfo(BaseModel):
    """Information about a model"""
    name: str = Field(..., description="Full name/version of the model")
    provider: ModelProvider = Field(..., description="Provider of the model")
    description: str = Field(..., description="Description of the model")
    max_tokens: Optional[int] = Field(None, description="Maximum context length")
    benchmarks: Optional[BenchmarkScores] = Field(
        default=None,
        description="Model benchmark scores"
    )
    tokenCost: Optional[float] = Field(
        default=None,
        description="Cost per 1000 tokens in USD"
    )
    latency: Optional[float] = Field(
        default=None,
        description="Average latency in seconds per request"
    )
    # Extended thinking parameters for Claude models
    extended_thinking: Optional[bool] = Field(
        default=False,
        description="Whether this model supports extended thinking mode"
    )
    thinking_threshold: Optional[float] = Field(
        default=0.5,
        description="Threshold to control when the model uses extended thinking"
    )
    thinking_budget: Optional[int] = Field(
        default=20000,
        description="Token budget for extended thinking process"
    )
    # Function calling capabilities
    supports_functions: Optional[bool] = Field(
        default=False,
        description="Whether this model supports function calling"
    )
    parallel_function_calling: Optional[bool] = Field(
        default=False,
        description="Whether this model supports parallel function calling"
    )

class SmartRouterRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., description="List of chat messages")
    max_latency: str = Field(..., description="Maximum latency preference (LIGHTNING, FAST, BALANCED, PERFORMANCE)")
    max_cost: str = Field(..., description="Maximum cost preference (CHEAP, BALANCED, PREMIUM, PERFORMANCE)")
    model_list: list = Field(..., description="List of models to consider (optional)")