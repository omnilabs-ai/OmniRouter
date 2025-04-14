from typing import Dict, Any, Optional
import os
import json
import asyncio
from openai import AsyncOpenAI
from serverRouter.core.interfaces import ChatProvider, ReasoningProvider, ImageProvider
from serverRouter.core.datamodels import (
    ChatCompletionRequest, 
    ChatCompletionResponse,
    ChatCompletionGenerator,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ChatReasoningRequest,
    ChatReasoningResponse,
    ChatReasoningGenerator,
    ReasoningEffort,
    ReasoningTokenUsage
)
from serverRouter.core.exceptions import ProviderError
from dotenv import load_dotenv
from sse_starlette.sse import EventSourceResponse
import time
import re

load_dotenv()

class XAIProvider(ChatProvider, ReasoningProvider, ImageProvider):
    """XAI provider supporting chat, reasoning and image generation capabilities"""
    
    def __init__(self, api_key: str = None):
        """Initialize the XAI provider with API key from environment"""
        try:
            # Get API key from parameter or environment
            api_key = api_key or os.getenv("XAI_API_KEY")
            if not api_key:
                raise ProviderError("XAI_API_KEY not set in environment.")
            
            # Check if API key has the correct format for x.ai
            if not api_key.startswith("xai-"):
                print(f"Warning: XAI API key doesn't start with 'xai-' prefix, this may cause issues")
            
            # Create the client with explicit base URL to ensure correct API endpoint
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.x.ai/v1"
            )
            
            print(f"XAI provider initialized successfully with API key: {api_key[:7]}...")
            
        except Exception as e:
            raise ProviderError(f"Failed to initialize XAI client: {str(e)}")

    async def chat_complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Generate a chat completion using XAI models"""
        try:
            response = await self.client.chat.completions.create(
                model=request.model,
                messages=[
                    {"role": msg.role, "content": msg.content}
                    for msg in request.messages
                ],
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            # Extract usage info
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            # Create response
            return ChatCompletionResponse(
                model=response.model,
                content=response.choices[0].message.content,
                provider="xai",
                usage=usage
            )
        except Exception as e:
            raise ProviderError(f"XAI API error: {str(e)}")
    
    async def chat_complete_stream(self, request: ChatCompletionRequest) -> ChatCompletionGenerator:
        """Stream a chat completion using XAI models with the standard AsyncOpenAI client"""
        async def event_generator():
            usage_data = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            try:
                # Send initial metadata
                yield {
                    "event": "metadata",
                    "data": json.dumps({
                        "model": request.model,
                        "provider": "xai",
                        "created": int(time.time())
                    })
                }
                
                # Create the stream using the standard async client
                stream = await self.client.chat.completions.create(
                    model=request.model,
                    messages=[
                            {"role": msg.role, "content": msg.content}
                            for msg in request.messages
                    ],
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    stream_options={"include_usage": True}, # Request usage info in the stream
                            stream=True
                        )
                        
                # Process the stream chunks
                async for chunk in stream:
                    # Check for content delta
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if delta and delta.content is not None:
                            yield {
                                                        "event": "content",
                                "data": json.dumps({"content": delta.content})
                            }
                    
                    # Check for usage information (often comes in the last chunk)
                    if hasattr(chunk, 'usage') and chunk.usage is not None:
                        usage_data = {
                            "prompt_tokens": chunk.usage.prompt_tokens or 0,
                            "completion_tokens": chunk.usage.completion_tokens or 0,
                            "total_tokens": chunk.usage.total_tokens or 0
                        }
                        # We capture the latest usage, but yield it only at the end

                # Yield final usage after stream is complete
                    yield {
                        "event": "usage",
                    "data": json.dumps(usage_data)
                }

            except Exception as e:
                error_message = str(e)
                yield {
                    "event": "error",
                    "data": json.dumps({"error": error_message})
                }
                print(f"XAI API streaming error: {error_message}")
                import traceback
                print(traceback.format_exc())
                # Re-raise potentially? Or just log and send error event?
                # For now, send error event and log.
        
        return EventSourceResponse(event_generator())

    async def chat_reason_complete(self, request: ChatReasoningRequest) -> ChatReasoningResponse:
        """Generate a reasoning chat completion using XAI models (non-streaming)"""
        try:
            # Check if model is a reasoning model
            if not self._is_reasoning_model(request.model):
                raise ProviderError(f"Model {request.model} does not support reasoning. Use grok-3-mini-beta or grok-3-mini-fast-beta.")
            
            # Map our reasoning effort to XAI's reasoning_effort (only low/high supported)
            reasoning_effort = "low"
            if request.reasoning_effort in [ReasoningEffort.MEDIUM, ReasoningEffort.HIGH]:
                reasoning_effort = "high"
            
            # Call with reasoning_effort parameter to enable thinking
            response = await self.client.chat.completions.create(
                model=self._get_base_model_name(request.model),  # Strip suffix if present
                messages=[
                    {"role": msg.role, "content": msg.content}
                    for msg in request.messages
                ],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                reasoning_effort=reasoning_effort
            )
            
            # Extract usage, including reasoning tokens if available
            reasoning_tokens = 0
            visible_tokens = 0
            input_tokens = response.usage.prompt_tokens or 0
            total_tokens = response.usage.total_tokens or 0
            
            # XAI provides reasoning tokens in completion_tokens_details
            if hasattr(response.usage, "completion_tokens_details") and response.usage.completion_tokens_details:
                details = response.usage.completion_tokens_details
                if hasattr(details, "reasoning_tokens"):
                    reasoning_tokens = details.reasoning_tokens or 0
            
            # Calculate visible tokens based on total completion tokens and reasoning tokens
            completion_tokens = response.usage.completion_tokens or 0
            visible_tokens = max(0, completion_tokens - reasoning_tokens)

            # Access reasoning content if available (only in non-streaming)
            # reasoning_content = getattr(response.choices[0].message, "reasoning_content", "") # Not needed for response model

            return ChatReasoningResponse(
                model=response.model,
                content=response.choices[0].message.content or "",
                provider="xai",
                usage=ReasoningTokenUsage(
                    input_tokens=input_tokens,
                    output_tokens=visible_tokens, # Report calculated visible tokens
                    reasoning_tokens=reasoning_tokens, # Report reasoning tokens from details
                    total_tokens=total_tokens
                )
            )
        except Exception as e:
            raise ProviderError(f"XAI reasoning API error: {str(e)}")

    async def chat_reason_complete_stream(self, request: ChatReasoningRequest) -> ChatReasoningGenerator:
        """Stream a reasoning chat completion using XAI models (streams content only)"""
        async def event_generator():
            usage_info = ReasoningTokenUsage() # Initialize usage object
            try:
                # Check if model supports reasoning
                if not self._is_reasoning_model(request.model):
                    yield {"event": "error", "data": json.dumps({"message": f"Model {request.model} does not support reasoning mode"})}
                    return

                # Map reasoning effort
                reasoning_effort = "low"
                if request.reasoning_effort in [ReasoningEffort.MEDIUM, ReasoningEffort.HIGH]:
                    reasoning_effort = "high"

                # Initial metadata event
                yield {"event": "metadata", "data": json.dumps({
                    "model": request.model,
                    "created": int(time.time()),
                    "provider": "xai"
                    # System fingerprint might be in the stream later, can't predict it
                })}

                # Create the stream using the standard async client
                stream = await self.client.chat.completions.create(
                    model=self._get_base_model_name(request.model),
                    messages=[
                        {"role": msg.role, "content": msg.content}
                        for msg in request.messages
                    ],
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    reasoning_effort=reasoning_effort, # Include reasoning effort
                    stream_options={"include_usage": True}, # Request usage info
                    stream=True
                )

                # Process the stream chunks
                async for chunk in stream:
                    # Check for content delta
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if delta and delta.content is not None:
                            yield {
                                "event": "content",
                                "data": json.dumps({"content": delta.content})
                            }

                    # Check for usage information and update cumulative usage
                    if hasattr(chunk, 'usage') and chunk.usage is not None:
                        current_usage = chunk.usage
                        usage_info.input_tokens = current_usage.prompt_tokens or usage_info.input_tokens
                        usage_info.total_tokens = current_usage.total_tokens or usage_info.total_tokens
                        
                        completion_tokens = current_usage.completion_tokens or 0
                        reasoning_tokens = 0
                        
                        # Check if reasoning token details are provided in this usage chunk
                        if hasattr(current_usage, "completion_tokens_details") and current_usage.completion_tokens_details:
                            details = current_usage.completion_tokens_details
                            if hasattr(details, "reasoning_tokens"):
                                reasoning_tokens = details.reasoning_tokens or 0
                        
                        # Update reasoning tokens and calculate visible output tokens
                        usage_info.reasoning_tokens = reasoning_tokens
                        usage_info.output_tokens = max(0, completion_tokens - reasoning_tokens)
                        
                        # Note: We update usage_info but only yield the final usage event after the loop

                # Yield final usage after stream is complete
                yield {
                    "event": "usage",
                    "data": usage_info.model_dump_json() # Use model_dump_json for Pydantic v2
                }

            except Exception as e: # Correctly aligned with the 'try' block
                error_message = str(e)
                yield {
                    "event": "error",
                    "data": json.dumps({"error": error_message})
                }
                print(f"XAI API reasoning streaming error: {error_message}")
                import traceback
                print(traceback.format_exc())
        
        return EventSourceResponse(event_generator())

    async def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        """Generate images using XAI models"""
        try:
            # Ensure we're using a valid image model
            if not request.model.startswith("grok-2-image"):
                raise ProviderError(f"Model {request.model} is not a supported XAI image model. Use grok-2-image-1212.")
            
            # XAI doesn't use size or quality parameters, just prompt and n
            response = await self.client.images.generate(
                model=request.model,
                prompt=request.prompt,
                n=request.n,
                response_format="b64_json"  # Get base64 directly
            )
            
            # Convert base64 to data URLs
            data_urls = [
                f"data:image/jpg;base64,{image.b64_json}" 
                for image in response.data
            ]
            
            return ImageGenerationResponse(
                urls=data_urls,
                model=request.model,
                provider="xai"
            )
        except Exception as e:
            raise ProviderError(f"XAI image generation error: {str(e)}")
            
    def _is_reasoning_model(self, model_name: str) -> bool:
        # Only grok-3-mini-beta and grok-3-mini-fast-beta support reasoning
        base_model = self._get_base_model_name(model_name)
        # Check if the base model name contains 'grok-3-mini'
        return "grok-3-mini" in base_model.lower()

    def _get_base_model_name(self, model_name: str) -> str:
        """Get the base model name, stripping any reasoning suffix if needed"""
        # Reasoning models in XAI don't have suffixes like Claude, 
        # but check just in case user adds one.
        # Use the specific reasoning model names directly.
        if model_name in ["grok-3-mini-beta", "grok-3-mini-fast-beta"]:
             return model_name
        # Fallback for potential future naming conventions or user error
        if model_name.endswith("-reasoning"):
             return model_name.replace("-reasoning", "")
        return model_name 

    def _split_into_natural_chunks(self, text: str, target_chunk_size: int) -> list[str]:
        """Split text into natural chunks of approximately target_chunk_size words,
        trying to break at sentence boundaries when possible."""
        if not text:
            return []
            
        # First split by sentences
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            # Count words in this sentence
            words = sentence.split()
            sentence_word_count = len(words)
            
            # If adding this sentence would exceed target size and we already have content,
            # finalize the current chunk
            if current_word_count > 0 and current_word_count + sentence_word_count > target_chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_word_count = 0
            
            # Add the sentence to the current chunk
            current_chunk.append(sentence)
            current_word_count += sentence_word_count
            
            # If this single sentence was bigger than our target, just use it as its own chunk
            if sentence_word_count > target_chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_word_count = 0
        
        # Add any remaining content
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks 

    def _serialize_messages(self, messages):
        """Convert message objects to string for token estimation"""
        result = ""
        for msg in messages:
            result += f"{msg.role}: {msg.content}\n"
        return result
        
    def _estimate_tokens(self, text):
        """Roughly estimate the number of tokens in text"""
        if not text:
            return 0
        # Rough approximation: 1 token ~= 4 characters in English
        return max(1, len(text) // 4) 