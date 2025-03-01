# serverRouter/providers/togetherai/provider.py
import os
import asyncio
from typing import AsyncGenerator, Dict, Any
from serverRouter.core.interfaces import ChatProvider
from serverRouter.core.datamodels import (
    ChatCompletionRequest, 
    ChatCompletionResponse,
    ChatCompletionChunk
)
from serverRouter.core.exceptions import ProviderError
import logging

class TogetherAIProvider(ChatProvider):
    """
    Provider for Together AI.
    Uses the official Together library to call chat completions.
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
            
    async def chat_complete_stream(self, request: ChatCompletionRequest) -> AsyncGenerator[ChatCompletionChunk, None]:
        """
        Stream chat completions from Together AI.
        Since the Together API's stream is synchronous, we need to process it carefully
        to avoid blocking the event loop.
        """
        try:
            # Get all parameters ready
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
            
            # Create a streaming response using asyncio.to_thread to avoid blocking
            loop = asyncio.get_event_loop()
            
            # Get the stream iterator
            stream_iterator = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=request.model,
                    messages=messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    stream=True
                )
            )
            
            # Process the chunks one by one
            while True:
                try:
                    # Get next chunk in a non-blocking way
                    chunk = await loop.run_in_executor(None, next, stream_iterator, None)
                    if chunk is None:  # End of iterator
                        break
                    
                    # Check if the chunk has content
                    if hasattr(chunk, 'choices') and chunk.choices:
                        choice = chunk.choices[0]
                        
                        if hasattr(choice, 'delta') and choice.delta and hasattr(choice.delta, 'content') and choice.delta.content:
                            # Yield the content chunk
                            yield ChatCompletionChunk(
                                model=chunk.model,
                                content=choice.delta.content,
                                provider="together",
                                finish_reason=choice.finish_reason if hasattr(choice, 'finish_reason') else None
                            )
                        elif hasattr(choice, 'finish_reason') and choice.finish_reason:
                            # Final chunk with finish reason
                            yield ChatCompletionChunk(
                                model=chunk.model,
                                content="",
                                provider="together",
                                finish_reason=choice.finish_reason
                            )
                except StopIteration:
                    break
                except Exception as e:
                    logging.error(f"Error processing Together AI stream chunk: {str(e)}")
                    continue
            
            # Send final chunk with finish reason if needed
            yield ChatCompletionChunk(
                model=request.model,
                content="",
                provider="together",
                finish_reason="stop"
            )
            
        except Exception as e:
            logging.exception("Together AI streaming error")
            raise ProviderError(f"Together AI streaming API error: {str(e)}")
    
    async def supports_streaming(self) -> bool:
        """Check if this provider supports streaming"""
        # Together AI supports streaming via the OpenAI-compatible API
        return True