from typing import Dict, Any, AsyncGenerator
import anthropic
from serverRouter.core.interfaces import ChatProvider
from serverRouter.core.datamodels import ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChunk, ChatMessage
from serverRouter.core.exceptions import ProviderError
from dotenv import load_dotenv
import logging

load_dotenv()

class AnthropicProvider(ChatProvider):
    """Anthropic chat completion provider"""
    
    def __init__(self):
        """Initialize the Anthropic provider with API key from environment"""
        try:
            self.client = anthropic.AsyncAnthropic()
        except Exception as e:
            raise ProviderError(f"Failed to initialize Anthropic client: {str(e)}")
    
    async def chat_complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """
        Generate a chat completion using Anthropic's API
        
        Args:
            request: ChatCompletionRequest containing the input parameters
            
        Returns:
            ChatCompletionResponse containing the generated response
        """
        try:
            # Create the completion
            response = await self.client.messages.create(
                model=request.model,
                messages=[
                    {"role": msg.role, "content": msg.content}
                    for msg in request.messages
                ],
                max_tokens=request.max_tokens or 4092,
                temperature=request.temperature or 1.0
            )
            
            # Convert Anthropic response to our generic format
            return ChatCompletionResponse(
                model=response.model,
                content=response.content[0].text,
                provider="anthropic",
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }

            )
            
        except anthropic.APIError as e:
            raise ProviderError(f"Anthropic API error: {str(e)}")
        except Exception as e:
            raise ProviderError(f"Unexpected error: {str(e)}")
    async def chat_complete_stream(self, request: ChatCompletionRequest) -> AsyncGenerator[ChatCompletionChunk, None]:
        """
        Stream a chat completion using Anthropic's API
        
        Args:
            request: ChatCompletionRequest containing the input parameters
            
        Returns:
            AsyncGenerator yielding ChatCompletionChunk objects
        """
        try:
            # Create the streaming completion
            stream = await self.client.messages.create(
                model=request.model,
                messages=[
                    {"role": msg.role, "content": msg.content}
                    for msg in request.messages
                ],
                max_tokens=request.max_tokens or 4092,
                temperature=request.temperature or 1.0,
                stream=True
            )
            
            # Process the stream
            async for chunk in stream:
                if chunk.type == "content_block_delta" and chunk.delta.text:
                    # Stream content chunks
                    yield ChatCompletionChunk(
                        model=chunk.model or request.model,
                        content=chunk.delta.text,
                        provider="anthropic",
                        finish_reason=None
                    )
                elif chunk.type == "message_stop":
                    # Final chunk with stop reason
                    yield ChatCompletionChunk(
                        model=chunk.model or request.model,
                        content="",
                        provider="anthropic",
                        finish_reason="stop"
                    )
                    
        except anthropic.APIError as e:
            raise ProviderError(f"Anthropic streaming API error: {str(e)}")
        except Exception as e:
            logging.exception("Anthropic streaming error")
            raise ProviderError(f"Unexpected streaming error: {str(e)}")
