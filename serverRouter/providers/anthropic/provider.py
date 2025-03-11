from typing import Dict, Any
import anthropic
from serverRouter.core.interfaces import ChatProvider
from serverRouter.core.datamodels import ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ChatCompletionGenerator
from serverRouter.core.exceptions import ProviderError
from dotenv import load_dotenv
import json

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
    
    async def chat_complete_stream(self, request: ChatCompletionRequest) -> ChatCompletionGenerator:
        try:
            async with self.client.messages.stream(
                model=request.model,
                messages=[
                    {"role": msg.role, "content": msg.content}
                    for msg in request.messages
                ],
                max_tokens=request.max_tokens or 4092,
                temperature=request.temperature or 1.0
            ) as stream:
                async for chunk in stream.text_stream:
                    yield f"data: {json.dumps({'content': chunk})}\n\n"
                
            yield "data: [DONE]\n\n"

        except Exception as e:
            raise ProviderError(f"Unexpected error: {str(e)}")


