# serverRouter/providers/togetherai/provider.py
import os
import asyncio
import logging
from typing import AsyncGenerator, Dict, Any, List

from serverRouter.core.interfaces import ChatProvider
from serverRouter.core.datamodels import (
    ChatCompletionRequest, 
    ChatCompletionResponse,
    ChatCompletionChunk
)
from serverRouter.core.exceptions import ProviderError

"""
FUNCTION CALLING / TOOLS IMPLEMENTATION NOTES:

* Tool Parameters Passing: We check if the request has tools, tool_choice, or 
  response_format parameters and pass them to the Together API if present.

* Tool Calls Extraction: We extract any tool calls from the response and format
  them consistently with our API response structure.

* Streaming Support: the streaming implementation have been adapted to handle tool
  calls in the stream similar to how the OpenAI provider does it.

https://docs.together.ai/reference/chat-completions-1

"""

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
            # Prepare base parameters
            params = {
                "model": request.model,
                "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
            }
            
            # Add tools if specified
            if hasattr(request, "tools") and request.tools:
                params["tools"] = request.tools
                
            # Add tool choice if specified
            if hasattr(request, "tool_choice") and request.tool_choice:
                params["tool_choice"] = request.tool_choice
                
            # Add response format if specified
            if hasattr(request, "response_format") and request.response_format:
                params["response_format"] = request.response_format
            
            # Run the synchronous API call in a thread pool to avoid blocking
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                **params
            )
            
            # Extract tool calls if present
            tool_calls = None
            if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in response.choices[0].message.tool_calls
                ]
                
            return ChatCompletionResponse(
                model=response.model,
                content=response.choices[0].message.content or "",  # Handle None content when only tool calls
                provider="together",
                tool_calls=tool_calls,
                usage={
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(response.usage, "completion_tokens", None)
                }
            )
        except Exception as e:
            logging.exception("Together AI API error")
            raise ProviderError(f"Together AI API error: {str(e)}")
            
    async def chat_complete_stream(self, request: ChatCompletionRequest) -> AsyncGenerator[ChatCompletionChunk, None]:
        """
        Stream chat completions from Together AI.
        Since the Together API's stream is synchronous, we need to process it carefully
        to avoid blocking the event loop.
        """
        try:
            # Prepare base parameters
            params = {
                "model": request.model,
                "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "stream": True
            }
            
            # Add tools if specified
            if hasattr(request, "tools") and request.tools:
                params["tools"] = request.tools
                
            # Add tool choice if specified
            if hasattr(request, "tool_choice") and request.tool_choice:
                params["tool_choice"] = request.tool_choice
                
            # Add response format if specified
            if hasattr(request, "response_format") and request.response_format:
                params["response_format"] = request.response_format
            
            # Create a streaming response using asyncio.to_thread to avoid blocking
            loop = asyncio.get_event_loop()
            
            # Get the stream iterator
            stream_iterator = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(**params)
            )
            
            # Variables to track tool calls
            current_tool_calls = []
            tool_call_chunks = {}  # Map from tool_call_id to accumulated chunks
            
            # Process the chunks one by one
            while True:
                try:
                    # Get next chunk in a non-blocking way
                    chunk = await loop.run_in_executor(None, next, stream_iterator, None)
                    if chunk is None:  # End of iterator
                        break
                    
                    # Check if the chunk has content or tool calls
                    if hasattr(chunk, 'choices') and chunk.choices:
                        choice = chunk.choices[0]
                        
                        # Handle tool call chunks (similar to OpenAI implementation)
                        if hasattr(choice, 'delta') and hasattr(choice.delta, 'tool_calls') and choice.delta.tool_calls:
                            for tool_call in choice.delta.tool_calls:
                                # Initialize tool call if new
                                tool_call_id = tool_call.id
                                if tool_call_id not in tool_call_chunks:
                                    tool_call_chunks[tool_call_id] = {
                                        "id": tool_call_id,
                                        "type": tool_call.type or "function",
                                        "function": {
                                            "name": tool_call.function.name or "",
                                            "arguments": tool_call.function.arguments or ""
                                        }
                                    }
                                else:
                                    # Update existing tool call
                                    if tool_call.function.name:
                                        tool_call_chunks[tool_call_id]["function"]["name"] = tool_call.function.name
                                    if tool_call.function.arguments:
                                        tool_call_chunks[tool_call_id]["function"]["arguments"] += tool_call.function.arguments
                                
                                # Add to current batch of tool calls
                                current_tool_calls = list(tool_call_chunks.values())
                                
                                # Yield a chunk with the updated tool calls
                                yield ChatCompletionChunk(
                                    model=chunk.model,
                                    content="",
                                    provider="together",
                                    finish_reason=None,
                                    tool_calls=current_tool_calls
                                )
                                continue
                        
                        # Handle content chunks
                        if hasattr(choice, 'delta') and hasattr(choice.delta, 'content') and choice.delta.content:
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
                                finish_reason=choice.finish_reason,
                                tool_calls=current_tool_calls if current_tool_calls else None
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
                finish_reason="stop",
                tool_calls=current_tool_calls if current_tool_calls else None
            )
            
        except Exception as e:
            logging.exception("Together AI streaming error")
            raise ProviderError(f"Together AI streaming API error: {str(e)}")
    
