from typing import Dict, Any, AsyncGenerator, List
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
            # Base parameters
            params = {
                "model": request.model,
                "messages": [
                    {"role": msg.role, "content": msg.content}
                    for msg in request.messages
                ],
                "max_tokens": request.max_tokens or 4092,
                "temperature": request.temperature or 1.0
            }
            
            # Handle tools and function calling for Claude
            if hasattr(request, "tools") and request.tools:
                # Convert OpenAI-style tools to Claude format
                claude_tools = self._convert_tools_to_claude_format(request.tools)
                params["tools"] = claude_tools
                
                # Handle tool choice if specified
                if hasattr(request, "tool_choice") and request.tool_choice:
                    # Claude doesn't have direct tool_choice parameter like OpenAI
                    # But we can handle some cases
                    if request.tool_choice == "none":
                        # Remove tools to prevent their use
                        params.pop("tools", None)
            
            # Handle response format if specified
            if hasattr(request, "response_format") and request.response_format:
                if request.response_format.get("type") == "json_object":
                    # Claude has a new parameter for forcing JSON output
                    params["system"] = params.get("system", "") + "\nYou must respond with JSON only."
            
            # Create the completion
            response = await self.client.messages.create(**params)
            
            # Extract tool calls if present (Claude returns them differently from OpenAI)
            tool_calls = []
            if hasattr(response, "content") and response.content:
                for block in response.content:
                    if block.type == "tool_use":
                        tool_call = {
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": block.input
                            }
                        }
                        tool_calls.append(tool_call)
            
            # Get the text content
            content = ""
            for block in response.content:
                if block.type == "text":
                    content += block.text
            
            # Convert Anthropic response to our generic format
            return ChatCompletionResponse(
                model=response.model,
                content=content,
                provider="anthropic",
                tool_calls=tool_calls if tool_calls else None,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            )
            
        except anthropic.APIError as e:
            raise ProviderError(f"Anthropic API error: {str(e)}")
        except Exception as e:
            logging.exception("Error in Anthropic provider")
            raise ProviderError(f"Unexpected error: {str(e)}")
    
    def _convert_tools_to_claude_format(self, openai_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert OpenAI-style tools to Claude's expected format
        
        Args:
            openai_tools: List of tools in OpenAI format
            
        Returns:
            List of tools in Claude format
        """
        claude_tools = []
        
        for tool in openai_tools:
            if tool["type"] == "function":
                claude_tool = {
                    "name": tool["function"]["name"],
                    "description": tool["function"].get("description", ""),
                    "input_schema": tool["function"]["parameters"]
                }
                claude_tools.append(claude_tool)
        
        return claude_tools
            
    async def chat_complete_stream(self, request: ChatCompletionRequest) -> AsyncGenerator[ChatCompletionChunk, None]:
        """
        Stream a chat completion using Anthropic's API
        
        Args:
            request: ChatCompletionRequest containing the input parameters
            
        Returns:
            AsyncGenerator yielding ChatCompletionChunk objects
        """
        try:
            # Base parameters
            params = {
                "model": request.model,
                "messages": [
                    {"role": msg.role, "content": msg.content}
                    for msg in request.messages
                ],
                "max_tokens": request.max_tokens or 4092,
                "temperature": request.temperature or 1.0,
                "stream": True
            }
            
            # Handle tools and function calling for Claude
            if hasattr(request, "tools") and request.tools:
                # Convert OpenAI-style tools to Claude format
                claude_tools = self._convert_tools_to_claude_format(request.tools)
                params["tools"] = claude_tools
                
                # Handle tool choice if specified
                if hasattr(request, "tool_choice") and request.tool_choice:
                    # Claude doesn't have direct tool_choice parameter like OpenAI
                    if request.tool_choice == "none":
                        # Remove tools to prevent their use
                        params.pop("tools", None)
            
            # Handle response format if specified
            if hasattr(request, "response_format") and request.response_format:
                if request.response_format.get("type") == "json_object":
                    # Claude has a new parameter for forcing JSON output
                    params["system"] = params.get("system", "") + "\nYou must respond with JSON only."
            
            # Create the streaming completion
            stream = await self.client.messages.create(**params)
            
            # Process the stream
            tool_calls_started = False
            current_tool_call = None
            
            async for chunk in stream:
                if chunk.type == "content_block_start":
                    if chunk.content_block.type == "tool_use":
                        tool_calls_started = True
                        current_tool_call = {
                            "id": chunk.content_block.id,
                            "name": chunk.content_block.name,
                            "input": ""
                        }
                elif chunk.type == "content_block_delta" and chunk.delta.text:
                    if not tool_calls_started:
                        # Normal content chunk
                        yield ChatCompletionChunk(
                            model=chunk.model or request.model,
                            content=chunk.delta.text,
                            provider="anthropic",
                            finish_reason=None
                        )
                    else:
                        # Tool call input being streamed - we don't yield these as content
                        # but accumulate them for the tool call
                        if current_tool_call:
                            current_tool_call["input"] += chunk.delta.text
                elif chunk.type == "content_block_stop":
                    if tool_calls_started and current_tool_call:
                        # We've completed a tool call - yield a special chunk that indicates a tool call
                        tool_call_obj = {
                            "id": current_tool_call["id"],
                            "type": "function",
                            "function": {
                                "name": current_tool_call["name"],
                                "arguments": current_tool_call["input"]
                            }
                        }
                        yield ChatCompletionChunk(
                            model=chunk.model or request.model,
                            content="",
                            provider="anthropic",
                            finish_reason=None,
                            tool_calls=[tool_call_obj]
                        )
                        current_tool_call = None
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