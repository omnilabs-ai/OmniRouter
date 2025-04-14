from typing import Dict, Any, Optional, Tuple
import anthropic
from serverRouter.core.interfaces import ChatProvider, ReasoningProvider
from serverRouter.core.datamodels import (
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, 
    ChatCompletionGenerator, ChatReasoningRequest, ChatReasoningResponse,
    ChatReasoningGenerator, ReasoningEffort, ReasoningTokenUsage
)
from serverRouter.core.exceptions import ProviderError
from dotenv import load_dotenv
import json
from sse_starlette.sse import EventSourceResponse
from serverRouter.core.models import MODELS

load_dotenv()

class AnthropicProvider(ChatProvider, ReasoningProvider):
    """Anthropic chat completion provider"""
    
    def __init__(self):
        """Initialize the Anthropic provider with API key from environment"""
        try:
            self.client = anthropic.AsyncAnthropic()
        except Exception as e:
            raise ProviderError(f"Failed to initialize Anthropic client: {str(e)}")
    
    def _safe_get_attr(self, obj: Any, attr_path: str, default: Any = None) -> Any:
        """Safely access nested object attributes"""
        if obj is None:
            return default
            
        parts = attr_path.split('.')
        current = obj
        
        try:
            for part in parts:
                if part.isdigit():
                    current = current[int(part)]
                elif hasattr(current, part):
                    current = getattr(current, part)
                else:
                    return default
            return current
        except (AttributeError, IndexError, TypeError):
            return default
    
    def _get_budget_for_effort(self, effort: ReasoningEffort, model_key: str) -> int:
        """Convert reasoning effort level to token budget"""
        default_budget = getattr(MODELS.get(model_key), 'thinking_budget', 20000)
        
        budget_map = {
            ReasoningEffort.LOW: max(1024, default_budget // 4),
            ReasoningEffort.MEDIUM: default_budget,
            ReasoningEffort.HIGH: min(default_budget * 2, 64000)
        }
        
        return budget_map.get(effort, default_budget)
    
    def _estimate_token_split(
        self, 
        output_tokens: int, 
        effort: ReasoningEffort,
        reasoning_tokens: int = 0, 
        content_tokens: int = 0
    ) -> Tuple[int, int]:
        """Calculate reasoning vs visible token counts"""
        # Use measured values if available
        if reasoning_tokens > 0 or content_tokens > 0:
            visible = content_tokens
            reasoning = reasoning_tokens or (output_tokens - visible)
            
            # Validate
            if visible > output_tokens:
                visible = output_tokens
                reasoning = 0
            return reasoning, visible
            
        # Otherwise estimate based on effort level
        ratio_map = {
            ReasoningEffort.LOW: 0.5,
            ReasoningEffort.MEDIUM: 0.7, 
            ReasoningEffort.HIGH: 0.8
        }
        ratio = ratio_map.get(effort, 0.7)
        
        reasoning = int(output_tokens * ratio)
        visible = output_tokens - reasoning
        
        if visible < 0:
            visible = 0
            reasoning = output_tokens
                
        return reasoning, visible
    
    async def chat_complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Generate a basic chat completion"""
        try:
            response = await self.client.messages.create(
                model=request.model,
                messages=[{"role": msg.role, "content": msg.content} for msg in request.messages],
                max_tokens=request.max_tokens or 4096,
                temperature=request.temperature or 1.0
            )
            
            return ChatCompletionResponse(
                model=response.model,
                content=self._safe_get_attr(response, "content.0.text", ""),
                provider="anthropic",
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                }
            )
            
        except anthropic.APIError as e:
            raise ProviderError(f"Anthropic API error: {str(e)}")
        except (AttributeError, IndexError) as e:
            raise ProviderError(f"Error parsing Anthropic response: {str(e)}")
        except Exception as e:
            raise ProviderError(f"Unexpected error: {str(e)}")
    
    async def chat_complete_stream(self, request: ChatCompletionRequest) -> EventSourceResponse:
        """Stream a basic chat completion"""
        async def generate():
            try:
                # Initial metadata
                yield {"event": "metadata", "data": json.dumps({
                    "model": request.model,
                    "provider": "anthropic"
                })}
                
                # Stream response
                async with self.client.messages.stream(
                    model=request.model,
                    messages=[{"role": msg.role, "content": msg.content} for msg in request.messages],
                    max_tokens=request.max_tokens or 4096,
                    temperature=request.temperature or 1.0
                ) as stream:
                    async for chunk in stream:
                        if chunk.type == "text":
                            yield {"event": "content", "data": json.dumps({"content": chunk.text})}
                        elif chunk.type == "message_stop":
                            yield {"event": "usage", "data": json.dumps({
                                "input_tokens": chunk.message.usage.input_tokens,
                                "output_tokens": chunk.message.usage.output_tokens,
                                "total_tokens": chunk.message.usage.input_tokens + chunk.message.usage.output_tokens
                            })}
            except Exception as e:
                yield {"event": "error", "data": json.dumps({"error": str(e)})}
                raise ProviderError(f"Streaming error: {str(e)}")
        
        return EventSourceResponse(generate())
    
    async def chat_reason_complete(self, request: ChatReasoningRequest) -> ChatReasoningResponse:
        """Generate a reasoning chat completion with extended thinking"""
        try:
            # Prepare parameters for request
            formatted_messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
            system_prompt = "Please think step-by-step when answering this question. Break down your reasoning process clearly, and show all your work."
            budget = self._get_budget_for_effort(request.reasoning_effort, request.model)
            
            # Ensure max_tokens is sufficient
            max_tokens = request.max_tokens or 4096
            if max_tokens <= budget:
                max_tokens = budget + 1000
            
            # Make API call
            response = await self.client.messages.create(
                model=request.model,
                messages=formatted_messages,
                system=system_prompt,
                max_tokens=max_tokens,
                temperature=1.0,  # Required for thinking
                thinking={"type": "enabled", "budget_tokens": budget}
            )
            
            # Extract visible content
            visible_content = ""
            if hasattr(response, "content") and response.content:
                # First look for text blocks
                for block in response.content:
                    if text := self._safe_get_attr(block, "text"):
                        visible_content = text
                        break
                
                # Fall back to first content item
                if not visible_content:
                    visible_content = self._safe_get_attr(response, "content.0.text", "")
            
            # Get token usage
            input_tokens = self._safe_get_attr(response, "usage.input_tokens", 0)
            output_tokens = self._safe_get_attr(response, "usage.output_tokens", 0)
            
            # Get thinking tokens (or estimate them)
            thinking_tokens = self._safe_get_attr(response, "usage.thinking_tokens", 0)
            if thinking_tokens:
                visible_tokens = max(0, output_tokens - thinking_tokens)
            else:
                thinking_tokens, visible_tokens = self._estimate_token_split(
                    output_tokens, request.reasoning_effort
                )
            
            # Prepare response
            return ChatReasoningResponse(
                model=response.model,
                content=visible_content,
                provider="anthropic",
                usage=ReasoningTokenUsage(
                    input_tokens=input_tokens,
                    output_tokens=visible_tokens,
                    reasoning_tokens=thinking_tokens,
                    total_tokens=input_tokens + output_tokens
                )
            )
            
        except Exception as e:
            import traceback
            raise ProviderError(f"Anthropic reasoning API error: {str(e)}\n{traceback.format_exc()}")
    
    async def chat_reason_complete_stream(self, request: ChatReasoningRequest) -> ChatReasoningGenerator:
        """Stream a reasoning chat completion with extended thinking"""
        async def generate():
            reasoning_tokens = 0
            content_tokens = 0
            in_thinking_block = False
            
            try:
                # Initial events
                yield {"event": "metadata", "data": json.dumps({
                    "model": request.model, 
                    "provider": "anthropic"
                })}
                
                yield {"event": "thinking_start", "data": json.dumps({
                    "message": "Starting reasoning process"
                })}
                
                # Prepare parameters
                formatted_messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
                system_prompt = "Please think step-by-step when answering this question. Break down your reasoning process clearly, and show all your work."
                budget = self._get_budget_for_effort(request.reasoning_effort, request.model)
                
                # Ensure max_tokens is sufficient
                max_tokens = request.max_tokens or 4096
                if max_tokens <= budget:
                    max_tokens = budget + 1000
                
                # Stream response
                async with self.client.messages.stream(
                    model=request.model,
                    messages=formatted_messages,
                    system=system_prompt,
                    max_tokens=max_tokens,
                    temperature=1.0,  # Required for thinking
                    thinking={"type": "enabled", "budget_tokens": budget}
                ) as stream:
                    async for event in stream:
                        # Handle event based on type
                        if event.type == "message_start":
                            yield {"event": "start", "data": json.dumps({"type": "message"})}
                        
                        elif event.type == "content_block_start" and hasattr(event, "content_block"):
                            block_type = getattr(event.content_block, "type", "")
                            
                            if block_type == "thinking":
                                in_thinking_block = True
                                yield {"event": "block_start", "data": json.dumps({"type": "reasoning"})}
                            elif block_type == "redacted_thinking":
                                yield {"event": "block_start", "data": json.dumps({"type": "redacted_reasoning"})}
                                
                                if hasattr(event.content_block, "data"):
                                    yield {"event": "redacted_reasoning", "data": json.dumps({
                                        "content": "This reasoning has been redacted for safety reasons."
                                    })}
                            else:
                                yield {"event": "block_start", "data": json.dumps({"type": block_type})}
                        
                        elif event.type == "content_block_delta" and hasattr(event, "delta"):
                            if in_thinking_block and hasattr(event.delta, "thinking"):
                                # Process thinking content
                                thinking_text = event.delta.thinking
                                reasoning_tokens += len(thinking_text.split()) * 1.3  # Estimate
                                yield {"event": "reasoning", "data": json.dumps({"content": thinking_text})}
                            elif hasattr(event.delta, "text"):
                                # Process visible content
                                text = event.delta.text
                                content_tokens += len(text.split()) * 1.3  # Estimate
                                yield {"event": "content", "data": json.dumps({"content": text})}
                        
                        elif event.type == "content_block_stop":
                            if in_thinking_block:
                                in_thinking_block = False
                                yield {"event": "block_stop", "data": json.dumps({"type": "reasoning"})}
                            else:
                                yield {"event": "block_stop", "data": json.dumps({})}
                        
                        elif event.type == "message_stop" and hasattr(event, "message"):
                            # Calculate final token usage
                            input_tokens = self._safe_get_attr(event.message, "usage.input_tokens", 0)
                            output_tokens = self._safe_get_attr(event.message, "usage.output_tokens", 0)
                            
                            # Get token split
                            reasoning_final, visible = self._estimate_token_split(
                                output_tokens, 
                                request.reasoning_effort,
                                reasoning_tokens,
                                content_tokens
                            )
                            
                            # Send usage information
                            yield {"event": "usage", "data": json.dumps({
                                "input_tokens": input_tokens,
                                "output_tokens": visible,
                                "reasoning_tokens": reasoning_final,
                                "total_tokens": input_tokens + output_tokens
                            })}
                
            except Exception as e:
                import traceback
                yield {"event": "error", "data": json.dumps({
                    "error": f"Streaming error: {str(e)}"
                })}
        
        return EventSourceResponse(generate())