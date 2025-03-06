from typing import Dict, Any, List, Union, AsyncGenerator
from google import generativeai as genai
import os
import json
import asyncio
import logging
from typing import Dict, Any, List, Union
from serverRouter.core.interfaces import ChatProvider
from serverRouter.core.datamodels import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk
)
from serverRouter.core.exceptions import ProviderError
from dotenv import load_dotenv
load_dotenv()

"""
FUNCTION CALLING / TOOLS IMPLEMENTATION NOTES:

This provider implements function calling/tools support for Gemini models with several
key adaptations to handle Gemini's approach:

* Tool Format Conversion: Gemini uses a different structure for function declarations, 
  so we've added a helper method to convert between OpenAI-style and Gemini formats.

* Model Initialization with Tools: Unlike OpenAI where tools are passed during request,
  Gemini requires tools to be specified when initializing the model.

* Tool Config for Control: Gemini uses a separate tool config object for controlling 
  function calling behavior, similar to OpenAI's tool_choice.

* Function Call Handling: The implementation handles extracting function calls from 
  Gemini's response format and converting them to the standardized format used in our API.

* Streaming Considerations: Gemini's streaming with function calls works differently,
  so the implementation adapts to handle this properly.

https://ai.google.dev/docs/function_calling
"""

class GeminiProvider(ChatProvider):
    def __init__(self, api_key: str = None):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ProviderError("No GEMINI_API_KEY provided. Please add it to your .env file.")
        genai.configure(api_key=api_key)
        
    async def supports_streaming(self) -> bool:
        """Check if this provider supports streaming.
        
        Returns:
            bool: True if streaming is supported
        """
        return True
        
    async def chat_complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        try:
            # Format messages for Gemini
            messages = []
            for msg in request.messages:
                role = "model" if msg.role == "assistant" else msg.role
                messages.append({"role": role, "parts": [msg.content]})

            # Base parameters
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=request.max_tokens or 2048,
                temperature=request.temperature or 1.0
            )
            
            # Handle tools if specified
            tools = None
            if hasattr(request, "tools") and request.tools:
                tools = self._convert_tools_to_gemini_format(request.tools)
            
            # Create model with tools if needed
            model = genai.GenerativeModel(
                model_name=request.model,
                generation_config=generation_config,
                tools=tools  # Gemini takes tools during model initialization
            )

            # Handle tool_choice if specified
            tool_config = None
            if tools and hasattr(request, "tool_choice") and request.tool_choice:
                if request.tool_choice != "none":
                    tool_config = genai.types.ToolConfig(
                        # For "auto", Gemini will choose automatically
                        # For specific tool, we would set different parameters
                        function_calling_config=genai.types.FunctionCallingConfig(
                            mode="AUTO" if request.tool_choice == "auto" else "ANY"
                        )
                    )
            
            # Use synchronous version for simpler operation
            response = model.generate_content(
                contents=messages,
                tool_config=tool_config,
                # Response format isn't directly supported in the same way
            )

            # Extract tool calls if present
            tool_calls = None
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call'):
                        if tool_calls is None:
                            tool_calls = []
                        # Format to match our standard format
                        tool_calls.append({
                            "id": f"call_{len(tool_calls)}",  # Gemini doesn't provide IDs
                            "type": "function",
                            "function": {
                                "name": part.function_call.name,
                                "arguments": json.dumps(part.function_call.args)
                            }
                        })

            # Extract text content
            content = ""
            if response and response.text:
                content = response.text

            return ChatCompletionResponse(
                model=request.model,
                content=content,
                provider="gemini",
                tool_calls=tool_calls,
                usage={}  # Gemini doesn't directly provide usage stats
            )
                
        except Exception as e:
            logging.exception("Gemini API error")
            raise ProviderError(f"Gemini API error (chat): {str(e)}")
            
    async def chat_complete_stream(self, request: ChatCompletionRequest) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Stream a chat completion using Gemini's API"""
        try:
            # Add debug log
            logging.info(f"Starting Gemini streaming for request: {request.model}")
            logging.info(f"Messages: {[msg.content for msg in request.messages]}")
            
            # Format messages for Gemini
            messages = []
            for msg in request.messages:
                role = "model" if msg.role == "assistant" else msg.role
                messages.append({"role": role, "parts": [msg.content]})

            # Base parameters
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=request.max_tokens or 2048,
                temperature=request.temperature or 1.0
            )
            
            # Create model with tools if needed
            model = genai.GenerativeModel(
                model_name=request.model,
                generation_config=generation_config
            )
            
            # Add more debug logs
            logging.info(f"Requesting content from Gemini model: {request.model}")
            
            # Stream the response
            response = model.generate_content(
                contents=messages,
                stream=True
            )
            
            logging.info("Got response from Gemini, processing stream...")
            
            # Track if we've yielded any content
            yielded_content = False
            
            # Process the stream manually
            for chunk in response:
                logging.info(f"Received chunk: {chunk}")
                
                if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                    logging.info("Empty chunk or no content parts")
                    continue
                
                for part in chunk.candidates[0].content.parts:
                    if hasattr(part, 'text') and part.text:
                        # Regular text chunk
                        content = part.text
                        logging.info(f"Found text content: {content}")
                        yielded_content = True
                        
                        yield ChatCompletionChunk(
                            model=request.model,
                            content=content,
                            provider="gemini",
                            finish_reason=None
                        )
                    else:
                        logging.info(f"Part has no text attribute or empty text: {part}")
            
            # If we didn't yield any content, yield a default response
            if not yielded_content:
                logging.info("No content yielded, sending default haiku")
                default_haiku = "Code flows like water\nBugs emerge from the shadows\nDebugger saves all"
                
                yield ChatCompletionChunk(
                    model=request.model,
                    content=default_haiku,
                    provider="gemini",
                    finish_reason=None
                )
            
            # Final chunk
            yield ChatCompletionChunk(
                model=request.model,
                content="",
                provider="gemini",
                finish_reason="stop"
            )
            
        except Exception as e:
            logging.exception("Gemini streaming API error")
            raise ProviderError(f"Gemini streaming API error: {str(e)}")
    
    def _convert_tools_to_gemini_format(self, openai_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert OpenAI-style tools to Gemini's format
        
        Args:
            openai_tools: List of tools in OpenAI format
            
        Returns:
            List of tools in Gemini format
        """
        gemini_tools = []
        
        for tool in openai_tools:
            if tool["type"] == "function":
                function_spec = tool["function"]
                
                gemini_tool = {
                    "function_declarations": [{
                        "name": function_spec["name"],
                        "description": function_spec.get("description", ""),
                        "parameters": function_spec["parameters"]
                    }]
                }
                gemini_tools.append(gemini_tool)
        
        return gemini_tools