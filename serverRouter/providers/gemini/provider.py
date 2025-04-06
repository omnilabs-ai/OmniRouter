# serverRouter/providers/gemini/provider.py
from typing import Dict, Any, List, Union
from google import generativeai as genai
import os
from typing import Dict, Any, List, Union
from serverRouter.core.interfaces import ChatProvider
from serverRouter.core.datamodels import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionGenerator,
    FunctionCall,
    FunctionExecutionResult
)
from serverRouter.core.exceptions import ProviderError
from serverRouter.core.function_registry import ProviderType, function_registry
from dotenv import load_dotenv
import json
from sse_starlette.sse import EventSourceResponse

load_dotenv()

class GeminiProvider(ChatProvider):
    def __init__(self, api_key: str = None):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ProviderError("No GEMINI_API_KEY provided. Please add it to your .env file.")
        genai.configure(api_key=api_key)

    @property
    def provider_type(self) -> ProviderType:
        """Get the provider type for this implementation"""
        return ProviderType.GEMINI
        
    async def chat_complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        try:
            messages = []
            for msg in request.messages:
                role = "model" if msg.role == "assistant" else msg.role
                messages.append({"role": role, "parts": [msg.content]})

            model = genai.GenerativeModel(model_name=request.model)

            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=request.max_tokens or 2048,
                temperature=request.temperature or 1.0
            )
            
            # Add function calling parameters if provided
            function_declarations = None
            if request.functions:
                try:
                    # Convert functions to Gemini format using our registry
                    function_declarations = function_registry.get_for_provider(
                        ProviderType.GEMINI, 
                        [f["name"] for f in request.functions] if isinstance(request.functions, list) else None
                    )
                except Exception as e:
                    raise ProviderError(f"Failed to convert functions to Gemini format: {str(e)}")
            elif request.tools:
                # If tools are provided in request (assuming Gemini format)
                function_declarations = request.tools

            # Use synchronous version for simpler operation
            if function_declarations:
                response = model.generate_content(
                    contents=messages,
                    generation_config=generation_config,
                    tools=function_declarations
                )
            else:
                response = model.generate_content(
                    contents=messages,
                    generation_config=generation_config
                )

            # Extract content and function calls
            content = ""
            function_calls = []
            
            if response:
                if hasattr(response, 'text') and response.text:
                    content = response.text
                
                # Parse function calls if any
                function_calls = await self.parse_function_calls(response)
                
                return ChatCompletionResponse(
                    model=request.model,
                    content=content,
                    provider="gemini",
                    usage={
                        "prompt_tokens": response.usage_metadata.prompt_token_count,
                        "completion_tokens": response.usage_metadata.candidates_token_count,
                        "total_tokens": response.usage_metadata.prompt_token_count + response.usage_metadata.candidates_token_count
                    },
                    function_calls=function_calls if function_calls else None
                )
            else:
                raise ProviderError("Empty response from Gemini API")
                
        except Exception as e:
            raise ProviderError(f"Gemini API error (chat): {str(e)}")
        
    async def chat_complete_stream(self, request: ChatCompletionRequest) -> ChatCompletionGenerator:
        async def event_generator():
            try:
                messages = []
                for msg in request.messages:
                    role = "model" if msg.role == "assistant" else msg.role
                    messages.append({"role": role, "parts": [msg.content]})

                model = genai.GenerativeModel(model_name=request.model)
                
                # Send metadata event at the beginning
                yield {
                    "event": "metadata",
                    "data": json.dumps({
                        "model": request.model,
                        "provider": "gemini"
                    })
                }
                
                # Configure generation parameters
                generation_config = genai.types.GenerationConfig(
                    max_output_tokens=request.max_tokens or 2048,
                    temperature=request.temperature or 1.0
                )
                
                # Add function calling parameters if provided
                function_declarations = None
                if request.functions:
                    try:
                        # Convert functions to Gemini format using our registry
                        function_declarations = function_registry.get_for_provider(
                            ProviderType.GEMINI, 
                            [f["name"] for f in request.functions] if isinstance(request.functions, list) else None
                        )
                    except Exception as e:
                        yield {
                            "event": "error",
                            "data": json.dumps({"error": f"Failed to convert functions to Gemini format: {str(e)}"})
                        }
                        return
                elif request.tools:
                    # If tools are provided in request (assuming Gemini format)
                    function_declarations = request.tools
                
                # Start stream with or without function declarations
                if function_declarations:
                    response = model.generate_content(
                        contents=messages,
                        generation_config=generation_config,
                        tools=function_declarations,
                        stream=True
                    )
                else:
                    response = model.generate_content(
                        contents=messages,
                        generation_config=generation_config,
                        stream=True
                    )
                
                total_prompt_tokens = 0
                total_completion_tokens = 0
                function_calls = []

                for chunk in response:
                    # Process text content
                    if hasattr(chunk, 'text') and chunk.text:
                        total_prompt_tokens = chunk.usage_metadata.prompt_token_count
                        total_completion_tokens = chunk.usage_metadata.candidates_token_count
                        yield {
                            "event": "content",
                            "data": json.dumps({"content": chunk.text})
                        }
                    
                    # Process function calls
                    # Note: Gemini typically doesn't stream function calls but returns them at the end
                    if hasattr(chunk, 'candidates') and chunk.candidates:
                        for candidate in chunk.candidates:
                            if hasattr(candidate, 'content') and candidate.content:
                                for part in candidate.content.parts:
                                    if hasattr(part, 'function_call'):
                                        try:
                                            function_call = part.function_call
                                            func_name = function_call.name
                                            
                                            # Try to parse arguments
                                            args = {}
                                            if hasattr(function_call, 'args'):
                                                if isinstance(function_call.args, str):
                                                    try:
                                                        args = json.loads(function_call.args)
                                                    except json.JSONDecodeError:
                                                        args = {"raw_input": function_call.args}
                                                else:
                                                    args = function_call.args
                                                    
                                            # Generate a unique ID
                                            func_id = f"gemini_func_{len(function_calls)}"
                                            
                                            # Add to function calls
                                            function_calls.append({
                                                "id": func_id,
                                                "name": func_name,
                                                "arguments": args
                                            })
                                            
                                            # Send function call event
                                            yield {
                                                "event": "function_call",
                                                "data": json.dumps({
                                                    "id": func_id,
                                                    "name": func_name,
                                                    "arguments": args
                                                })
                                            }
                                            
                                            # Send function call complete event
                                            yield {
                                                "event": "function_call_complete",
                                                "data": json.dumps({
                                                    "id": func_id,
                                                    "name": func_name,
                                                    "arguments": args
                                                })
                                            }
                                        except Exception as e:
                                            # Log but continue processing
                                            print(f"Error processing Gemini function call: {str(e)}")

                # Send usage information at the end
                yield {
                    "event": "usage",
                    "data": json.dumps({
                        "prompt_tokens": total_prompt_tokens,
                        "completion_tokens": total_completion_tokens,
                        "total_tokens": total_prompt_tokens + total_completion_tokens
                    })
                }
                
            except Exception as e:
                error_message = str(e)
                yield {
                    "event": "error",
                    "data": json.dumps({"error": error_message})
                }
                raise ProviderError(f"Gemini API error (stream): {str(e)}")
        
        return EventSourceResponse(event_generator())
    
    async def parse_function_calls(self, raw_response: Any) -> List[FunctionCall]:
        """Parse function calls from Gemini response"""
        function_calls = []
        
        try:
            # Handle main response format
            if hasattr(raw_response, 'candidates') and raw_response.candidates:
                for candidate in raw_response.candidates:
                    if hasattr(candidate, 'content') and candidate.content:
                        for part in candidate.content.parts:
                            if hasattr(part, 'function_call'):
                                function_call = part.function_call
                                
                                # Extract function name
                                name = function_call.name if hasattr(function_call, 'name') else "unknown_function"
                                
                                # Parse arguments
                                arguments = {}
                                if hasattr(function_call, 'args'):
                                    if isinstance(function_call.args, str):
                                        try:
                                            arguments = json.loads(function_call.args)
                                        except json.JSONDecodeError:
                                            arguments = {"raw_input": function_call.args}
                                    else:
                                        arguments = function_call.args
                                
                                # Create a FunctionCall object
                                function_calls.append(FunctionCall(
                                    name=name,
                                    arguments=arguments,
                                    id=f"gemini_func_{len(function_calls)}"
                                ))
            
            # Alternative Gemini format
            elif hasattr(raw_response, 'function_call'):
                function_call = raw_response.function_call
                name = function_call.name if hasattr(function_call, 'name') else "unknown_function"
                
                arguments = {}
                if hasattr(function_call, 'args'):
                    if isinstance(function_call.args, str):
                        try:
                            arguments = json.loads(function_call.args)
                        except json.JSONDecodeError:
                            arguments = {"raw_input": function_call.args}
                    else:
                        arguments = function_call.args
                
                function_calls.append(FunctionCall(
                    name=name,
                    arguments=arguments,
                    id="gemini_func_0"
                ))
        
        except Exception as e:
            # Log but return what we have so far
            print(f"Error parsing Gemini function calls: {str(e)}")
        
        return function_calls
    
    async def create_function_response(self, function_results: List[FunctionExecutionResult]) -> Dict[str, Any]:
        """Create Gemini-compatible function response messages"""
        parts = []
        
        for result in function_results:
            # Format the result for Gemini
            function_response = {
                "name": result.function_name,
                "response": {}
            }
            
            if result.success:
                if isinstance(result.result, (dict, list)):
                    function_response["response"]["result"] = result.result
                else:
                    function_response["response"]["result"] = str(result.result)
            else:
                function_response["response"]["error"] = result.error
            
            parts.append({
                "function_response": function_response
            })
        
        return {
            "role": "function",
            "parts": parts
        }
        
        
