from fastapi import APIRouter, Depends, HTTPException
from serverRouter.routes.utils import *
from serverRouter.core.datamodels import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    FunctionCall,
    FunctionExecutionResult
)
from serverRouter.core.models import CHAT_MODELS, IMAGE_MODELS
from serverRouter.core.function_registry import function_registry, ProviderType
from sse_starlette.sse import EventSourceResponse
import json
from typing import List, Dict, Any

router = APIRouter(prefix="/v1", tags=["completions"])

@router.post("/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    api_key: str = Depends(verify_api_key)
) -> ChatCompletionResponse:
    """Create a chat completion using the specified model."""
    model_name, provider = get_model_and_provider(request.model, CHAT_MODELS)
    request.model = model_name
    user_id = get_user_id_by_api_key(api_key)
    
    # Process functions if provided or requested
    if request.functions:
        # Functions already provided in request
        pass
    elif request.tools:
        # Tools already provided in request (OpenAI format)
        pass
    else:
        # No functions specified, but handle auto_execute registered functions
        auto_functions = [func for func in function_registry.get_all_functions() if func.auto_execute]
        if auto_functions:
            # Convert auto-execute functions to provider format 
            request.functions = function_registry.get_for_provider(
                provider.provider_type, 
                [func.name for func in auto_functions]
            )
    
    # Call provider
    response = await provider.chat_complete(request)
    
    # Execute functions if requested and available
    if response.function_calls:
        function_results = []
        
        for func_call in response.function_calls:
            # Execute function
            result = function_registry.execute_function(func_call.name, func_call.arguments)
            function_results.append(result)
        
        # Store the results
        response.function_results = function_results
    
    # Track token usage
    token_count = response.usage['total_tokens']
    add_usage_to_user(user_id, token_count)
    
    return response

@router.post("/chat/completions/stream")
async def create_chat_completion_stream(
    request: ChatCompletionRequest,
    api_key: str = Depends(verify_api_key)
):
    """Create a streaming chat completion using the specified model."""
    model_name, provider = get_model_and_provider(request.model, CHAT_MODELS)
    request.model = model_name
    user_id = get_user_id_by_api_key(api_key)
    
    # Process functions if provided or requested
    if request.functions:
        # Functions already provided in request
        pass
    elif request.tools:
        # Tools already provided in request (OpenAI format)
        pass
    else:
        # No functions specified, but handle auto_execute registered functions
        auto_functions = [func for func in function_registry.get_all_functions() if func.auto_execute]
        if auto_functions:
            # Convert auto-execute functions to provider format 
            request.functions = function_registry.get_for_provider(
                provider.provider_type, 
                [func.name for func in auto_functions]
            )
    
    # Get streaming response
    response = await provider.chat_complete_stream(request)
    
    # Track pending function calls for execution
    pending_function_calls: List[Dict[str, Any]] = []
    
    async def function_handling_generator():
        # Track usage
        usage_tracked = False
        
        async for chunk in response.body_iterator:
            # Handle function calls
            if chunk.get("event") == "function_call_complete":
                data = json.loads(chunk.get("data", "{}"))
                
                # Add to pending function calls
                pending_function_calls.append({
                    "name": data.get("name", ""),
                    "arguments": data.get("arguments", {}),
                    "id": data.get("id", "function_call_0")
                })
                
                # Execute function
                result = function_registry.execute_function(
                    data.get("name", ""), 
                    data.get("arguments", {})
                )
                
                # Send function result
                yield {
                    "event": "function_result",
                    "data": json.dumps({
                        "function_name": result.function_name,
                        "id": data.get("id", "function_call_0"),
                        "success": result.success,
                        "result": result.result if result.success else None,
                        "error": result.error if not result.success else None
                    })
                }
            
            # Track usage
            elif chunk.get("event") == "usage" and not usage_tracked:
                usage_data = json.loads(chunk.get("data", "{}"))
                total_tokens = usage_data.get("total_tokens", 0)
                add_usage_to_user(user_id, total_tokens)
                usage_tracked = True
            
            # Pass through all other events
            yield chunk
    
    return EventSourceResponse(function_handling_generator())

@router.post("/images/generate")
async def create_image(
    request: ImageGenerationRequest,
    api_key: str = Depends(verify_api_key)
) -> ImageGenerationResponse:
    """Generate images using the specified model."""
    model_name, provider = get_model_and_provider(request.model, IMAGE_MODELS)
    request.model = model_name
    return await provider.generate_image(request)