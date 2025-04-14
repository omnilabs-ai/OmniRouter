from fastapi import APIRouter, Depends
from serverRouter.routes.utils import *
from serverRouter.core.datamodels import (
    ChatReasoningRequest,
    ChatReasoningResponse,
    ReasoningTokenUsage
)
from serverRouter.core.models import REASONING_MODELS
from sse_starlette.sse import EventSourceResponse
import json

router = APIRouter(prefix="/v1", tags=["reasoning"])

@router.post("/reason/completions")
async def create_reasoning_completion(
    request: ChatReasoningRequest,
    api_key: str = Depends(verify_api_key)
) -> ChatReasoningResponse:
    """
    Create a reasoning completion using a model with extended thinking capabilities.
    This endpoint generates a response with detailed reasoning process.
    """
    try:
        model_name, provider = get_model_and_provider(request.model, REASONING_MODELS)
        request.model = model_name
        user_id = get_user_id_by_api_key(api_key)
        
        # Ensure the provider supports reasoning
        if not hasattr(provider, 'chat_reason_complete'):
            raise HTTPException(
                status_code=400,
                detail=f"Provider for model {model_name} does not support reasoning capabilities"
            )
        
        # Get the response with reasoning
        response = await provider.chat_reason_complete(request)
        
        # Track usage
        total_tokens = response.usage.total_tokens
        add_usage_to_user(user_id, total_tokens)
        
        return response
    except Exception as e:
        import traceback
        error_detail = f"Error in reasoning completion: {str(e)}\n{traceback.format_exc()}"
        print(f"REASONING ERROR: {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)

@router.post("/reason/completions/stream")
async def create_reasoning_completion_stream(
    request: ChatReasoningRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Stream a reasoning completion using a model with extended thinking capabilities.
    This endpoint streams both the reasoning process and final response.
    """
    try:
        model_name, provider = get_model_and_provider(request.model, REASONING_MODELS)
        request.model = model_name
        # user_id = get_user_id_by_api_key(api_key) # Usage tracking removed for now
        
        # Ensure the provider supports reasoning
        if not hasattr(provider, 'chat_reason_complete_stream'):
            raise HTTPException(
                status_code=400,
                detail=f"Provider for model {model_name} does not support streaming reasoning capabilities"
            )
        
        # Set stream to true for request
        request.stream = True
        
        # Get streaming response directly from provider
        response = await provider.chat_reason_complete_stream(request)
        return response

        # Removed usage tracking wrapper:
        # async def usage_tracking_generator():
        #     async for chunk in response.body_iterator:
        #         yield chunk
        #         if chunk.get("event") == "usage":
        #             usage_data = json.loads(chunk.get("data", {}))
        #             total_tokens = usage_data.get("total_tokens", 0)
        #             add_usage_to_user(user_id, total_tokens)
        # 
        # return EventSourceResponse(usage_tracking_generator())
    except Exception as e:
        import traceback
        error_detail = f"Error in streaming reasoning completion: {str(e)}\n{traceback.format_exc()}"
        print(f"REASONING ERROR (STREAM): {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail) 