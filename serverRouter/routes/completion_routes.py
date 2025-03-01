from fastapi import APIRouter, HTTPException, Security, Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.responses import StreamingResponse
import json
import asyncio
from serverRouter.core.config import VALID_API_KEYS, PROVIDERS, smart_router

from serverRouter.core.datamodels import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    ImageGenerationRequest,
    ImageGenerationResponse,
    SmartRouterRequest
)
from serverRouter.core.models import CHAT_MODELS, IMAGE_MODELS
from serverRouter.core.exceptions import ProviderError


router = APIRouter(prefix="/v1", tags=["completions"])

security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    if credentials.credentials not in VALID_API_KEYS:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid API key"
        )
    return credentials.credentials

@router.post("/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    api_key: str = Depends(verify_api_key)
):
    """Create a chat completion using the specified model."""
    try:
        # Check if streaming is requested
        if request.stream:
            # Use the streaming response
            return await create_chat_completion_stream(request, api_key)
        
        # For non-streaming requests, continue with regular completion
        # Look up the model info
        model_info = CHAT_MODELS.get(request.model)
        if not model_info:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {request.model}"
            )
        request.model = model_info.name

        # Get the provider for this model
        provider = PROVIDERS.get(model_info.provider)
        if not provider:
            raise HTTPException(
                status_code=500,
                detail=f"Provider not configured: {model_info.provider}"
            )

        # Create the completion
        response = await provider.chat_complete(request)
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def create_chat_completion_stream(
    request: ChatCompletionRequest,
    api_key: str
) -> StreamingResponse:
    """Create a streaming chat completion."""
    
    async def stream_generator():
        try:
            # Look up the model info
            model_info = CHAT_MODELS.get(request.model)
            if not model_info:
                error_json = json.dumps({
                    "error": {
                        "message": f"Unknown model: {request.model}",
                        "type": "invalid_request_error",
                        "code": 400
                    }
                })
                yield f"data: {error_json}\n\n"
                yield "data: [DONE]\n\n"
                return
                
            # Use the provider's full model name
            request.model = model_info.name

            # Get the provider for this model
            provider = PROVIDERS.get(model_info.provider)
            if not provider:
                error_json = json.dumps({
                    "error": {
                        "message": f"Provider not configured: {model_info.provider}",
                        "type": "server_error",
                        "code": 500
                    }
                })
                yield f"data: {error_json}\n\n"
                yield "data: [DONE]\n\n"
                return
                
            # Check if provider supports streaming
            if not await provider.supports_streaming():
                error_json = json.dumps({
                    "error": {
                        "message": f"Streaming not supported for provider: {model_info.provider}",
                        "type": "unsupported_operation",
                        "code": 400
                    }
                })
                yield f"data: {error_json}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Create and stream the completion
            async for chunk in provider.chat_complete_stream(request):
                chunk_json = json.dumps(chunk.model_dump())
                yield f"data: {chunk_json}\n\n"
                
                # Add a small delay to avoid overwhelming the client
                await asyncio.sleep(0.01)
                
            # Signal end of stream
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            error_json = json.dumps({
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "code": 500
                }
            })
            yield f"data: {error_json}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream"
    )

@router.post("/router/select-model-stream")
async def select_model_stream(
    request: SmartRouterRequest,
    api_key: str = Depends(verify_api_key)
):
    """Get model recommendations and stream the response."""
    try:
        result = smart_router.get_top_user_models(
            query=request.messages[-1].content,
            k=request.k,
            model_names=request.model_names,
            rel_cost=request.rel_cost,
            rel_latency=request.rel_latency,
            rel_accuracy=request.rel_accuracy
        )

        # Create a chat completion request with streaming enabled
        chat_request = ChatCompletionRequest(
            model=result["model"],
            messages=request.messages,
            stream=True,
            temperature=0.7,
            max_tokens=None
        )

        # Return the streaming response
        return await create_chat_completion_stream(chat_request, api_key)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/images/generate")
async def create_image(
    request: ImageGenerationRequest,
    api_key: str = Depends(verify_api_key)
) -> ImageGenerationResponse:
    """Generate images using the specified model."""
    try:
        # Look up the model info
        model_info = IMAGE_MODELS.get(request.model)
        if not model_info:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {request.model}"
            )
        request.model = model_info.name

        # Get the provider for this model
        provider = PROVIDERS.get(model_info.provider)
        if not provider:
            raise HTTPException(
                status_code=500,
                detail=f"Provider not configured: {model_info.provider}"
            )

        response = await provider.generate_image(request)
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/router/select-model")
async def select_model(
    request: SmartRouterRequest,
    api_key: str = Depends(verify_api_key)
):
    """Get model recommendations based on the query and preferences."""
    try:
        result = smart_router.get_top_user_models(
            query=request.messages[-1].content,
            k=request.k,
            model_names=request.model_names,
            rel_cost=request.rel_cost,
            rel_latency=request.rel_latency,
            rel_accuracy=request.rel_accuracy
        )

        chat_request = ChatCompletionRequest(
            model=result["model"],
            messages=request.messages,
        )
        
        response = await create_chat_completion(
            request=chat_request,
            api_key=api_key
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))