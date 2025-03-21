from fastapi import APIRouter, Depends
from serverRouter.routes.utils import *
from serverRouter.core.datamodels import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
)
from serverRouter.core.models import CHAT_MODELS, IMAGE_MODELS
from sse_starlette.sse import EventSourceResponse

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
    response = await provider.chat_complete(request)
    token_count = response.usage['total_tokens']
    add_usage_to_user(user_id, token_count)
    return response

@router.post("/chat/completions/stream")
async def create_chat_completion_stream(
    request: ChatCompletionRequest,
    api_key: str = Depends(verify_api_key)
):
    """Create a chat completion using the specified model."""
    model_name, provider = get_model_and_provider(request.model, CHAT_MODELS)
    request.model = model_name
    user_id = get_user_id_by_api_key(api_key)
    
    response = await provider.chat_complete_stream(request)

    async def usage_tracking_generator():
        async for chunk in response.body_iterator:
            yield chunk
            if chunk.get("event") == "usage":
                usage_data = chunk.get("data", {})
                total_tokens = usage_data.get("total_tokens", 0)
                add_usage_to_user(user_id, total_tokens)
    
    return EventSourceResponse(usage_tracking_generator())
    

@router.post("/images/generate")
async def create_image(
    request: ImageGenerationRequest,
    api_key: str = Depends(verify_api_key)
) -> ImageGenerationResponse:
    """Generate images using the specified model."""
    model_name, provider = get_model_and_provider(request.model, IMAGE_MODELS)
    request.model = model_name
    return await provider.generate_image(request)