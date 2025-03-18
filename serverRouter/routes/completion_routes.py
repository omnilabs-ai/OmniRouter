from fastapi import APIRouter, Security, Depends
from fastapi.responses import StreamingResponse
from serverRouter.routes.utils import *
from serverRouter.core.datamodels import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
)
from serverRouter.core.models import CHAT_MODELS, IMAGE_MODELS
import json

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
) -> StreamingResponse:
    """Create a chat completion using the specified model."""
    model_name, provider = get_model_and_provider(request.model, CHAT_MODELS)
    request.model = model_name
    user_id = get_user_id_by_api_key(api_key)
    async def usage_tracking_wrapper():
        async for chunk in provider.chat_complete_stream(request):
            yield chunk
            if "event: usage" in chunk:
                data_part = chunk.split("data: ")[1].strip()
                if data_part:
                    usage_data = json.loads(data_part)
                    if "usage" in usage_data:
                        token_count = usage_data["usage"].get('total_tokens', 0)
                        print(f"Token count: {token_count}")
                        add_usage_to_user(user_id, token_count)

    return StreamingResponse(
        usage_tracking_wrapper(),
        media_type="text/event-stream",
    )
    
@router.post("/images/generate")
async def create_image(
    request: ImageGenerationRequest,
    api_key: str = Depends(verify_api_key)
) -> ImageGenerationResponse:
    """Generate images using the specified model."""
    model_name, provider = get_model_and_provider(request.model, IMAGE_MODELS)
    request.model = model_name
    return await provider.generate_image(request)