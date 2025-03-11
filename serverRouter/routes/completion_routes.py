from fastapi import APIRouter, Security, Depends
from fastapi.responses import StreamingResponse
from serverRouter.routes.utils import verify_api_key, get_model_and_provider
from serverRouter.core.datamodels import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
)
from serverRouter.core.models import CHAT_MODELS, IMAGE_MODELS

router = APIRouter(prefix="/v1", tags=["completions"])

@router.post("/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    api_key: str = Depends(verify_api_key)
) -> ChatCompletionResponse:
    """Create a chat completion using the specified model."""
    model_name, provider = get_model_and_provider(request.model, CHAT_MODELS)
    request.model = model_name
    return await provider.chat_complete(request)

@router.post("/chat/completions/stream")
async def create_chat_completion_stream(
    request: ChatCompletionRequest,
    api_key: str = Depends(verify_api_key)
) -> StreamingResponse:
    """Create a chat completion using the specified model."""
    model_name, provider = get_model_and_provider(request.model, CHAT_MODELS)
    request.model = model_name
    return StreamingResponse(
        provider.chat_complete_stream(request),
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