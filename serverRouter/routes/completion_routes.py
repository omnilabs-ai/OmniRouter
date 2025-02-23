from fastapi import APIRouter, HTTPException, Security, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from serverRouter.core.config import VALID_API_KEYS, PROVIDERS, smart_router

from serverRouter.core.datamodels import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    SmartRouterRequest
)
from serverRouter.core.models import CHAT_MODELS, IMAGE_MODELS

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
) -> ChatCompletionResponse:
    """Create a chat completion using the specified model."""
    try:
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

        response = await create_chat_completion(
            request=ChatCompletionRequest(
                model=result["model"],
                messages=request.messages,
            )
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 