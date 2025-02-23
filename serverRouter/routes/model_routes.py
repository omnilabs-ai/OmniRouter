from fastapi import APIRouter, HTTPException, Security, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from serverRouter.core.config import VALID_API_KEYS

from serverRouter.core.datamodels import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelProvider,
    ImageGenerationRequest,
    ImageGenerationResponse
)
from serverRouter.core.models import MODELS, CHAT_MODELS, IMAGE_MODELS

router = APIRouter(prefix="/v1", tags=["models"])

security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    if credentials.credentials not in VALID_API_KEYS:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid API key"
        )
    return credentials.credentials

@router.get("/models")
async def list_models(api_key: str = Depends(verify_api_key)):
    """List all available models"""
    return {
        "models": [
            {
                "id": model_id,
                **model_info.model_dump()
            }
            for model_id, model_info in MODELS.items()
        ]
    }

@router.get("/models/chat")
async def list_chat_models(api_key: str = Depends(verify_api_key)):
    """List all available chat models"""
    return {
        "models": [
            {
                "id": model_id,
                **model_info.model_dump()
            }
            for model_id, model_info in CHAT_MODELS.items()
        ]
    }

@router.get("/models/image")
async def list_image_models(api_key: str = Depends(verify_api_key)):
    """List all available image models"""
    return {
        "models": [
            {
                "id": model_id,
                **model_info.model_dump()
            }
            for model_id, model_info in IMAGE_MODELS.items()
        ]
    }