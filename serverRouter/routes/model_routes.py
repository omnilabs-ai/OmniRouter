from fastapi import APIRouter, Security, Depends
from serverRouter.routes.utils import verify_api_key
from serverRouter.core.models import MODELS, CHAT_MODELS, IMAGE_MODELS, REASONING_MODELS

router = APIRouter(prefix="/v1", tags=["models"])

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

@router.get("/models/reasoning")
async def list_reasoning_models(api_key: str = Depends(verify_api_key)):
    """List all available reasoning models with extended thinking capabilities"""
    return {
        "models": [
            {
                "id": model_id,
                **model_info.model_dump()
            }
            for model_id, model_info in REASONING_MODELS.items()
        ]
    }