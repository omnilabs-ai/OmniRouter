from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from serverRouter.core.config import VALID_API_KEYS, PROVIDERS
from serverRouter.core.models import CHAT_MODELS, IMAGE_MODELS
from serverRouter.core.datamodels import ModelProvider

security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    if credentials.credentials not in VALID_API_KEYS:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid API key"
        )
    return credentials.credentials

def get_model_and_provider(model_id: str, models_dict):
    """Get model info and provider for a given model ID."""
    model_info = models_dict.get(model_id)
    if not model_info:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model_id}"
        )
    
    provider = PROVIDERS.get(model_info.provider)
    if not provider:
        raise HTTPException(
            status_code=500,
            detail=f"Provider not configured: {model_info.provider}"
        )
    
    return model_info.name, provider
