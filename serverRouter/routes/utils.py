from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from serverRouter.core import config
from datetime import datetime
security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    if credentials.credentials not in config.VALID_API_KEYS:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid API key"
        )
    
    user_id = get_user_id_by_api_key(credentials.credentials)
    user_usage = get_user_usage(user_id)
    if user_usage['total_tokens'] >= config.MAX_TOKENS:
        raise HTTPException(
            status_code=429,
            detail="User has reached the maximum number of tokens"
        )
    return credentials.credentials

def get_user_id_by_api_key(api_key):
    """
    Get the user associated with the provided API key.
    
    Args:
        api_key (str): The API key to look up
        
    Returns:
        dict: User data if found, None otherwise
    """
    api_key_doc = config.db.collection('api_keys').document(api_key).get()
    api_key_data = api_key_doc.to_dict()
    user_id = api_key_data.get('userid')
    return user_id

def add_usage_to_user(user_id, token_count):
    """
    Add usage to the user's document in Firestore.
    
    Args:
        user_id (str): The ID of the user to add usage to
        token_count (int): The number of tokens to add
    """
    user_doc = config.db.collection('users').document(user_id).get()
    user_data = user_doc.to_dict()
    user_data['usage']['total_tokens'] += token_count
    user_data['usage']['total_messages'] += 1
    user_data['usage']['last_updated'] = datetime.now()
    config.db.collection('users').document(user_id).update(user_data)

def get_user_usage(user_id):
    """
    Get the usage of the user's document in Firestore.
    """
    user_doc = config.db.collection('users').document(user_id).get()
    user_data = user_doc.to_dict()
    return user_data.get('usage', 0)

def get_model_and_provider(model_id: str, models_dict):
    """Get model info and provider for a given model ID."""
    model_info = models_dict.get(model_id)
    if not model_info:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model_id}"
        )
    
    provider = config.PROVIDERS.get(model_info.provider)
    if not provider:
        raise HTTPException(
            status_code=500,
            detail=f"Provider not configured: {model_info.provider}"
        )
    
    return model_info.name, provider
