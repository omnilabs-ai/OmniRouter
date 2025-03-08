from fastapi import HTTPException

class ProviderNotFoundException(HTTPException):
    """Raised when a provider is not found"""
    def __init__(self, provider_name: str):
        super().__init__(
            status_code=400,
            detail=f"Unknown provider: {provider_name}"
        )

class ProviderError(HTTPException):
    """Raised when a provider encounters an error"""
    def __init__(self, message: str):
        super().__init__(
            status_code=500,
            detail=message
        )