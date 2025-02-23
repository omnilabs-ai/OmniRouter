from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from serverRouter.core.datamodels import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelProvider,
    ImageGenerationRequest,
    ImageGenerationResponse,
    SmartRouterRequest
)
from serverRouter.providers.anthropic.provider import AnthropicProvider
from serverRouter.providers.openai.provider import OpenAIProvider
from serverRouter.providers.gemini.provider import GeminiProvider  # Make sure the path is correct
from serverRouter.providers.deepseek.provider import DeepSeekProvider
from serverRouter.core.models import (
    MODELS,
    CHAT_MODELS,
    IMAGE_MODELS
)
from serverRouter.smartRouter.SmartRouter import SmartRouter

app = FastAPI(title="OmniLLM", description="One Key, One API, Hundreds of Models")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

security = HTTPBearer()

# Provider instances cache
PROVIDERS = {}

# Function to initialize providers (allows for error handling)
def initialize_providers():
    global PROVIDERS
    try:
        PROVIDERS = {
            ModelProvider.OPENAI: OpenAIProvider(),
            ModelProvider.ANTHROPIC: AnthropicProvider(),
            ModelProvider.GEMINI: GeminiProvider(), # modified to be variable
            ModelProvider.DEEPSEEK: DeepSeekProvider()
        }
    except Exception as e:
        raise  # Re-raise to prevent the server from starting

# Initialize providers during startup
try:
    initialize_providers()
except Exception:
    # Handle the error appropriately (e.g., log, exit)
    import sys
    sys.exit(1)  # Exit if provider initialization fails

# Initialize SmartRouter
smart_router = SmartRouter()

VALID_API_KEYS = {
    "test-sk1o83e",
}

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    if credentials.credentials not in VALID_API_KEYS:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid API key"
        )
    return credentials.credentials

@app.get("/v1/models")
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

@app.get("/")
async def root():
    return {"message": "Welcome to OmniLLM!"}

@app.get("/v1/models/chat")
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

@app.get("/v1/models/image")
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

@app.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    api_key: str = Depends(verify_api_key)
) -> ChatCompletionResponse:
    """
    Create a chat completion using the specified model.
    """
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

@app.post("/v1/images/generate")
async def create_image(
    request: ImageGenerationRequest,
    api_key: str = Depends(verify_api_key)
) -> ImageGenerationResponse:
    """
    Generate images using the specified model.
    """
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

@app.post("/v1/router/select-model")
async def select_model(
    request: SmartRouterRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Get model recommendations based on the query and preferences.
    """
    try:
        result = smart_router.get_top_user_models(
            query=request.messages[-1].content,  # last message is the query
            k=request.k,  # defaults to 5 if not provided
            model_names=request.model_names,  # defaults to None if not provided
            rel_cost=request.rel_cost,  # defaults to 0.5 if not provided
            rel_latency=request.rel_latency,  # defaults to 0.0 if not provided
            rel_accuracy=request.rel_accuracy  # defaults to 0.5 if not provided
        )
        # return result

        response = await create_chat_completion(
            request=ChatCompletionRequest(
                model=result["model"],
                messages=request.messages,
            )
        )

        return response
    

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))