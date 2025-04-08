from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time
from serverRouter.core.datamodels import ModelProvider
from serverRouter.providers.anthropic.provider import AnthropicProvider
from serverRouter.providers.openai.provider import OpenAIProvider
from serverRouter.providers.gemini.provider import GeminiProvider
from serverRouter.providers.deepseek.provider import DeepSeekProvider
from serverRouter.providers.together.provider import TogetherAIProvider
from serverRouter.providers.stablediffusion.provider import StableDiffusionProvider
from serverRouter.providers.xai.provider import XAIProvider
from serverRouter.providers.deepai.provider import DeepAIProvider
from serverRouter.routes import model_routes, completion_routes, smart_routes, function_routes
from serverRouter.core.config import PROVIDERS
from serverRouter.core.exceptions import ProviderError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("serverRouter")

# Create FastAPI app
app = FastAPI(
    title="Omni Router API",
    description="A universal API router for AI models across multiple providers",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Function to initialize providers
def initialize_providers():
    global PROVIDERS
    try:
        PROVIDERS.update({
            ModelProvider.OPENAI: OpenAIProvider(),
            ModelProvider.ANTHROPIC: AnthropicProvider(),
            ModelProvider.GEMINI: GeminiProvider(),
            # ModelProvider.DEEPSEEK: DeepSeekProvider(), # NOT Fast Enough, using together instead
            ModelProvider.TOGETHER: TogetherAIProvider(),
            ModelProvider.STABLEDIFFUSION: StableDiffusionProvider(),
            ModelProvider.XAI: XAIProvider(),
            ModelProvider.DEEPAI: DeepAIProvider()
        })
    except Exception as e:
        raise

# # Initialize providers during startup
initialize_providers()
# # Include routers from separate files
app.include_router(model_routes.router)
app.include_router(completion_routes.router)
app.include_router(smart_routes.router)
app.include_router(function_routes.router)

# Add middleware for request timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
    except ProviderError as e:
        logger.error(f"Provider error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"An unexpected error occurred: {str(e)}"},
        )

@app.get("/")
async def root():
    return {
        "message": "Welcome to Omni Router API",
        "docs": "/docs",
    }

# uvicorn serverRouter.router:app --reload


