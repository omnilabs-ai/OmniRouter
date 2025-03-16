from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from serverRouter.core.datamodels import ModelProvider
from serverRouter.providers.anthropic.provider import AnthropicProvider
from serverRouter.providers.openai.provider import OpenAIProvider
from serverRouter.providers.gemini.provider import GeminiProvider
from serverRouter.providers.deepseek.provider import DeepSeekProvider
from serverRouter.providers.together.provider import TogetherAIProvider
from serverRouter.providers.stablediffusion.provider import StableDiffusionProvider
from serverRouter.routes import model_routes, completion_routes, smart_routes
from serverRouter.core.config import PROVIDERS


app = FastAPI(title="OmniLLM", description="One Key, One API, Hundreds of Models")

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
            ModelProvider.DEEPSEEK: DeepSeekProvider(),
            ModelProvider.TOGETHER: TogetherAIProvider(),
            ModelProvider.STABLEDIFFUSION: StableDiffusionProvider()
        })
    except Exception as e:
        raise

# # Initialize providers during startup
initialize_providers()
# # Include routers from separate files
app.include_router(model_routes.router)
app.include_router(completion_routes.router)
app.include_router(smart_routes.router)


@app.get("/")
async def root():
    return {"message": "Welcome to OmniLLM!"}


# uvicorn serverRouter.router:app --reload


