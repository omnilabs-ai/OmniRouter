from fastapi import APIRouter, Depends
from serverRouter.routes.utils import verify_api_key
from serverRouter.core.datamodels import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    SmartRouterRequest
)

from serverRouter.routes.completion_routes import create_chat_completion, create_chat_completion_stream
from serverRouter.smartRouter.main import SmartRouter
from sse_starlette.sse import EventSourceResponse

router = APIRouter(prefix="/v1", tags=["smart"])

@router.post("/smartRouterStream")
async def smartRouterStream(request: SmartRouterRequest):
    return EventSourceResponse(SmartRouter(request.messages, request.max_latency, request.max_cost, request.model_list))

@router.post("/smartRouter")
async def smartRouter(request: SmartRouterRequest):
    response_generator = SmartRouter(request.messages, request.max_latency, request.max_cost, request.model_list)

    for event in response_generator:
        if event["event"] == "return":
            return event["data"]
    
    return {"error": "No valid response found"}