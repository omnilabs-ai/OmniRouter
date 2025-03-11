from fastapi import APIRouter, HTTPException, Depends
from serverRouter.routes.utils import verify_api_key
from serverRouter.core.datamodels import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    SmartRouterRequest
)

from serverRouter.routes.completion_routes import create_chat_completion, create_chat_completion_stream
from serverRouter.smartRouter.SmartRouter import SmartRouter

router = APIRouter(prefix="/v1", tags=["smart"])

smart_router = SmartRouter()


@router.post("/router/smart_select")
async def smart_select(
    request: SmartRouterRequest,
    api_key: str = Depends(verify_api_key)
) -> ChatCompletionResponse:
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
    

@router.post("/router/smart_select/stream")
async def smart_select_stream(
    request: SmartRouterRequest,
    api_key: str = Depends(verify_api_key)
) -> ChatCompletionResponse:
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

        response = await create_chat_completion_stream(
            request=ChatCompletionRequest(
                model=result["model"],
                messages=request.messages,
            )
        )

        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

