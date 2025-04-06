from fastapi import APIRouter, Depends, HTTPException
from serverRouter.routes.utils import verify_api_key, get_user_id_by_api_key, add_usage_to_user
from serverRouter.core.function_registry import function_registry, ProviderType
from serverRouter.core.datamodels import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    FunctionCall,
    FunctionExecutionResult
)
from typing import List, Dict, Any, Optional
import json

router = APIRouter(prefix="/v1/functions", tags=["functions"])

@router.get("/")
async def list_functions(
    api_key: str = Depends(verify_api_key)
) -> List[Dict[str, Any]]:
    """
    List all registered functions
    """
    functions = function_registry.get_all_functions()
    
    # Convert to serializable format
    function_list = []
    for func in functions:
        function_list.append({
            "name": func.name,
            "description": func.description,
            "parameters": {
                param_name: {
                    "type": param.type,
                    "description": param.description,
                    "required": param.required
                }
                for param_name, param in func.parameters.items()
            },
            "auto_execute": func.auto_execute
        })
    
    return function_list

@router.get("/provider/{provider}")
async def get_provider_functions(
    provider: str,
    functions: Optional[List[str]] = None,
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Get functions in provider-specific format
    """
    try:
        provider_type = ProviderType(provider.lower())
        provider_functions = function_registry.get_for_provider(provider_type, functions)
        return {"format": provider, "functions": provider_functions}
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{function_name}/execute")
async def execute_function(
    function_name: str,
    arguments: Dict[str, Any],
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Execute a registered function
    """
    # Check if function exists
    if not function_registry.exists(function_name):
        raise HTTPException(status_code=404, detail=f"Function {function_name} not found")
    
    # Execute the function
    result = function_registry.execute_function(function_name, arguments)
    
    # Return the result
    return {
        "function_name": result.function_name,
        "success": result.success,
        "result": result.result if result.success else None,
        "error": result.error if not result.success else None,
        "execution_time": result.execution_time
    }

@router.get("/history")
async def get_execution_history(
    api_key: str = Depends(verify_api_key)
) -> List[Dict[str, Any]]:
    """
    Get function execution history
    """
    history = function_registry.get_execution_history()
    
    # Convert to serializable format
    history_list = []
    for item in history:
        history_list.append({
            "function_name": item.function_name,
            "arguments": item.arguments,
            "success": item.success,
            "execution_time": item.execution_time,
            "error": item.error if not item.success else None
        })
    
    return history_list 