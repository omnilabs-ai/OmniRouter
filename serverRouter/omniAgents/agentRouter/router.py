from fastapi import APIRouter, HTTPException, Security, Depends, BackgroundTasks
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sse_starlette.sse import EventSourceResponse
import json
import asyncio
from typing import Dict, Any, List, Optional

from serverRouter.core.config import VALID_API_KEYS
from serverRouter.omniAgents.agentRegistry.datamodels import (
    AgentInfo,
    AgentRunRequest,
    AgentRunResponse,
    AgentStep
)
from serverRouter.omniAgents.agentRunner.agent_runner import OmniAgentRunner

# Initialize the router
router = APIRouter(prefix="/v1/agents", tags=["agents"])

# Initialize security
security = HTTPBearer()

# Create a global agent runner
agent_runner = OmniAgentRunner()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """Verify the API key"""
    if credentials.credentials not in VALID_API_KEYS:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid API key"
        )
    return credentials.credentials

@router.get("/")
async def list_agents(
    capabilities: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    status: Optional[str] = None,
    api_key: str = Depends(verify_api_key)
):
    """
    List available agents
    
    Args:
        capabilities: Filter by capabilities
        tags: Filter by tags
        status: Filter by status
        api_key: API key for authentication
        
    Returns:
        List of agents matching the criteria
    """
    # This is a simplified implementation - in practice, you'd use a database
    # or registry to store and query agent information
    
    # Get all registered agent classes
    agents = []
    for agent_id, agent_class in agent_runner.agents.items():
        # Create an instance to get its properties
        agent_instance = agent_class()
        
        # Create a simple agent info object
        agent_info = {
            "id": agent_id,
            "name": getattr(agent_instance, "name", agent_id),
            "description": getattr(agent_instance, "description", ""),
            "capabilities": getattr(agent_instance, "capabilities", []),
            "status": "active"
        }
        
        # Apply filters
        if capabilities and not any(c in agent_info["capabilities"] for c in capabilities):
            continue
        if tags and not any(t in getattr(agent_instance, "tags", []) for t in tags):
            continue
        if status and agent_info["status"] != status:
            continue
        
        agents.append(agent_info)
    
    return {"agents": agents}

@router.get("/{agent_id}")
async def get_agent(
    agent_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Get information about an agent
    
    Args:
        agent_id: ID of the agent
        api_key: API key for authentication
        
    Returns:
        Information about the agent
    """
    # Check if the agent exists
    if agent_id not in agent_runner.agents:
        raise HTTPException(
            status_code=404,
            detail=f"Agent not found: {agent_id}"
        )
    
    # Create an instance to get its properties
    agent_class = agent_runner.agents[agent_id]
    agent_instance = agent_class()
    
    # Create a simple agent info object
    agent_info = {
        "id": agent_id,
        "name": getattr(agent_instance, "name", agent_id),
        "description": getattr(agent_instance, "description", ""),
        "capabilities": getattr(agent_instance, "capabilities", []),
        "status": "active",
        "default_model": getattr(agent_instance, "default_model", None),
        "tools": getattr(agent_instance, "tools", {})
    }
    
    return agent_info

@router.post("/run")
async def run_agent(
    request: AgentRunRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """
    Run an agent
    
    Args:
        request: The run request
        background_tasks: FastAPI background tasks
        api_key: API key for authentication
        
    Returns:
        The run ID
    """
    try:
        # Create the run
        run_id = await agent_runner.create_run(request)
        
        # Return the run ID
        return {"run_id": run_id}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error running agent: {str(e)}"
        )

@router.get("/runs/{run_id}")
async def get_agent_run(
    run_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Get information about an agent run
    
    Args:
        run_id: ID of the run
        api_key: API key for authentication
        
    Returns:
        Information about the run
    """
    try:
        # Get the run
        run = await agent_runner.get_run(run_id)
        
        # Return the run
        return run
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Run not found: {str(e)}"
        )

@router.get("/runs/{run_id}/stream")
async def stream_agent_run(
    run_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Stream an agent run
    
    Args:
        run_id: ID of the run
        api_key: API key for authentication
        
    Returns:
        Server-sent events stream of run steps
    """
    async def event_generator():
        """Generate events for SSE"""
        # Check if the run exists
        if run_id not in agent_runner.runs:
            yield json.dumps({"error": f"Run not found: {run_id}"})
            return
        
        # Get the initial run state
        run = agent_runner.runs[run_id]
        last_step_idx = -1
        
        # Stream until the run is complete
        while run["status"] in ["pending", "running"]:
            # Get any new steps
            current_steps = run["steps"]
            for i in range(last_step_idx + 1, len(current_steps)):
                yield json.dumps({
                    "event": "step",
                    "data": current_steps[i]
                })
                last_step_idx = i
            
            # If the run is complete, send the final result
            if run["status"] not in ["pending", "running"]:
                yield json.dumps({
                    "event": "complete",
                    "data": {
                        "status": run["status"],
                        "output": run["output"],
                        "error": run["error"],
                        "elapsed_time": run["elapsed_time"]
                    }
                })
                break
            
            # Wait a bit before checking again
            await asyncio.sleep(0.1)
    
    return EventSourceResponse(event_generator())

@router.post("/runs/{run_id}/cancel")
async def cancel_agent_run(
    run_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Cancel an agent run
    
    Args:
        run_id: ID of the run to cancel
        api_key: API key for authentication
        
    Returns:
        Success message
    """
    try:
        # Cancel the run
        success = await agent_runner.cancel_run(run_id)
        
        if success:
            return {"message": f"Run {run_id} cancelled successfully"}
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to cancel run: {run_id} (may not be active)"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error cancelling run: {str(e)}"
        )

@router.get("/runs")
async def list_agent_runs(
    agent_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    api_key: str = Depends(verify_api_key)
):
    """
    List agent runs
    
    Args:
        agent_id: Filter by agent ID
        status: Filter by status
        limit: Maximum number of runs to return
        api_key: API key for authentication
        
    Returns:
        List of runs matching the criteria
    """
    try:
        # Get the runs
        runs = await agent_runner.list_runs(
            agent_id=agent_id,
            status=status,
            limit=limit
        )
        
        # Return the runs
        return {"runs": runs}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing runs: {str(e)}"
        )