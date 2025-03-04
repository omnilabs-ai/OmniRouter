import asyncio
import uuid
import time
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator, Tuple, Type
from datetime import datetime

from serverRouter.omniAgents.agentRegistry.interfaces import AgentRunner, Agent
from serverRouter.omniAgents.agentRegistry.datamodels import AgentRunRequest, AgentRunResponse, AgentStep
from serverRouter.core.datamodels import ChatCompletionRequest, ChatMessage
from serverRouter.core.exceptions import ProviderError
from serverRouter.core.models import CHAT_MODELS
from serverRouter.core.config import PROVIDERS

logger = logging.getLogger(__name__)

class OmniAgentRunner(AgentRunner):
    """Implementation of AgentRunner for OmniLabs"""
    
    def __init__(self):
        """Initialize the OmniAgentRunner"""
        self.agents: Dict[str, Type[Agent]] = {}  # Registry of agent classes
        self.agent_instances: Dict[str, Agent] = {}  # Cached agent instances
        self.runs: Dict[str, Dict[str, Any]] = {}  # Run history
        self.active_runs: Dict[str, asyncio.Task] = {}  # Currently running agents
    
    def register_agent_class(self, agent_id: str, agent_class: Type[Agent]):
        """
        Register an agent class
        
        Args:
            agent_id: ID of the agent
            agent_class: Agent class to register
        """
        self.agents[agent_id] = agent_class
        logger.info(f"Registered agent class: {agent_id}")
    
    async def _get_agent_instance(self, agent_id: str) -> Agent:
        """
        Get or create an agent instance
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Agent instance
            
        Raises:
            ProviderError: If the agent is not registered
        """
        if agent_id not in self.agents:
            raise ProviderError(f"Agent not registered: {agent_id}")
        
        if agent_id not in self.agent_instances:
            self.agent_instances[agent_id] = self.agents[agent_id]()
            
        return self.agent_instances[agent_id]
    
    async def create_run(self, request: AgentRunRequest) -> str:
        """
        Create a new agent run
        
        Args:
            request: The run request
            
        Returns:
            ID of the created run
            
        Raises:
            ProviderError: If the agent is not registered or the run fails to start
        """
        # Generate a unique ID for this run
        run_id = str(uuid.uuid4())
        
        # Get the agent instance
        agent = await self._get_agent_instance(request.agent_id)
        
        # Validate inputs
        if not await agent.validate_inputs(request.inputs):
            raise ProviderError(f"Invalid inputs for agent: {request.agent_id}")
        
        # Create run record
        self.runs[run_id] = {
            "run_id": run_id,
            "agent_id": request.agent_id,
            "request": request.dict(),
            "status": "pending",
            "created_at": datetime.utcnow().isoformat(),
            "steps": [],
            "output": None,
            "error": None,
            "elapsed_time": 0
        }
        
        # Start the run in a background task if streaming, otherwise run synchronously
        if request.stream:
            self.active_runs[run_id] = asyncio.create_task(self._run_agent_background(run_id, agent, request))
        else:
            try:
                start_time = time.time()
                response = await agent.run(request)
                elapsed_time = time.time() - start_time
                
                self.runs[run_id].update({
                    "status": "completed" if response.status != "failed" else "failed",
                    "steps": [step.dict() for step in (response.steps or [])],
                    "output": response.output,
                    "error": response.error,
                    "elapsed_time": elapsed_time,
                    "completed_at": datetime.utcnow().isoformat()
                })
            except Exception as e:
                self.runs[run_id].update({
                    "status": "failed",
                    "error": str(e),
                    "completed_at": datetime.utcnow().isoformat()
                })
                logger.exception(f"Error running agent {request.agent_id}")
                raise ProviderError(f"Error running agent: {str(e)}")
        
        return run_id
    
    async def _run_agent_background(self, run_id: str, agent: Agent, request: AgentRunRequest):
        """
        Run an agent in the background and update its status
        
        Args:
            run_id: ID of the run
            agent: Agent instance
            request: The run request
        """
        try:
            start_time = time.time()
            steps = []
            
            self.runs[run_id]["status"] = "running"
            self.runs[run_id]["started_at"] = datetime.utcnow().isoformat()
            
            # Run the agent and collect steps
            async for step in agent.run_stream(request):
                steps.append(step.dict())
                self.runs[run_id]["steps"] = steps
            
            # Get the final result from the last step
            if steps:
                output = steps[-1].get("observation")
            else:
                output = None
            
            elapsed_time = time.time() - start_time
            
            self.runs[run_id].update({
                "status": "completed",
                "output": output,
                "elapsed_time": elapsed_time,
                "completed_at": datetime.utcnow().isoformat()
            })
        except Exception as e:
            self.runs[run_id].update({
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.utcnow().isoformat()
            })
            logger.exception(f"Error running agent {request.agent_id} in background")
        finally:
            if run_id in self.active_runs:
                del self.active_runs[run_id]
    
    async def get_run(self, run_id: str) -> AgentRunResponse:
        """
        Get information about a run
        
        Args:
            run_id: ID of the run
            
        Returns:
            Information about the run
            
        Raises:
            ProviderError: If the run is not found
        """
        if run_id not in self.runs:
            raise ProviderError(f"Run not found: {run_id}")
        
        run_data = self.runs[run_id]
        
        # Convert to AgentRunResponse
        return AgentRunResponse(
            run_id=run_data["run_id"],
            agent_id=run_data["agent_id"],
            status=run_data["status"],
            steps=[AgentStep(**step) for step in run_data.get("steps", [])],
            output=run_data.get("output"),
            error=run_data.get("error"),
            usage=run_data.get("usage", {}),
            elapsed_time=run_data.get("elapsed_time", 0)
        )
    
    async def list_runs(self, agent_id: Optional[str] = None, 
                      status: Optional[str] = None, 
                      limit: int = 100) -> List[Dict[str, Any]]:
        """
        List runs matching the given criteria
        
        Args:
            agent_id: Filter by agent ID
            status: Filter by status
            limit: Maximum number of runs to return
            
        Returns:
            List of matching runs
        """
        # Filter runs by criteria
        filtered_runs = []
        for run in self.runs.values():
            if agent_id and run["agent_id"] != agent_id:
                continue
            if status and run["status"] != status:
                continue
            filtered_runs.append(run)
        
        # Sort by creation time (newest first) and limit
        filtered_runs.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        return filtered_runs[:limit]
    
    async def cancel_run(self, run_id: str) -> bool:
        """
        Cancel a running agent
        
        Args:
            run_id: ID of the run to cancel
            
        Returns:
            True if the run was cancelled, False otherwise
        """
        if run_id not in self.runs:
            return False
        
        if run_id in self.active_runs:
            # Cancel the task
            self.active_runs[run_id].cancel()
            del self.active_runs[run_id]
            
            # Update the run status
            self.runs[run_id].update({
                "status": "cancelled",
                "completed_at": datetime.utcnow().isoformat()
            })
            
            return True
        else:
            # Run is not active
            return False