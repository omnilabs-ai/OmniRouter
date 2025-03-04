from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, AsyncGenerator
from serverRouter.core.datamodels import ChatMessage
from serverRouter.omniAgents.agentRegistry.datamodels import AgentRunRequest, AgentRunResponse, AgentStep

class Agent(ABC):
    """Abstract base class for all agents"""
    
    @abstractmethod
    async def run(self, request: AgentRunRequest) -> AgentRunResponse:
        """
        Run the agent with the given request
        
        Args:
            request: The request containing inputs and configuration
            
        Returns:
            The agent's final response
        """
        pass
    
    @abstractmethod
    async def run_stream(self, request: AgentRunRequest) -> AsyncGenerator[AgentStep, None]:
        """
        Run the agent and stream each step
        
        Args:
            request: The request containing inputs and configuration
            
        Yields:
            Each step taken by the agent
        """
        pass
    
    @abstractmethod
    async def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """
        Validate the inputs for this agent
        
        Args:
            inputs: The inputs to validate
            
        Returns:
            True if inputs are valid, False otherwise
        """
        pass

class AgentRegistry(ABC):
    """Abstract base class for agent registries"""
    
    @abstractmethod
    async def register_agent(self, agent_info: Dict[str, Any]) -> str:
        """
        Register a new agent in the registry
        
        Args:
            agent_info: Information about the agent
            
        Returns:
            The ID of the registered agent
        """
        pass
    
    @abstractmethod
    async def get_agent(self, agent_id: str) -> Dict[str, Any]:
        """
        Get information about an agent
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Information about the agent
        """
        pass
    
    @abstractmethod
    async def list_agents(self, 
                         capabilities: Optional[List[str]] = None, 
                         tags: Optional[List[str]] = None, 
                         status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List agents matching the given criteria
        
        Args:
            capabilities: Filter by capabilities
            tags: Filter by tags
            status: Filter by status
            
        Returns:
            List of matching agents
        """
        pass
    
    @abstractmethod
    async def update_agent(self, agent_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an agent's information
        
        Args:
            agent_id: ID of the agent to update
            updates: Fields to update
            
        Returns:
            Updated agent information
        """
        pass
    
    @abstractmethod
    async def delete_agent(self, agent_id: str) -> bool:
        """
        Delete an agent from the registry
        
        Args:
            agent_id: ID of the agent to delete
            
        Returns:
            True if the agent was deleted, False otherwise
        """
        pass

class AgentRunner(ABC):
    """Abstract base class for agent runners"""
    
    @abstractmethod
    async def create_run(self, request: AgentRunRequest) -> str:
        """
        Create a new agent run
        
        Args:
            request: The run request
            
        Returns:
            ID of the created run
        """
        pass
    
    @abstractmethod
    async def get_run(self, run_id: str) -> AgentRunResponse:
        """
        Get information about a run
        
        Args:
            run_id: ID of the run
            
        Returns:
            Information about the run
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    async def cancel_run(self, run_id: str) -> bool:
        """
        Cancel a running agent
        
        Args:
            run_id: ID of the run to cancel
            
        Returns:
            True if the run was cancelled, False otherwise
        """
        pass