from typing import List, Optional, Dict, Literal, Union, Any, Set
from pydantic import BaseModel, Field, HttpUrl
from enum import Enum
from datetime import datetime
from serverRouter.core.datamodels import ModelProvider, BenchmarkScores

class AgentCapability(str, Enum):
    """Capabilities that agents can have"""
    WEB_SEARCH = "web_search"
    CODE_EXECUTION = "code_execution"
    DATABASE_ACCESS = "database_access"
    FILE_MANAGEMENT = "file_management"
    EMAIL_COMMUNICATION = "email_communication"
    BROWSER_AUTOMATION = "browser_automation"
    API_CALLING = "api_calling"
    PLANNING = "planning"
    MEMORY = "memory"
    MULTI_MODAL = "multi_modal"
    
class AgentBenchmarkScores(BenchmarkScores):
    """Extended benchmark scores specific to agents"""
    AGENT_BENCH: Optional[float] = Field(None, ge=0.0, le=1.0)
    TOOL_USE: Optional[float] = Field(None, ge=0.0, le=1.0)
    REASONING: Optional[float] = Field(None, ge=0.0, le=1.0)
    PLANNING: Optional[float] = Field(None, ge=0.0, le=1.0)
    RETRIEVAL: Optional[float] = Field(None, ge=0.0, le=1.0)
    MULTI_STEP: Optional[float] = Field(None, ge=0.0, le=1.0)

class AgentStatus(str, Enum):
    """Current status of an agent"""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"
    BETA = "beta"

class AgentSource(BaseModel):
    """Source information for an agent"""
    repository_url: HttpUrl = Field(..., description="GitHub repository URL")
    license: str = Field(..., description="License type (e.g., MIT, Apache-2.0)")
    stars: Optional[int] = Field(None, description="Number of GitHub stars")
    last_updated: Optional[datetime] = Field(None, description="Last update timestamp")
    maintainers: Optional[List[str]] = Field(None, description="List of maintainer names or handles")

class AgentTool(BaseModel):
    """A tool that an agent can use"""
    name: str = Field(..., description="Name of the tool")
    description: str = Field(..., description="Description of what the tool does")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Parameters for the tool")
    required_permissions: Optional[List[str]] = Field(None, description="Required permissions to use this tool")

class AgentInfo(BaseModel):
    """Information about an agent"""
    id: str = Field(..., description="Unique identifier for the agent")
    name: str = Field(..., description="Display name of the agent")
    description: str = Field(..., description="Description of what the agent does")
    version: str = Field(..., description="Version of the agent")
    capabilities: List[AgentCapability] = Field(..., description="List of agent capabilities")
    compatible_models: List[str] = Field(..., description="List of compatible models")
    tools: Optional[List[AgentTool]] = Field(None, description="Tools that the agent can use")
    source: AgentSource = Field(..., description="Source information")
    benchmarks: Optional[AgentBenchmarkScores] = Field(None, description="Agent benchmark scores")
    status: AgentStatus = Field(default=AgentStatus.EXPERIMENTAL, description="Current status of the agent")
    documentation_url: Optional[HttpUrl] = Field(None, description="URL to the agent's documentation")
    example_prompts: Optional[List[str]] = Field(None, description="Example prompts to use with this agent")
    tags: Optional[List[str]] = Field(None, description="Tags for categorizing the agent")
    
class AgentRunRequest(BaseModel):
    """Request to run an agent"""
    agent_id: str = Field(..., description="ID of the agent to run")
    inputs: Dict[str, Any] = Field(..., description="Input parameters for the agent")
    model: Optional[str] = Field(None, description="Model to use for the agent (if not using default)")
    max_steps: Optional[int] = Field(default=10, description="Maximum number of steps the agent can take")
    timeout_seconds: Optional[int] = Field(default=60, description="Maximum time in seconds the agent can run")
    stream: bool = Field(default=False, description="Whether to stream the agent's steps and outputs")
    
class AgentStep(BaseModel):
    """A single step in an agent's execution"""
    step_id: int = Field(..., description="ID of this step")
    thought: Optional[str] = Field(None, description="Agent's reasoning for this step")
    action: str = Field(..., description="Action that the agent took")
    action_input: Optional[Dict[str, Any]] = Field(None, description="Input to the action")
    observation: Optional[Any] = Field(None, description="Result of the action")
    
class AgentRunResponse(BaseModel):
    """Response from running an agent"""
    run_id: str = Field(..., description="Unique ID for this run")
    agent_id: str = Field(..., description="ID of the agent that was run")
    status: str = Field(..., description="Status of the run (completed, failed, etc.)")
    steps: Optional[List[AgentStep]] = Field(None, description="Steps taken by the agent")
    output: Any = Field(..., description="Final output from the agent")
    error: Optional[str] = Field(None, description="Error message if the run failed")
    usage: Dict[str, Any] = Field(default_factory=dict, description="Usage statistics")
    elapsed_time: float = Field(..., description="Time taken to complete the run in seconds")