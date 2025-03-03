"""
Browser Use Agent adapter for OmniAgents platform.

This adapter integrates the browser-use agent into the OmniAgents platform, 
allowing developers to perform browser automation tasks through the unified API.
"""

import os
import json
import asyncio
import tempfile
from typing import Dict, Any, List, Optional, AsyncGenerator, Union
import logging

from pydantic import BaseModel, Field
from serverRouter.omniAgents.agentRegistry.interfaces import Agent
from serverRouter.omniAgents.agentRegistry.datamodels import AgentRunRequest, AgentRunResponse, AgentStep, AgentCapability
from serverRouter.core.exceptions import ProviderError

# Setup logging
logger = logging.getLogger(__name__)

class BrowserUseAgentAdapter(Agent):
    """
    Adapter for the browser-use agent for web automation tasks.
    
    This adapter wraps the browser-use agent's functionality to make it compatible
    with the OmniAgents platform's unified API.
    """
    
    def __init__(self):
        """Initialize the Browser Use agent adapter"""
        self.name = "BrowserUseAgent"
        self.description = "Agent for browser automation and web interaction tasks"
        self.default_model = "gpt-4o"  # Default to GPT-4o for best performance
        self.capabilities = [
            AgentCapability.BROWSER_AUTOMATION,
            AgentCapability.WEB_SEARCH
        ]
        self.tags = ["browsing", "automation", "web", "scraping"]
        
        # Verify dependencies are installed
        try:
            import browser_use
            from langchain_openai import ChatOpenAI
            from langchain_anthropic import ChatAnthropic
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:
            logger.error(f"Failed to import browser-use dependencies: {e}")
            raise ProviderError(f"browser-use dependencies not installed. Please install with: pip install browser-use")
    
    async def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """
        Validate the inputs for the browser-use agent
        
        Args:
            inputs: The inputs to validate
            
        Returns:
            True if inputs are valid, False otherwise
        """
        # Check for required prompt/task
        if "prompt" not in inputs and "task" not in inputs:
            return False
            
        return True
    
    def _map_model_to_llm(self, model_name: str, api_keys: Dict[str, str] = None) -> Any:
        """
        Map our platform's model names to LangChain LLM instances
        
        Args:
            model_name: Name of the model to use
            api_keys: Optional dictionary of API keys
            
        Returns:
            LangChain LLM instance
        """
        # Import here to avoid dependency issues
        from langchain_openai import ChatOpenAI
        from langchain_anthropic import ChatAnthropic
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        # Set API keys from input or environment
        api_keys = api_keys or {}
        for key, value in api_keys.items():
            if key and value:
                os.environ[key] = value
        
        # Map model names to LangChain models
        if model_name.startswith("gpt-"):
            return ChatOpenAI(model=model_name, temperature=0.0)
        elif model_name.startswith("claude-"):
            return ChatAnthropic(model=model_name, temperature=0.0)
        elif model_name.startswith("gemini-"):
            return ChatGoogleGenerativeAI(model=model_name, temperature=0.0)
        elif model_name.startswith("deepseek-"):
            # Special handling for DeepSeek
            if model_name == "deepseek-v3":
                return ChatOpenAI(
                    base_url='https://api.deepseek.com/v1', 
                    model='deepseek-chat', 
                    temperature=0.0
                )
            elif model_name == "deepseek-r1":
                return ChatOpenAI(
                    base_url='https://api.deepseek.com/v1', 
                    model='deepseek-reasoner', 
                    temperature=0.0
                )
        else:
            # Default to OpenAI for unknown models
            return ChatOpenAI(model="gpt-4o", temperature=0.0)
    
    async def _initialize_browser_use_agent(self, request: AgentRunRequest) -> Any:
        """
        Initialize the browser-use agent with the given configuration
        
        Args:
            request: The request containing configuration
            
        Returns:
            Initialized browser-use Agent instance
        """
        from browser_use import Agent as BrowserAgent
        from browser_use import Browser, BrowserConfig
        
        # Extract configuration from request
        config = request.inputs.get("config", {})
        
        # Extract task from prompt or task field
        task = request.inputs.get("task", request.inputs.get("prompt", ""))
        
        # Get model to use (from request or default)
        model_name = request.model or self.default_model
        
        # Initialize the LLM
        llm = self._map_model_to_llm(
            model_name, 
            api_keys=request.inputs.get("api_keys", {})
        )
        
        # Initialize any browser configuration if provided
        browser_config = None
        if "browser_config" in config:
            browser_config = BrowserConfig(**config["browser_config"])
            
        # Initialize browser if needed
        browser = None
        if browser_config:
            browser = Browser(config=browser_config)
            
        # Initialize agent with configuration
        # Create a dictionary of parameters to handle version differences
        agent_params = {
            "task": task,
            "llm": llm,
            "browser": browser,
            "use_vision": config.get("use_vision", True),
            "save_conversation_path": config.get("save_conversation_path", None),
            "sensitive_data": config.get("sensitive_data", None),
            "initial_actions": config.get("initial_actions", None)
        }
        
        # Check if other parameters are supported in this version
        system_message = config.get("extend_system_message", None)
        if system_message:
            # Try different parameter names that might be used in different versions
            try:
                # For newer versions
                agent_params["extend_system_message"] = system_message
            except TypeError:
                try:
                    # For newer versions with different param name
                    agent_params["system_message"] = system_message
                except:
                    # Log that we couldn't use the system message
                    logger.warning("System message parameter not supported in this browser-use version")
                    
        # Initialize the agent with filtered parameters
        try:
            browser_use_agent = BrowserAgent(**agent_params)
        except TypeError as e:
            # If we still get errors, try with minimal parameters
            logger.warning(f"Error with parameters: {e}. Trying with minimal parameters.")
            browser_use_agent = BrowserAgent(
                task=task,
                llm=llm,
                browser=browser,
                use_vision=config.get("use_vision", True)
            )
        
        return browser_use_agent
    
    def _convert_agent_history_to_steps(self, history) -> List[AgentStep]:
        """
        Convert browser-use agent history to our platform's AgentStep format
        
        Args:
            history: browser-use agent history
            
        Returns:
            List of AgentStep objects
        """
        steps = []
        
        # Extract relevant fields from agent history
        action_names = history.action_names()
        action_results = history.action_results()
        model_actions = history.model_actions()
        model_thoughts = history.model_thoughts()
        
        # Create steps from history
        for i, (action_name, action_result) in enumerate(zip(action_names, action_results)):
            # Get action input if available
            action_input = {}
            if i < len(model_actions):
                action_input = model_actions[i].get("parameters", {})
                
            # Get thought if available - ensure it's a string
            thought = ""
            if i < len(model_thoughts):
                # Convert thought to string if it's not already
                if hasattr(model_thoughts[i], '__str__'):
                    thought = str(model_thoughts[i])
                elif isinstance(model_thoughts[i], dict) and 'content' in model_thoughts[i]:
                    thought = model_thoughts[i]['content']
                elif isinstance(model_thoughts[i], str):
                    thought = model_thoughts[i]
                
            # Convert action result to string if it's not already
            observation = ""
            if action_result is not None:
                if isinstance(action_result, str):
                    observation = action_result
                else:
                    observation = str(action_result)
                
            # Create the step
            step = AgentStep(
                step_id=i,
                thought=thought,
                action=action_name,
                action_input=action_input,
                observation=observation
            )
            steps.append(step)
            
        return steps
    
    async def run(self, request: AgentRunRequest) -> AgentRunResponse:
        """
        Run the browser-use agent with the given request
        
        Args:
            request: The request containing inputs and configuration
            
        Returns:
            The agent's final response
        """
        try:
            # Initialize browser-use agent
            browser_use_agent = await self._initialize_browser_use_agent(request)
            
            # Track start time
            import time
            start_time = time.time()
            
            # Run the agent
            history = await browser_use_agent.run()
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            
            # Convert history to steps
            steps = self._convert_agent_history_to_steps(history)
            
            # Extract final result
            final_result = history.final_result()
            
            # Close browser if it was created by the agent
            if hasattr(browser_use_agent, "browser") and browser_use_agent.browser:
                await browser_use_agent.browser.close()
            
            # Return the response
            return AgentRunResponse(
                run_id=request.run_id if hasattr(request, "run_id") else "",
                agent_id="BrowserUseAgent",
                status="completed",
                steps=steps,
                output=final_result,
                error=None,
                usage={},  # Will be filled in by runner
                elapsed_time=elapsed_time
            )
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"Error running browser-use agent: {error_details}")
            
            # Return error response
            return AgentRunResponse(
                run_id=request.run_id if hasattr(request, "run_id") else "",
                agent_id="BrowserUseAgent",
                status="failed",
                steps=[],
                output=None,
                error=f"Error running browser-use agent: {str(e)}",
                usage={},
                elapsed_time=0.0
            )
    
    async def run_stream(self, request: AgentRunRequest) -> AsyncGenerator[AgentStep, None]:
        """
        Run the browser-use agent and stream each step
        
        Args:
            request: The request containing inputs and configuration
            
        Yields:
            Each step taken by the agent
        """
        browser_use_agent = None
        
        try:
            # Initialize browser-use agent
            browser_use_agent = await self._initialize_browser_use_agent(request)
            
            # Run the agent with streaming (we'll use a local queue to simulate streaming)
            queue = asyncio.Queue()
            
            # Create a task to run the agent and populate the queue
            async def run_agent():
                try:
                    # Run the agent
                    history = await browser_use_agent.run()
                    
                    # Convert history to steps
                    steps = self._convert_agent_history_to_steps(history)
                    
                    # Put steps in the queue
                    for step in steps:
                        await queue.put(step)
                    
                    # Signal completion    
                    await queue.put(None)
                except Exception as e:
                    # Handle errors
                    error_step = AgentStep(
                        step_id=0,
                        thought="Error occurred",
                        action="error",
                        action_input={},
                        observation=f"Error: {str(e)}"
                    )
                    await queue.put(error_step)
                    await queue.put(None)
            
            # Start the agent task
            agent_task = asyncio.create_task(run_agent())
            
            # Yield steps from the queue as they become available
            while True:
                step = await queue.get()
                if step is None:  # End of stream
                    break
                yield step
                
            # Wait for the agent task to complete
            await agent_task
        
        except Exception as e:
            # Handle errors
            error_step = AgentStep(
                step_id=0,
                thought="Error in run_stream",
                action="error",
                action_input={},
                observation=f"Error: {str(e)}"
            )
            yield error_step
        
        finally:
            # Close browser if it was created
            if browser_use_agent and hasattr(browser_use_agent, "browser") and browser_use_agent.browser:
                await browser_use_agent.browser.close()