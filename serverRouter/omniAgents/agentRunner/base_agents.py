import asyncio
import json
import time
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator, Type, Tuple
from abc import ABC, abstractmethod

from serverRouter.omniAgents.agentRegistry.interfaces import Agent
from serverRouter.omniAgents.agentRegistry.datamodels import AgentRunRequest, AgentRunResponse, AgentStep
from serverRouter.core.datamodels import ChatCompletionRequest, ChatMessage
from serverRouter.core.exceptions import ProviderError
from serverRouter.core.models import CHAT_MODELS
from serverRouter.core.config import PROVIDERS

logger = logging.getLogger(__name__)

class BaseAgent(Agent):
    """Base class for all agents"""
    
    def __init__(self):
        """Initialize the base agent"""
        self.name = "BaseAgent"
        self.description = "Base agent class"
    
    async def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """
        Validate inputs - base implementation always returns True
        
        Args:
            inputs: The inputs to validate
            
        Returns:
            True if inputs are valid
        """
        return True
    
    async def _run_llm(self, model: str, messages: List[ChatMessage]) -> str:
        """
        Run an LLM with the given messages
        
        Args:
            model: The model to use
            messages: The messages to send
            
        Returns:
            The model's response
            
        Raises:
            ProviderError: If the model fails to generate a response
        """
        try:
            # Look up the model info
            model_info = CHAT_MODELS.get(model)
            if not model_info:
                raise ProviderError(f"Unknown model: {model}")
            
            # Get the provider
            provider = PROVIDERS.get(model_info.provider)
            if not provider:
                raise ProviderError(f"Provider not configured: {model_info.provider}")
            
            # Create the request
            request = ChatCompletionRequest(
                model=model_info.name,
                messages=messages,
                temperature=0.7,  # Default temperature
                max_tokens=None  # Use default max tokens
            )
            
            # Run the model
            response = await provider.chat_complete(request)
            
            return response.content
        except Exception as e:
            logger.exception(f"Error running LLM {model}")
            raise ProviderError(f"Error running LLM: {str(e)}")

class ReActAgent(BaseAgent):
    """Base class for ReAct (Reasoning and Acting) agents"""
    
    def __init__(self):
        """Initialize the ReAct agent"""
        super().__init__()
        self.name = "ReActAgent"
        self.description = "Agent using the ReAct (Reasoning and Acting) framework"
        self.tools = {}  # Tools available to this agent
        self.default_model = "gpt-4o-mini"  # Default model for the agent
        self.max_iterations = 10  # Default max iterations
    
    def register_tool(self, name: str, func: callable, description: str):
        """
        Register a tool with this agent
        
        Args:
            name: Name of the tool
            func: Function implementing the tool
            description: Description of what the tool does
        """
        self.tools[name] = {
            "func": func,
            "description": description
        }
    
    async def run(self, request: AgentRunRequest) -> AgentRunResponse:
        """
        Run the agent with the given request
        
        Args:
            request: The request containing inputs and configuration
            
        Returns:
            The agent's final response
        """
        # Start timing
        start_time = time.time()
        
        # Get model from request or use default
        model = request.model or self.default_model
        
        # Get max steps from request or use default
        max_steps = request.max_steps or self.max_iterations
        
        # Get input prompt
        prompt = request.inputs.get("prompt", "")
        if not prompt:
            raise ProviderError("No prompt provided in inputs")
        
        # Initial context for the agent
        context = self._build_agent_context(request.inputs)
        
        # Track steps
        steps = []
        
        try:
            # Main agent loop
            for i in range(max_steps):
                # Generate a step
                thought, action, action_input = await self._generate_step(model, context, prompt, steps)
                
                # Execute the action
                if action in self.tools:
                    try:
                        tool_func = self.tools[action]["func"]
                        observation = await tool_func(action_input)
                    except Exception as e:
                        observation = f"Error executing {action}: {str(e)}"
                else:
                    observation = f"Unknown action: {action}"
                
                # Create a step
                step = AgentStep(
                    step_id=i,
                    thought=thought,
                    action=action,
                    action_input=action_input,
                    observation=observation
                )
                steps.append(step)
                
                # Update context with the step
                context += f"\nStep {i+1}:\nThought: {thought}\nAction: {action}\nAction Input: {json.dumps(action_input)}\nObservation: {observation}"
                
                # Check if we're done
                if action == "finish":
                    break
            
            # Extract final answer from the last step
            final_output = steps[-1].observation if steps else None
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            
            # Create response
            return AgentRunResponse(
                run_id="",  # Will be filled in by the runner
                agent_id=request.agent_id,
                status="completed",
                steps=steps,
                output=final_output,
                error=None,
                usage={},  # Will be filled in by the runner
                elapsed_time=elapsed_time
            )
        except Exception as e:
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            
            # Create response with error
            return AgentRunResponse(
                run_id="",  # Will be filled in by the runner
                agent_id=request.agent_id,
                status="failed",
                steps=steps,
                output=None,
                error=str(e),
                usage={},  # Will be filled in by the runner
                elapsed_time=elapsed_time
            )
    
    async def run_stream(self, request: AgentRunRequest) -> AsyncGenerator[AgentStep, None]:
        """
        Run the agent and stream each step
        
        Args:
            request: The request containing inputs and configuration
            
        Yields:
            Each step taken by the agent
        """
        # Get model from request or use default
        model = request.model or self.default_model
        
        # Get max steps from request or use default
        max_steps = request.max_steps or self.max_iterations
        
        # Get input prompt
        prompt = request.inputs.get("prompt", "")
        if not prompt:
            raise ProviderError("No prompt provided in inputs")
        
        # Initial context for the agent
        context = self._build_agent_context(request.inputs)
        
        # Main agent loop
        for i in range(max_steps):
            # Generate a step
            thought, action, action_input = await self._generate_step(model, context, prompt, [])
            
            # Execute the action
            if action in self.tools:
                try:
                    tool_func = self.tools[action]["func"]
                    observation = await tool_func(action_input)
                except Exception as e:
                    observation = f"Error executing {action}: {str(e)}"
            else:
                observation = f"Unknown action: {action}"
            
            # Create a step
            step = AgentStep(
                step_id=i,
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation
            )
            
            # Yield the step
            yield step
            
            # Update context with the step
            context += f"\nStep {i+1}:\nThought: {thought}\nAction: {action}\nAction Input: {json.dumps(action_input)}\nObservation: {observation}"
            
            # Check if we're done
            if action == "finish":
                break
    
    def _build_agent_context(self, inputs: Dict[str, Any]) -> str:
        """
        Build the initial context for the agent
        
        Args:
            inputs: The inputs from the request
            
        Returns:
            The agent context as a string
        """
        # Build tool descriptions
        tool_descriptions = "\n".join([
            f"- {name}: {tool['description']}"
            for name, tool in self.tools.items()
        ])
        
        # Build initial context
        context = f"""You are an AI assistant that can use tools to help answer questions.

Available tools:
{tool_descriptions}
- finish: Use this when you have the final answer

For each step:
1. Think about what to do
2. Choose a tool to use
3. Provide the input for the tool
4. Receive the observation
5. Repeat steps 1-4 until you have the answer
6. Use the "finish" tool with your final answer

User query: {inputs.get('prompt', '')}
"""
        return context
    
    async def _generate_step(self, model: str, context: str, prompt: str, steps: List[AgentStep]) -> Tuple[str, str, Dict[str, Any]]:
        """
        Generate the next step for the agent
        
        Args:
            model: The model to use
            context: The current context
            prompt: The original prompt
            steps: Previous steps
            
        Returns:
            Tuple of (thought, action, action_input)
        """
        # Create messages for the LLM
        messages = [
            ChatMessage(role="system", content=context),
            ChatMessage(role="user", content=prompt)
        ]
        
        # Add previous steps if any
        if steps:
            steps_content = "\n\n".join([
                f"Step {s.step_id+1}:\nThought: {s.thought}\nAction: {s.action}\nAction Input: {json.dumps(s.action_input)}\nObservation: {s.observation}"
                for s in steps
            ])
            messages.append(ChatMessage(role="assistant", content=steps_content))
        
        # Add instruction for next step
        messages.append(ChatMessage(role="user", content="What's your next step? Respond using the format:\nThought: <your reasoning>\nAction: <tool name>\nAction Input: <input to the tool as JSON>"))
        
        # Get the LLM response
        response = await self._run_llm(model, messages)
        
        # Parse the response
        try:
            # Extract thought, action, and action input
            thought_match = re.search(r"Thought:\s*(.*?)(?:\n|$)", response, re.DOTALL)
            action_match = re.search(r"Action:\s*(.*?)(?:\n|$)", response, re.DOTALL)
            action_input_match = re.search(r"Action Input:\s*(.*?)(?:\n|$)", response, re.DOTALL)
            
            thought = thought_match.group(1).strip() if thought_match else ""
            action = action_match.group(1).strip() if action_match else ""
            action_input_str = action_input_match.group(1).strip() if action_input_match else "{}"
            
            # Parse action input as JSON
            try:
                action_input = json.loads(action_input_str)
                if not isinstance(action_input, dict):
                    action_input = {"value": action_input}
            except json.JSONDecodeError:
                # If it's not valid JSON, treat it as a string
                action_input = {"value": action_input_str}
            
            return thought, action, action_input
        except Exception as e:
            logger.exception(f"Error parsing LLM response: {response}")
            raise ProviderError(f"Error parsing LLM response: {str(e)}")


class ChatAgent(BaseAgent):
    """Base class for chat-based agents"""
    
    def __init__(self):
        """Initialize the chat agent"""
        super().__init__()
        self.name = "ChatAgent"
        self.description = "Simple chat-based agent"
        self.default_model = "gpt-4o-mini"  # Default model for the agent
        self.system_prompt = "You are a helpful AI assistant."
    
    async def run(self, request: AgentRunRequest) -> AgentRunResponse:
        """
        Run the agent with the given request
        
        Args:
            request: The request containing inputs and configuration
            
        Returns:
            The agent's final response
        """
        # Start timing
        start_time = time.time()
        
        # Get model from request or use default
        model = request.model or self.default_model
        
        # Get input messages
        user_prompt = request.inputs.get("prompt", "")
        if not user_prompt:
            raise ProviderError("No prompt provided in inputs")
        
        chat_history = request.inputs.get("chat_history", [])
        
        # Build messages for the LLM
        messages = [
            ChatMessage(role="system", content=self.system_prompt)
        ]
        
        # Add chat history
        for message in chat_history:
            role = message.get("role", "user")
            content = message.get("content", "")
            messages.append(ChatMessage(role=role, content=content))
        
        # Add the current user prompt
        messages.append(ChatMessage(role="user", content=user_prompt))
        
        try:
            # Run the LLM
            response = await self._run_llm(model, messages)
            
            # Create a single step
            step = AgentStep(
                step_id=0,
                thought="Responding to user query",
                action="respond",
                action_input={"prompt": user_prompt},
                observation=response
            )
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            
            # Create response
            return AgentRunResponse(
                run_id="",  # Will be filled in by the runner
                agent_id=request.agent_id,
                status="completed",
                steps=[step],
                output=response,
                error=None,
                usage={},  # Will be filled in by the runner
                elapsed_time=elapsed_time
            )
        except Exception as e:
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            
            # Create response with error
            return AgentRunResponse(
                run_id="",  # Will be filled in by the runner
                agent_id=request.agent_id,
                status="failed",
                steps=[],
                output=None,
                error=str(e),
                usage={},  # Will be filled in by the runner
                elapsed_time=elapsed_time
            )
    
    async def run_stream(self, request: AgentRunRequest) -> AsyncGenerator[AgentStep, None]:
        """
        Run the agent and stream each step
        
        Args:
            request: The request containing inputs and configuration
            
        Yields:
            Each step taken by the agent
        """
        # Get model from request or use default
        model = request.model or self.default_model
        
        # Get input messages
        user_prompt = request.inputs.get("prompt", "")
        if not user_prompt:
            raise ProviderError("No prompt provided in inputs")
        
        chat_history = request.inputs.get("chat_history", [])
        
        # Build messages for the LLM
        messages = [
            ChatMessage(role="system", content=self.system_prompt)
        ]
        
        # Add chat history
        for message in chat_history:
            role = message.get("role", "user")
            content = message.get("content", "")
            messages.append(ChatMessage(role=role, content=content))
        
        # Add the current user prompt
        messages.append(ChatMessage(role="user", content=user_prompt))
        
        try:
            # Run the LLM
            response = await self._run_llm(model, messages)
            
            # Create a single step
            step = AgentStep(
                step_id=0,
                thought="Responding to user query",
                action="respond",
                action_input={"prompt": user_prompt},
                observation=response
            )
            
            # Yield the step
            yield step
        except Exception as e:
            # Create error step
            error_step = AgentStep(
                step_id=0,
                thought="Error occurred",
                action="error",
                action_input={"prompt": user_prompt},
                observation=f"Error: {str(e)}"
            )
            
            # Yield the error step
            yield error_step


import re
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory

class LangChainAgent(BaseAgent):
    """Base class for integrating LangChain agents"""
    
    def __init__(self):
        """Initialize the LangChain agent"""
        super().__init__()
        self.name = "LangChainAgent"
        self.description = "Agent using LangChain framework"
        self.tools = []  # LangChain tools
        self.default_model = "gpt-4o-mini"  # Default model for the agent
        self.memory = ConversationBufferMemory(return_messages=True)
    
    def add_tool(self, tool: Tool):
        """
        Add a LangChain tool to this agent
        
        Args:
            tool: The LangChain tool to add
        """
        self.tools.append(tool)
    
    async def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """
        Validate inputs for LangChain agent
        
        Args:
            inputs: The inputs to validate
            
        Returns:
            True if inputs are valid
        """
        # Check for required prompt
        if "prompt" not in inputs:
            return False
        return True