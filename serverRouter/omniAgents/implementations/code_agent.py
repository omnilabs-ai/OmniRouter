import json
import re
from typing import Dict, Any, List, Optional, AsyncGenerator

from serverRouter.omniAgents.agentRunner.base_agents import ReActAgent
from serverRouter.omniAgents.agentRegistry.datamodels import AgentCapability

class CodeGenerationAgent(ReActAgent):
    """Agent for generating and running code"""
    
    def __init__(self):
        """Initialize the code generation agent"""
        super().__init__()
        self.name = "CodeGenerationAgent"
        self.description = "Agent that can generate code to solve problems"
        self.default_model = "gpt-4o-mini"
        self.capabilities = [
            AgentCapability.CODE_EXECUTION,
            AgentCapability.PLANNING
        ]
        self.tags = ["coding", "development", "programming"]
        
        # Register tools
        self.register_tool(
            "generate_code",
            self.generate_code,
            "Generate code to solve a problem"
        )
        self.register_tool(
            "run_code",
            self.run_code,
            "Run the generated code and return the output"
        )
        self.register_tool(
            "finish",
            self.finish,
            "Complete the task with a final answer"
        )
    
    async def generate_code(self, action_input: Dict[str, Any]) -> str:
        """
        Generate code based on a description
        
        Args:
            action_input: Dictionary with "task" and "language" keys
            
        Returns:
            Generated code as a string
        """
        task = action_input.get("task", "")
        language = action_input.get("language", "python")
        
        if not task:
            return "No task provided"
        
        # For testing, return a simple code example
        if language.lower() == "python":
            return """
def hello_world():
    return "Hello, World!"

print(hello_world())
"""
        else:
            return f"// Code generation for {language} not implemented yet"
    
    async def run_code(self, action_input: Dict[str, Any]) -> str:
        """
        Run code (mock implementation)
        
        Args:
            action_input: Dictionary with "code" and "language" keys
            
        Returns:
            Output from running the code
        """
        code = action_input.get("code", "")
        language = action_input.get("language", "python")
        
        if not code:
            return "No code provided"
        
        # For testing, return a mock output
        if "hello_world" in code:
            return "Hello, World!"
        else:
            return "Code execution result would appear here"
    
    async def finish(self, action_input: Dict[str, Any]) -> str:
        """
        Complete the task with a final answer
        
        Args:
            action_input: Dictionary with "answer" key
            
        Returns:
            The final answer
        """
        answer = action_input.get("answer", "")
        if not answer:
            return "No answer provided"
        
        return answer