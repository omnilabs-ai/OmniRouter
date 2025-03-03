import json
import re
import requests
import asyncio
from typing import Dict, Any, List, Optional, AsyncGenerator

from serverRouter.omniAgents.agentRunner.base_agents import ReActAgent
from serverRouter.omniAgents.agentRegistry.datamodels import AgentCapability

class WebSearchAgent(ReActAgent):
    """Agent for web search and information retrieval"""
    
    def __init__(self):
        """Initialize the web search agent"""
        super().__init__()
        self.name = "WebSearchAgent"
        self.description = "Agent that can search the web for information"
        self.default_model = "gpt-4o"  # Use a more capable model for complex tasks
        self.capabilities = [
            AgentCapability.WEB_SEARCH,
            AgentCapability.PLANNING
        ]
        self.tags = ["research", "information-retrieval", "web"]
        
        # Register tools
        self.register_tool(
            "search",
            self.search_web,
            "Search the web for information about a query"
        )
        self.register_tool(
            "fetch_page",
            self.fetch_page,
            "Fetch the content of a specific web page"
        )
        self.register_tool(
            "finish",
            self.finish,
            "Complete the task and provide the final answer"
        )
    
    async def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """
        Validate inputs for the web search agent
        
        Args:
            inputs: The inputs to validate
            
        Returns:
            True if inputs are valid
        """
        return "prompt" in inputs
    
    async def search_web(self, action_input: Dict[str, Any]) -> str:
        """
        Search the web for information
        
        Args:
            action_input: Dictionary containing "query" key
            
        Returns:
            Search results as a string
        """
        query = action_input.get("query", "")
        if not query:
            return "No query provided"
        
        # In a real implementation, you would use a search API like Google, Bing, or DuckDuckGo
        # This is a simplified mock implementation
        try:
            # Mock search results
            search_results = [
                {
                    "title": f"Result for {query} - 1",
                    "snippet": f"This is a snippet about {query} from the first result.",
                    "url": f"https://example.com/result1?q={query}"
                },
                {
                    "title": f"Result for {query} - 2",
                    "snippet": f"Another snippet about {query} from the second result.",
                    "url": f"https://example.com/result2?q={query}"
                },
                {
                    "title": f"Result for {query} - 3",
                    "snippet": f"More information about {query} from the third result.",
                    "url": f"https://example.com/result3?q={query}"
                }
            ]
            
            # Format the results
            results_text = "\n\n".join([
                f"Title: {result['title']}\nURL: {result['url']}\nSnippet: {result['snippet']}"
                for result in search_results
            ])
            
            return f"Search results for '{query}':\n\n{results_text}"
        except Exception as e:
            return f"Error searching the web: {str(e)}"
    
    async def fetch_page(self, action_input: Dict[str, Any]) -> str:
        """
        Fetch the content of a web page
        
        Args:
            action_input: Dictionary containing "url" key
            
        Returns:
            Page content as a string
        """
        url = action_input.get("url", "")
        if not url:
            return "No URL provided"
        
        # In a real implementation, you would make an HTTP request and parse the HTML
        # This is a simplified mock implementation
        try:
            # Mock page content based on URL
            if "result1" in url:
                return f"This is the content of the page at {url}. It contains detailed information about the query."
            elif "result2" in url:
                return f"This is the content of the page at {url}. It has additional facts and figures about the topic."
            elif "result3" in url:
                return f"This is the content of the page at {url}. It provides a different perspective on the subject."
            else:
                return f"This is generic content for the page at {url}."
        except Exception as e:
            return f"Error fetching the page: {str(e)}"
    
    async def finish(self, action_input: Dict[str, Any]) -> str:
        """
        Complete the task and provide the final answer
        
        Args:
            action_input: Dictionary containing "answer" key
            
        Returns:
            The final answer
        """
        answer = action_input.get("answer", "")
        if not answer:
            return "No answer provided"
        
        return answer


class CodeGenerationAgent(ReActAgent):
    """Agent for generating and running code"""
    
    def __init__(self):
        """Initialize the code generation agent"""
        super().__init__()
        self.name = "CodeGenerationAgent"
        self.description = "Agent that can generate and run code to solve problems"
        self.default_model = "gpt-4o"  # Use a more capable model for coding
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
            "Run generated code and return the output"
        )
        self.register_tool(
            "finish",
            self.finish,
            "Complete the task and provide the final answer"
        )
    
    async def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """
        Validate inputs for the code generation agent
        
        Args:
            inputs: The inputs to validate
            
        Returns:
            True if inputs are valid
        """
        return "prompt" in inputs
    
    async def generate_code(self, action_input: Dict[str, Any]) -> str:
        """
        Generate code to solve a problem
        
        Args:
            action_input: Dictionary containing "task" and "language" keys
            
        Returns:
            Generated code as a string
        """
        task = action_input.get("task", "")
        language = action_input.get("language", "python")
        
        if not task:
            return "No task provided"
        
        # In a real implementation, you would generate code using an LLM
        # This is a simplified mock implementation for Python code generation
        if language.lower() == "python":
            if "fibonacci" in task.lower():
                return """
def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    
    return fib

# Generate the first 10 Fibonacci numbers
result = fibonacci(10)
print(result)
"""
            elif "sort" in task.lower():
                return """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

# Sort an example array
example = [64, 34, 25, 12, 22, 11, 90]
result = bubble_sort(example)
print(result)
"""
            else:
                return """
# Simple example code
def process_task(task):
    return f"Processed: {task}"

result = process_task("{}") 
print(result)
""".format(task)
        else:
            return f"Code generation for {language} is not supported yet."
    
    async def run_code(self, action_input: Dict[str, Any]) -> str:
        """
        Run generated code and return the output
        
        Args:
            action_input: Dictionary containing "code" and "language" keys
            
        Returns:
            Output from running the code
        """
        code = action_input.get("code", "")
        language = action_input.get("language", "python")
        
        if not code:
            return "No code provided"
        
        # In a real implementation, you would use a sandboxed environment to run the code
        # This is a simplified mock implementation that only supports Python
        if language.lower() == "python":
            try:
                # Mock execution result based on code content
                if "fibonacci" in code:
                    return "[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]"
                elif "bubble_sort" in code:
                    return "[11, 12, 22, 25, 34, 64, 90]"
                else:
                    # Extract the task from the code if possible
                    task_match = re.search(r'process_task\("([^"]+)"\)', code)
                    task = task_match.group(1) if task_match else "unknown task"
                    return f"Processed: {task}"
            except Exception as e:
                return f"Error running code: {str(e)}"
        else:
            return f"Code execution for {language} is not supported yet."
    
    
    async def finish(self, action_input: Dict[str, Any]) -> str:
        """
        Complete the task and provide the final answer
        
        Args:
            action_input: Dictionary containing "answer" key
            
        Returns:
            The final answer
        """
        # Check if the answer is directly in action_input
        if isinstance(action_input, str):
            return action_input
            
        # Or check different possible keys
        for key in ["answer", "value", "result"]:
            if key in action_input and action_input[key]:
                return action_input[key]
        
        return "No answer provided"