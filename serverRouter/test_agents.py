import asyncio
import json
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Imports for agent testing
from omniAgents.agentRunner.agent_runner import OmniAgentRunner
from omniAgents.agentRegistry.datamodels import AgentRunRequest
from omniAgents.implementations.web_search_agent import WebSearchAgent
from omniAgents.implementations.code_agent import CodeGenerationAgent

# Imports for provider initialization
from core.config import PROVIDERS
from core.datamodels import ModelProvider
from providers.openai.provider import OpenAIProvider
from providers.anthropic.provider import AnthropicProvider

# Initialize providers for testing
def initialize_test_providers():
    print("Initializing providers for testing...")
    try:
        # Initialize OpenAI provider
        PROVIDERS[ModelProvider.OPENAI] = OpenAIProvider()
        print("OpenAI provider initialized")
        
        # Optionally initialize other providers if needed
        PROVIDERS[ModelProvider.ANTHROPIC] = AnthropicProvider()
        print("Anthropic provider initialized")

    except Exception as e:
        print(f"Error initializing providers: {str(e)}")
        raise

async def test_web_search_agent():
    print("\n===== Testing Web Search Agent =====")
    
    # Create the runner
    runner = OmniAgentRunner()
    
    # Register the agent
    runner.register_agent_class("web-search-agent", WebSearchAgent)
    
    # Create a run request
    request = AgentRunRequest(
        agent_id="web-search-agent",
        inputs={"prompt": "What is the capital of France?"},
        stream=False
    )
    
    # Run the agent
    print("Creating run...")
    run_id = await runner.create_run(request)
    print(f"Run created with ID: {run_id}")
    
    # Get the results
    print("Waiting for results...")
    run = await runner.get_run(run_id)
    
    # Print the results
    print("\nRun results:")
    print(f"Status: {run.status}")
    print(f"Output: {run.output}")
    print(f"Time taken: {run.elapsed_time:.2f} seconds")
    
    # Print the steps
    print("\nSteps:")
    for step in (run.steps or []):
        print(f"\nStep {step.step_id + 1}:")
        print(f"Thought: {step.thought}")
        print(f"Action: {step.action}")
        print(f"Action Input: {json.dumps(step.action_input)}")
        print(f"Observation: {step.observation}")

async def test_code_agent():
    print("\n===== Testing Code Generation Agent =====")
    
    # Create the runner
    runner = OmniAgentRunner()
    
    # Register the agent
    runner.register_agent_class("code-agent", CodeGenerationAgent)
    
    # Create a run request
    request = AgentRunRequest(
        agent_id="code-agent",
        inputs={"prompt": "Write a function that prints 'Hello, World!'"},
        stream=False
    )
    
    # Run the agent
    print("Creating run...")
    run_id = await runner.create_run(request)
    print(f"Run created with ID: {run_id}")
    
    # Get the results
    print("Waiting for results...")
    run = await runner.get_run(run_id)
    
    # Print the results
    print("\nRun results:")
    print(f"Status: {run.status}")
    print(f"Output: {run.output}")
    print(f"Time taken: {run.elapsed_time:.2f} seconds")
    
    # Print the steps
    print("\nSteps:")
    for step in (run.steps or []):
        print(f"\nStep {step.step_id + 1}:")
        print(f"Thought: {step.thought}")
        print(f"Action: {step.action}")
        print(f"Action Input: {json.dumps(step.action_input)}")
        print(f"Observation: {step.observation}")

async def main():
    try:
        # Initialize providers before running tests
        initialize_test_providers()
        
        # Run the tests
        await test_web_search_agent()
        await test_code_agent()
    except Exception as e:
        print(f"Error in main: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting agent tests...")
    asyncio.run(main())