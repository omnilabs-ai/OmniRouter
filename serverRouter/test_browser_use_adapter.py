"""
Test script for the BrowserUseAgentAdapter.

This script demonstrates how to register and use the BrowserUseAgentAdapter
with the OmniAgents platform.
"""

import os
import sys
import asyncio
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the parent directory to sys.path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary modules
from omniAgents.agentRegistry.datamodels import AgentRunRequest
from omniAgents.agentRunner.agent_runner import OmniAgentRunner
from omniAgents.agentAdapters.browserUseAdapter import BrowserUseAgentAdapter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_browser_use_agent():
    """Test the BrowserUseAgentAdapter"""
    try:
        # Initialize agent runner
        runner = OmniAgentRunner()
        
        # Register the BrowserUseAgentAdapter
        runner.register_agent_class("browser-use", BrowserUseAgentAdapter)
        
        # Create a run request
        request = AgentRunRequest(
            agent_id="browser-use",
            model="gpt-4o", # Using OpenAI for best performance
            inputs={
                "prompt": "Compare the pricing and features of OpenAI's GPT-4o and DeepSeek-V3",
                "config": {
                    "use_vision": True,
                    "save_conversation_path": "logs/browser_use_conversation.json",
                    # Optional browser configuration
                    "browser_config": {
                        "headless": False,  # Set to True for production
                        "disable_security": True
                    }
                },
                # API keys can be passed here or read from environment
                "api_keys": {
                    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")
                }
            },
            max_steps=30,
            timeout_seconds=300,
            stream=False
        )
        
        # Create the run
        run_id = await runner.create_run(request)
        logger.info(f"Created run with ID: {run_id}")
        
        # Wait for the run to complete
        while True:
            run_info = await runner.get_run(run_id)
            if run_info.status not in ["pending", "running"]:
                break
            logger.info(f"Run status: {run_info.status}")
            await asyncio.sleep(5)
        
        # Get the completed run
        completed_run = await runner.get_run(run_id)
        
        # Print the result
        logger.info(f"Run completed with status: {completed_run.status}")
        if completed_run.status == "completed":
            logger.info(f"Output: {completed_run.output}")
        else:
            logger.error(f"Error: {completed_run.error}")
        
        # Print step details
        if completed_run.steps:
            logger.info(f"Total steps: {len(completed_run.steps)}")
            for i, step in enumerate(completed_run.steps):
                logger.info(f"Step {i+1}: {step.action}")
                
        return completed_run
        
    except Exception as e:
        logger.exception(f"Error testing BrowserUseAgentAdapter: {e}")
        raise

async def test_streaming():
    """Test streaming with the BrowserUseAgentAdapter"""
    try:
        # Initialize agent runner
        runner = OmniAgentRunner()
        
        # Register the BrowserUseAgentAdapter
        runner.register_agent_class("browser-use", BrowserUseAgentAdapter)
        
        # Create a run request with streaming
        request = AgentRunRequest(
            agent_id="browser-use",
            model="gpt-4o",
            inputs={
                "prompt": "Search for the latest AI news and summarize the top 3 stories",
                "config": {
                    "use_vision": True,
                    # Optional browser configuration
                    "browser_config": {
                        "headless": False,
                        "disable_security": True
                    }
                }
            },
            max_steps=30,
            timeout_seconds=300,
            stream=True  # Enable streaming
        )
        
        # Create the run
        run_id = await runner.create_run(request)
        logger.info(f"Created streaming run with ID: {run_id}")
        
        # For demonstration purposes, we'll poll for new steps
        # In a real application, you would use SSE or WebSockets
        previous_step_count = 0
        while True:
            run_info = await runner.get_run(run_id)
            
            # Display new steps
            if run_info.steps and len(run_info.steps) > previous_step_count:
                for i in range(previous_step_count, len(run_info.steps)):
                    step = run_info.steps[i]
                    logger.info(f"Step {i+1}: {step.action} - {step.observation[:100]}...")
                
                previous_step_count = len(run_info.steps)
            
            # Check if completed
            if run_info.status not in ["pending", "running"]:
                break
                
            await asyncio.sleep(2)
        
        # Get final result
        completed_run = await runner.get_run(run_id)
        logger.info(f"Streaming run completed with status: {completed_run.status}")
        if completed_run.status == "completed":
            logger.info(f"Final output: {completed_run.output}")
        
        return completed_run
        
    except Exception as e:
        logger.exception(f"Error testing streaming: {e}")
        raise

if __name__ == "__main__":
    # Choose which test to run
    TEST_MODE = "regular"  # Options: "regular", "streaming", "both"
    
    async def main():
        if TEST_MODE in ["regular", "both"]:
            logger.info("Testing regular mode...")
            await test_browser_use_agent()
            
        if TEST_MODE in ["streaming", "both"]:
            logger.info("\nTesting streaming mode...")
            await test_streaming()
    
    # Run the tests
    asyncio.run(main())