"""
Simple test script for the BrowserUseAgentAdapter.

This script uses minimal configuration to test the adapter.
"""

import os
import sys
import asyncio
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the current directory to sys.path to import project modules
sys.path.append(os.path.abspath("."))

# Import necessary modules
from omniAgents.agentRegistry.datamodels import AgentRunRequest
from omniAgents.agentRunner.agent_runner import OmniAgentRunner
from omniAgents.agentAdapters.browserUseAdapter import BrowserUseAgentAdapter

async def test_browser_use_agent_simple():
    """Test the BrowserUseAgentAdapter with minimal configuration"""
    try:
        # Initialize agent runner
        runner = OmniAgentRunner()
        
        # Register the BrowserUseAgentAdapter
        runner.register_agent_class("browser-use", BrowserUseAgentAdapter)
        
        # Create a simple run request with minimal configuration
        request = AgentRunRequest(
            agent_id="browser-use",
            model="gpt-4o",
            inputs={
                "prompt": "Search for 'weather today' on Google and tell me the current temperature",
                # Minimal configuration
                "config": {
                    "use_vision": True
                }
            },
            max_steps=10,
            timeout_seconds=120,
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
        
        return completed_run
        
    except Exception as e:
        logger.exception(f"Error testing BrowserUseAgentAdapter: {e}")
        raise

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_browser_use_agent_simple())