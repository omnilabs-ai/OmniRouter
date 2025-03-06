"""
PyTest configuration file for OmniRouter tests.
This file contains fixtures and settings for all tests.
"""

import os
import sys
import pytest
from pathlib import Path

# Add parent directory to path to allow imports
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

# Import utilities
from testLib.test_utils import test_logger


@pytest.fixture(scope="session", autouse=True)
def setup_environment():
    """Setup environment variables and paths for tests."""
    # Set environment variables for testing
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "dummy-key-for-testing")
    os.environ["ANTHROPIC_API_KEY"] = os.environ.get("ANTHROPIC_API_KEY", "dummy-key-for-testing")
    os.environ["GEMINI_API_KEY"] = os.environ.get("GEMINI_API_KEY", "dummy-key-for-testing")
    
    # Log test environment setup
    test_logger.info("Test environment set up")
    test_logger.info(f"Python version: {sys.version}")
    test_logger.info(f"Working directory: {os.getcwd()}")
    test_logger.info(f"Parent directory added to path: {parent_dir}")

@pytest.fixture(scope="session")
def test_api_key():
    """Return the test API key for the API."""
    return "test-sk1o83e"

# Define custom markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark a test as requiring a running API server"
    )

# Add option to run integration tests, but now they run by default
def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--skip-integration", action="store_true", default=False, 
        help="skip integration tests that require a running server"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle markers."""
    if config.getoption("--skip-integration"):
        skip_integration = pytest.mark.skip(reason="integration tests skipped with --skip-integration")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)
    # Else, run them by default - now integration tests run unless specifically skipped