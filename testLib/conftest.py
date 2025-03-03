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

# Import fix_imports module to fix path issues
try:
    from serverRouter.fix_imports import fix_imports
    fix_imports()
    test_logger.info("Import fixes applied successfully")
except ImportError:
    test_logger.warning("Could not import fix_imports module")

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

# Skip integration tests by default unless --run-integration flag is provided
def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-integration", action="store_true", default=False, help="run integration tests"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle markers."""
    if not config.getoption("--run-integration"):
        skip_integration = pytest.mark.skip(reason="need --run-integration option to run")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)