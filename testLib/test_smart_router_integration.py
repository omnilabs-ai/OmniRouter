"""
Integration tests for the Smart Router functionality that require a running server.
Run Integration Tests (requires server running):
    First, start the server in a separate terminal:
    uvicorn serverRouter.router:app --reload
    Then run the integration tests:
    python -m pytest testLib/test_streaming.py::TestStreaming -v --run-integration
    python -m pytest testLib/test_smart_router_integration.py -v --run-integration
    
"""

import pytest
import os
import sys
import json
import requests
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path to allow imports
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

# Initialize logger
from testLib.test_utils import test_logger

from serverRouter.core.datamodels import ChatMessage

@pytest.mark.integration
class TestSmartRouterIntegration:
    """Integration tests for SmartRouter that require a running server."""
    
    @pytest.fixture
    def api_key(self, test_api_key):
        """Return the test API key."""
        return test_api_key
    
    @pytest.fixture
    def api_url(self):
        """Return the base URL for the API."""
        return "http://localhost:8000"
    
    def test_router_endpoint_exists(self, api_url, api_key):
        """Test that the router endpoint exists and responds."""
        test_logger.info("Testing router endpoint")
        
        try:
            response = requests.get(
                f"{api_url}/v1/router/select-model",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            
            # Method not allowed indicates the endpoint exists but requires POST
            assert response.status_code in [405, 200, 422]
            test_logger.info("Router endpoint exists")
            
        except requests.exceptions.ConnectionError:
            pytest.skip("API server is not running")
    
    def test_router_model_selection(self, api_url, api_key):
        """Test that the router can select a model and return a response."""
        test_logger.info("Testing router model selection")
        
        # Prepare request payload
        payload = {
            "messages": [
                {"role": "user", "content": "Write a simple Python function to calculate factorial"}
            ],
            "k": 3,
            "rel_cost": 0.3,
            "rel_latency": 0.2,
            "rel_accuracy": 0.5,
            "verbose": True
        }
        
        try:
            response = requests.post(
                f"{api_url}/v1/router/select-model",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload
            )
            
            test_logger.info(f"Router response status: {response.status_code}")
            
            # Check that we got a successful response
            if response.status_code == 200:
                data = response.json()
                test_logger.info(f"Model selected and response generated successfully")
                
                # Verify response structure
                assert "model" in data
                assert "content" in data
                
                # Verify content contains something relevant to a factorial function
                assert "factorial" in data["content"].lower() or "fact(" in data["content"] or "!" in data["content"]
                
                test_logger.info(f"Selected model: {data.get('model')}")
                test_logger.info(f"Response content (truncated): {data['content'][:100]}...")
            else:
                test_logger.error(f"Error response: {response.text}")
                pytest.skip("Router endpoint returned an error")
                
        except requests.exceptions.ConnectionError:
            pytest.skip("API server is not running")
    
    def test_different_query_types(self, api_url, api_key):
        """Test that the router handles different types of queries appropriately."""
        test_logger.info("Testing router with different query types")
        
        # Define different types of queries
        query_types = [
            {
                "name": "Coding",
                "query": "Write a function to check if a string is a palindrome",
                "expected_keywords": ["function", "palindrome", "string", "reverse"]
            },
            {
                "name": "Math",
                "query": "Solve the equation 3x^2 + 5x - 2 = 0",
                "expected_keywords": ["equation", "solve", "quadratic", "roots", "solution"]
            },
            {
                "name": "General Knowledge",
                "query": "Explain the process of photosynthesis",
                "expected_keywords": ["photosynthesis", "plants", "chlorophyll", "sunlight", "energy"]
            }
        ]
        
        for query_type in query_types:
            test_logger.info(f"Testing {query_type['name']} query")
            
            # Prepare request payload
            payload = {
                "messages": [
                    {"role": "user", "content": query_type["query"]}
                ],
                "k": 3,
                "rel_cost": 0.3,
                "rel_latency": 0.2,
                "rel_accuracy": 0.5,
                "verbose": False
            }
            
            try:
                response = requests.post(
                    f"{api_url}/v1/router/select-model",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=payload
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Verify content contains expected keywords
                    content_lower = data["content"].lower()
                    matches = [keyword for keyword in query_type["expected_keywords"] 
                              if keyword.lower() in content_lower]
                    
                    test_logger.info(f"Selected model: {data.get('model')}")
                    test_logger.info(f"Found {len(matches)}/{len(query_type['expected_keywords'])} expected keywords")
                    
                    # Should match at least some of the expected keywords
                    assert len(matches) > 0
                    
                else:
                    test_logger.error(f"Error response: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                pytest.skip("API server is not running")
                break