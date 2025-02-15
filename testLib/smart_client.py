import pytest
import requests
from clientLib.APIClient import APIClient
from typing import Dict

class TestRouter:
    @pytest.fixture
    def client(self):
        """Create an API client instance for testing."""
        return APIClient()
    
    def test_model_selection(self, client):
        """Test the model selection endpoint with various queries and preferences."""
        
        test_cases = [
            {
                "name": "Math problem with accuracy focus",
                "params": {
                    "query": "Solve this complex calculus problem involving multiple integrals",
                    "rel_accuracy": 0.8,
                    "rel_cost": 0.2,
                    "rel_latency": 0.0,
                    "verbose": True
                }
            },
            {
                "name": "Quick response with latency focus",
                "params": {
                    "query": "What's the weather like?",
                    "rel_accuracy": 0.3,
                    "rel_cost": 0.3,
                    "rel_latency": 0.4,
                    "verbose": True
                }
            }
        ]
        
        for case in test_cases:
            # Get model recommendation
            result = client.smart_select(**case["params"])
            
            # Verify response structure
            assert isinstance(result, dict), f"Failed {case['name']}: Expected dict response for verbose mode"
            assert "selected_model" in result, f"Missing model field in response for {case['name']}"
            assert "explanation" in result, f"Missing explanation field in response for {case['name']}"
            
            # Verify model field is non-empty string
            assert isinstance(result["selected_model"], str) and result["selected_model"], "Model field should be non-empty string"
            
            # Verify explanation format
            explanation = result["explanation"]
            assert "=== Benchmark Similarities ===" in explanation, "Missing benchmark similarities section"
            assert "=== Model Metrics ===" in explanation, "Missing model metrics section"
            assert "=== Normalized Metrics ===" in explanation, "Missing normalized metrics section"
            assert "=== Final Weighted Scores ===" in explanation, "Missing final scores section"
            assert "Chosen Model:" in explanation, "Missing chosen model in explanation"
    

if __name__ == "__main__":
    pytest.main([__file__]) 