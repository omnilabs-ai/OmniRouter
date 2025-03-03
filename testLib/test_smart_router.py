"""
Test suite for the Smart Router functionality.
"""

import pytest
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path to allow imports
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

# Initialize logger
from testLib.test_utils import test_logger

# Import required modules
from serverRouter.core.datamodels import ChatMessage, SmartRouterRequest
from serverRouter.smartRouter.smart_router import SmartRouter


class TestSmartRouter:
    """Test class for SmartRouter functionality"""
    
    @pytest.fixture
    def router(self):
        """Fixture to create a SmartRouter instance."""
        # Path to benchmark embeddings relative to test file
        embeddings_path = os.path.join(
            parent_dir, "serverRouter", "smartRouter", "benchmark_embeddings.pkl"
        )
        
        # Create router instance
        try:
            return SmartRouter(embeddings_path=embeddings_path)
        except Exception as e:
            test_logger.error(f"Error initializing SmartRouter: {e}")
            # Return a basic SmartRouter instance even if embeddings file isn't found
            return SmartRouter()
    
    def test_initialize_router(self, router):
        """Test that the router can be initialized."""
        assert router is not None
        test_logger.info("SmartRouter initialized successfully")
    
    def test_identify_tasks(self, router):
        """Test the task identification functionality."""
        # Define test messages for different tasks
        test_cases = [
            {
                "messages": [ChatMessage(role="user", content="Write a Python function to calculate factorial")],
                "expected_task": "coding"
            },
            {
                "messages": [ChatMessage(role="user", content="Solve this equation: 2x^2 + 3x - 5 = 0")],
                "expected_task": "math"
            },
            {
                "messages": [ChatMessage(role="user", content="Explain quantum mechanics and its applications")],
                "expected_task": "science"
            },
            {
                "messages": [ChatMessage(role="user", content="Analyze the logical fallacies in this argument")],
                "expected_task": "reasoning"
            },
            {
                "messages": [ChatMessage(role="user", content="What is the capital of France?")],
                "expected_task": "general_knowledge"
            },
            {
                "messages": [ChatMessage(role="user", content="Write a short story about a detective")],
                "expected_task": "creative_writing"
            }
        ]
        
        for tc in test_cases:
            # Get task scores
            task_scores = router.identify_tasks(tc["messages"])
            test_logger.info(f"Message: '{tc['messages'][0].content}'")
            test_logger.info(f"Task scores: {task_scores}")
            
            # Check if expected task is among the top scores
            top_task = max(task_scores.items(), key=lambda x: x[1])[0]
            test_logger.info(f"Top task identified: {top_task}")
            
            # Assert that the expected task has a significant score
            assert task_scores[tc["expected_task"]] > 0.1, f"Expected {tc['expected_task']} to have a significant score"
    
    def test_compute_benchmark_weights(self, router):
        """Test that benchmark weights are computed correctly."""
        # Define task scores
        task_scores = {
            "coding": 0.7,
            "math": 0.2,
            "general_knowledge": 0.1
        }
        
        # Compute benchmark weights
        benchmark_weights = router.compute_benchmark_weights(task_scores)
        test_logger.info(f"Task scores: {task_scores}")
        test_logger.info(f"Benchmark weights: {benchmark_weights}")
        
        # Verify that weights sum to approximately 1.0
        assert abs(sum(benchmark_weights.values()) - 1.0) < 0.01
        
        # Verify that HumanEval has high weight for coding task
        assert "HumanEval" in benchmark_weights
        assert benchmark_weights["HumanEval"] >= 0.4
    
    def test_score_models(self, router):
        """Test that models are scored correctly based on benchmarks."""
        # Define benchmark weights
        benchmark_weights = {
            "HumanEval": 0.6,
            "MMLU": 0.3,
            "MATH": 0.1
        }
        
        # Get model scores with balanced preferences
        model_scores = router.score_models(
            benchmark_weights,
            rel_cost=0.33,
            rel_latency=0.33,
            rel_accuracy=0.34
        )
        
        test_logger.info(f"Model scores (balanced): {json.dumps({k: v['score'] for k, v in model_scores.items()}, indent=2)}")
        
        # Verify that we have scores for multiple models
        assert len(model_scores) > 0
        
        # Get model scores prioritizing accuracy
        model_scores_accuracy = router.score_models(
            benchmark_weights,
            rel_cost=0.1,
            rel_latency=0.1,
            rel_accuracy=0.8
        )
        
        test_logger.info(f"Model scores (accuracy): {json.dumps({k: v['score'] for k, v in model_scores_accuracy.items()}, indent=2)}")
        
        # Verify that we have scores for multiple models
        assert len(model_scores_accuracy) > 0
    
    def test_model_selection(self, router):
        """Test the full model selection process."""
        # Define test messages
        messages = [
            ChatMessage(role="user", content="Write a Python function to sort a list of numbers")
        ]
        
        # Create request with default preferences
        request = SmartRouterRequest(
            messages=messages,
            k=3,
            model_names=None,
            rel_cost=0.5,
            rel_latency=0.0,
            rel_accuracy=0.5,
            verbose=True
        )
        
        # Get model recommendations
        result = router.select_models(request)
        
        # Log the results
        test_logger.info(f"Selected models: {result['selected_models']}")
        if result.get("explanation"):
            test_logger.info(f"Explanation: {result['explanation'][:100]}...")  # Truncated for brevity
        
        # Verify that models were selected
        assert len(result["selected_models"]) > 0
        assert result["selected_models"][0] in router.models
        
        # Verify that we have model details
        assert "model_details" in result
        assert len(result["model_details"]) > 0
    
    def test_different_user_preferences(self, router):
        """Test model selection with different user preferences."""
        # Define test messages
        messages = [
            ChatMessage(role="user", content="Analyze the performance of different sorting algorithms")
        ]
        
        # Define preference profiles
        profiles = [
            {"name": "Balanced", "cost": 0.33, "latency": 0.33, "accuracy": 0.34},
            {"name": "Cost-Conscious", "cost": 0.8, "latency": 0.1, "accuracy": 0.1},
            {"name": "High-Performance", "cost": 0.1, "latency": 0.1, "accuracy": 0.8}
        ]
        
        for profile in profiles:
            # Create request with profile preferences
            request = SmartRouterRequest(
                messages=messages,
                k=3,
                model_names=None,
                rel_cost=profile["cost"],
                rel_latency=profile["latency"],
                rel_accuracy=profile["accuracy"],
                verbose=False
            )
            
            # Get model recommendations
            result = router.select_models(request)
            
            # Log the results
            test_logger.info(f"Profile: {profile['name']}")
            test_logger.info(f"Selected models: {result['selected_models']}")
            
            # Verify that models were selected
            assert len(result["selected_models"]) > 0
            
            # Store top model for comparison
            profile["top_model"] = result["selected_models"][0]
        
        # Compare results across profiles
        test_logger.info("Comparison of top models across profiles:")
        for profile in profiles:
            test_logger.info(f"{profile['name']}: {profile['top_model']}")
        
        # Different preferences should ideally lead to different model selections
        # but we can't guarantee this in all cases, so we don't assert it