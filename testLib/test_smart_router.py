import time
import json
from statistics import mean, median, stdev
from datetime import datetime
from typing import Dict, List, Any
import os
from dotenv import load_dotenv

from .test_core import BaseTest
from serverRouter.core.datamodels import ModelProvider
from serverRouter.smartRouter.smart_router import SmartRouter
from serverRouter.smartRouter.session_tracker import SessionTracker
from serverRouter.smartRouter.task_vector_db import TaskVectorDB

# Load environment variables from .env file
load_dotenv()

class TestSmartRouter(BaseTest):
    """
    Test the SmartRouter functionality including model selection and response performance.
    
    This test class measures:
    1. Smart router model selection accuracy
    2. Response time for model selection
    3. Response time for actual model completions of selected models
    4. Quality of responses from selected models
    """
    
    # Set this to False to skip making actual API calls that incur costs
    # Can be overridden by setting environment variable TEST_COMPLETIONS=1
    TEST_COMPLETIONS = os.environ.get("TEST_COMPLETIONS", "1") == "1"
    
    def setup_method(self):
        """Set up test environment with proper API keys"""
        super().setup_method()
        
        # Get the api key for authentication
        self.omni_api_key = os.environ.get("OMNI_API_KEY", "test-sk1o83e")
        
        # Set the authorization header for the API gateway
        self.client.headers.update({
            "Authorization": f"Bearer {self.omni_api_key}"
        })
        
        # Set OMNI_DEV=1 to bypass API key verification in routes/utils.py
        os.environ["OMNI_DEV"] = "1"
        
        # Create dynamic provider-to-env-var mapping from ModelProvider enum
        self.provider_env_mapping = {}
        for provider in ModelProvider:
            # Convert UPPERCASE or CamelCase to lowercase for env variables
            provider_env_key = f"{provider.value.upper()}_API_KEY"
            self.provider_env_mapping[provider.value.lower()] = provider_env_key
        
        # Auto-generate model prefix mappings from known patterns
        # This will detect most common providers without manual updates
        self.model_prefix_to_provider = {
            # Common model prefixes - can be extended with new prefixes
            "gpt": ModelProvider.OPENAI.value.lower(),
            "claude": ModelProvider.ANTHROPIC.value.lower(),
            "gemini": ModelProvider.GEMINI.value.lower(),
            "deepseek": ModelProvider.DEEPSEEK.value.lower(),
            "meta-llama": ModelProvider.TOGETHER.value.lower(),
            "Qwen": ModelProvider.TOGETHER.value.lower(),
            "mistralai": ModelProvider.TOGETHER.value.lower(),
            "learnlm": ModelProvider.GEMINI.value.lower(),
        }
        
        # Auto-detect other providers by matching model prefix with provider name
        # This helps catch new providers without manual updates
        for provider in ModelProvider:
            provider_lower = provider.value.lower()
            # Add the provider's own name as a prefix if not already defined
            # For example: if a provider is named "NewProvider", it will match models starting with "newprovider"
            if provider_lower not in self.model_prefix_to_provider.values():
                self.model_prefix_to_provider[provider_lower] = provider_lower
        
        # Load API keys directly from .env
        self.api_keys = {}
        for provider, env_var in self.provider_env_mapping.items():
            self.api_keys[env_var] = os.environ.get(env_var, "")
        
        # Log which providers are available for testing
        self.logger.info("Testing with available API keys:")
        for provider, key in self.api_keys.items():
            if key:
                masked_key = key[:4] + "..." + key[-4:] if len(key) > 8 else "***" 
                self.logger.info(f"  ✓ {provider} available: {masked_key}")
            else:
                self.logger.warning(f"  ✗ {provider} not available")
    
    # Test cases with different query types to test task classification
    test_cases = [
        {
            "messages": [{"role": "user", "content": "Write a Python function to calculate factorial"}],
            "expected_task": "coding"
        },
        {
            "messages": [{"role": "user", "content": "Solve this equation: 2x^2 + 3x - 5 = 0"}],
            "expected_task": "math"
        },
        {
            "messages": [{"role": "user", "content": "Explain quantum mechanics and its applications"}],
            "expected_task": "science"
        },
        {
            "messages": [{"role": "user", "content": "Analyze the logical fallacies in this argument: All birds can fly. Penguins are birds. Therefore, penguins can fly."}],
            "expected_task": "reasoning"
        },
        {
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
            "expected_task": "general_knowledge"
        },
        {
            "messages": [{"role": "user", "content": "Write a short story about a detective solving a mysterious case"}],
            "expected_task": "creative_writing"
        }
    ]
    
    def test_smart_router_selection(self):
        """Test the SmartRouter's model selection capability and response time."""
        self.logger.info("Testing Smart Router Model Selection")
        
        selection_times = []
        results = {}
        
        # Configure preferences for testing
        config = {
            "rel_cost": 0.3,           # Emphasize cost somewhat
            "rel_latency": 0.3,        # Emphasize latency somewhat
            "rel_accuracy": 0.4,       # Emphasize accuracy more
            "k": 3,                    # Get top 3 models
            "verbose": True            # Get detailed explanations
        }
        
        for i, test_case in enumerate(self.test_cases):
            case_id = f"case_{i+1}_{test_case['expected_task']}"
            self.logger.info(f"Testing {case_id}: {test_case['messages'][0]['content'][:50]}...")
            
            # Create smart router request
            request = {
                "messages": test_case["messages"],
                **config
            }
            
            # Measure time for model selection
            start_time = time.time()
            response = self.client.post("/smart/select-model", json=request)
            selection_time = time.time() - start_time
            selection_times.append(selection_time)
            
            # Verify response
            assert response.status_code == 200, f"Failed with status {response.status_code}: {response.text}"
            selection_data = response.json()
            
            # Store results
            results[case_id] = {
                "selection_time": selection_time,
                "selected_models": selection_data["selected_models"],
                "task_classifications": selection_data["task_classifications"],
                "model_scores": selection_data.get("model_scores"),
                "model_responses": []
            }
            
            self.logger.info(f"  Selected models: {selection_data['selected_models']}")
            self.logger.info(f"  Task classifications: {selection_data['task_classifications']}")
            self.logger.info(f"  Selection time: {selection_time:.3f} seconds")
            
            # Verify selected models - at least one model should be selected
            assert len(selection_data["selected_models"]) > 0, "No models selected"
            
            # Verify if expected task is in classifications with reasonable score
            if "task_classifications" in selection_data:
                found_match = False
                for task, score in selection_data["task_classifications"].items():
                    if test_case["expected_task"] in task.lower():
                        found_match = True
                        assert score > 0.3, f"Expected task {test_case['expected_task']} has low score {score}"
                        break
                assert found_match, f"Expected task {test_case['expected_task']} not found in classifications"
        
        # Report overall selection performance
        self.logger.info("Smart Router Selection Performance Summary:")
        self.logger.info(f"  Average selection time: {mean(selection_times):.3f} seconds")
        self.logger.info(f"  Median selection time: {median(selection_times):.3f} seconds")
        if len(selection_times) > 1:
            self.logger.info(f"  Selection time std dev: {stdev(selection_times):.3f} seconds")
        self.logger.info(f"  Min selection time: {min(selection_times):.3f} seconds")
        self.logger.info(f"  Max selection time: {max(selection_times):.3f} seconds")
        
        # Write detailed results to file for analysis
        self._write_results("selection_results.json", results)
        
        # Return results for use in test_smart_router_completions
        return results
    
    def test_smart_router_completions(self):
        """Test completions from models selected by the SmartRouter."""
        self.logger.info("Testing Smart Router Completions")
            
        # Skip if completions testing is disabled
        if not self.TEST_COMPLETIONS:
            self.logger.info("Skipping completions testing - TEST_COMPLETIONS not enabled")
            self.logger.info("To test completions, set TEST_COMPLETIONS=1 environment variable")
            return
            
        # First get model selections
        selection_results = self.test_smart_router_selection()
        completion_times = []
        
        # Save original environment variables
        original_env = {}
        for key in self.api_keys.keys():
            original_env[key] = os.environ.get(key, "")
        
        # Set all API keys in environment for the duration of the test
        # This ensures they're available to the providers
        for env_var, key_value in self.api_keys.items():
            if key_value:
                os.environ[env_var] = key_value
                self.logger.info(f"Setting {env_var}={key_value[:4]}...{key_value[-4:]}")
        
        # Set OMNI_DEV=1 to bypass API key verification
        os.environ["OMNI_DEV"] = "1"
                
        for case_id, selection in selection_results.items():
            self.logger.info(f"\nTesting completions for {case_id}")
            test_case_idx = int(case_id.split("_")[1]) - 1
            test_case = self.test_cases[test_case_idx]
            
            model_responses = []
            
            # Get completions from top 2 models (or all if less than 2)
            models_to_test = selection["selected_models"][:2]
            
            for model_id in models_to_test:
                self.logger.info(f"\n  Getting completion from model: {model_id}")
                
                # Find matching provider
                provider_key = None
                model_prefix = None
                
                # Extract the model prefix (e.g., "gpt" from "gpt-4")
                for prefix in self.model_prefix_to_provider:
                    if model_id.lower().startswith(prefix.lower()):
                        model_prefix = prefix
                        break
                
                # If not found by prefix, look for partial matches in model_id
                if not model_prefix:
                    for prefix in self.model_prefix_to_provider:
                        if prefix.lower() in model_id.lower():
                            model_prefix = prefix
                            break
                
                # Special case for Qwen models which use Together API
                if "qwen" in model_id.lower():
                    model_prefix = "Qwen"
                
                # Map prefix to provider and environment variable
                if model_prefix:
                    provider = self.model_prefix_to_provider.get(model_prefix)
                    if provider:
                        provider_key = self.provider_env_mapping.get(provider)
                        self.logger.info(f"    Using {provider_key} for {model_id} (provider: {provider})")
                else:
                    self.logger.warning(f"    Could not determine provider for {model_id}")
                
                # Skip if no API key is available for this provider
                if provider_key and not self.api_keys.get(provider_key):
                    self.logger.warning(f"    Skipping {model_id} - no {provider_key} available")
                    model_responses.append({
                        "model": model_id,
                        "error": f"No API key available for this provider ({provider_key})"
                    })
                    continue
                
                # Create custom headers to ensure proper API key passing
                headers = {
                    "Authorization": f"Bearer {self.omni_api_key}",
                    "X-Provider-API-Key": self.api_keys.get(provider_key, ""),
                    "X-Provider-Name": provider if provider else "",
                    "Content-Type": "application/json"
                }
                
                completion_request = {
                    "model": model_id,
                    "messages": test_case["messages"],
                    "temperature": 0.7,
                    "max_tokens": 200
                }
                
                try:
                    # Measure completion time
                    start_time = time.time()
                    
                    # Make direct API call to /v1/chat/completions
                    completion_response = self.client.post(
                        "/v1/chat/completions", 
                        json=completion_request,
                        headers=headers
                    )
                    
                    completion_time = time.time() - start_time
                    completion_times.append(completion_time)
                    
                    # Log response status for debugging
                    self.logger.info(f"    Response status: {completion_response.status_code}")
                    
                    if completion_response.status_code == 200:
                        # Successful response
                        completion_data = completion_response.json()
                        response_text = completion_data.get("content", "")
                        
                        # Store response data
                        model_responses.append({
                            "model": model_id,
                            "time": completion_time,
                            "response": response_text[:500] + "..." if len(response_text) > 500 else response_text,
                            "full_response": response_text
                        })
                        
                        self.logger.info(f"    Response time: {completion_time:.3f} seconds")
                        self.logger.info(f"    Response: {response_text[:200]}...")
                    else:
                        # Try fallback method with direct API key
                        self.logger.warning(f"    First attempt failed with status {completion_response.status_code}")
                        
                        # Add API key directly to request
                        completion_request["api_key"] = self.api_keys.get(provider_key, "")
                        
                        # Second attempt 
                        retry_start = time.time()
                        retry_response = self.client.post(
                            "/v1/chat/completions", 
                            json=completion_request,
                            headers=headers
                        )
                        retry_time = time.time() - retry_start
                        
                        if retry_response.status_code == 200:
                            # Successful retry
                            retry_data = retry_response.json()
                            retry_text = retry_data.get("content", "")
                            
                            # Update completion time
                            completion_time = retry_time
                            completion_times.append(completion_time)
                            
                            # Store response data
                            model_responses.append({
                                "model": model_id,
                                "time": completion_time,
                                "response": retry_text[:500] + "..." if len(retry_text) > 500 else retry_text,
                                "full_response": retry_text
                            })
                            
                            self.logger.info(f"    Response time (retry): {completion_time:.3f} seconds")
                            self.logger.info(f"    Response: {retry_text[:200]}...")
                        else:
                            # Both attempts failed
                            error_msg = f"Completion failed: {retry_response.text}"
                            self.logger.error(f"    Error getting completion from {model_id}: {error_msg}")
                            model_responses.append({
                                "model": model_id,
                                "error": error_msg
                            })
                            
                except Exception as e:
                    self.logger.error(f"    Error getting completion from {model_id}: {str(e)}")
                    model_responses.append({
                        "model": model_id,
                        "error": str(e)
                    })
            
            # Update results with model responses
            selection_results[case_id]["model_responses"] = model_responses
        
        # Restore original environment variables
        for key, value in original_env.items():
            if value:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]
                
        # Generate comprehensive report
        self._write_results("completion_results.json", selection_results)
        
        # Generate performance report
        if completion_times:
            self._generate_performance_report(selection_results, completion_times)
        else:
            # Generate selection-only report if no completions were successful
            self._generate_performance_report(selection_results)
    
    def test_session_tracker(self):
        """Test the SessionTracker functionality."""
        self.logger.info("Testing Session Tracker")
        
        # Create session tracker
        session_tracker = SessionTracker(session_timeout_seconds=300)  # 5 minutes timeout for testing
        
        # Create a new session
        session_id = "test-session-1"
        user_id = "test-user-1"
        session = session_tracker.get_session(session_id, user_id)
        
        # Check session initialization
        assert session.session_id == session_id
        assert session.user_id == user_id
        
        # Record a message and check it was added
        message_id = "msg-1"
        session_tracker.record_message(session_id, message_id, "Hello world", "user", user_id)
        session = session_tracker.get_session(session_id)
        assert len(session.messages) == 1
        assert session.messages[0]["message_id"] == message_id
        
        # Record model selection
        session_tracker.record_model_selection(session_id, "query-1", "gpt-4", {"task1": 0.8, "task2": 0.2})
        session = session_tracker.get_session(session_id)
        assert "query-1" in session.model_selections
        assert session.model_selections["query-1"]["model_name"] == "gpt-4"
        
        # Get engagement metrics
        metrics = session_tracker.get_engagement_metrics(session_id)
        assert "session_length" in metrics
        assert "message_count" in metrics
        assert metrics["message_count"] == 1
        
        # Get implicit feedback
        feedback = session_tracker.get_implicit_feedback_data(session_id)
        assert "session_id" in feedback
        assert feedback["session_id"] == session_id
        assert "engagement_score" in feedback
        
        self.logger.info("Session Tracker test completed successfully")
    
    def test_task_vector_db(self):
        """Test the task vector database."""
        self.logger.info("Testing Task Vector Database")
        
        # Create a smart router instance to get the embeddings client
        router = SmartRouter()
        embeddings_client = getattr(router, "_embeddings_client", None)
        
        # Create task vector DB
        db = TaskVectorDB(embeddings_client)
        
        # Add examples
        db.add_example("Write a Python function to calculate factorial", "coding")
        db.add_example("Solve this equation: 2x^2 + 3x - 5 = 0", "math")
        db.add_example("Explain quantum mechanics and its applications", "science")
        
        # Test task type retrieval
        task_types = db.get_task_types()
        assert "coding" in task_types
        assert "math" in task_types
        assert "science" in task_types
        
        # Test example count
        assert db.get_example_count("coding") == 1
        assert db.get_example_count() == 3
        
        # Test search
        results = db.search("How do I write a Python function?", top_k=2)
        assert len(results) == 2
        assert results[0][0].task_type == "coding"
        
        # Test classification
        classification = db.classify_task("How do I solve a quadratic equation?")
        assert "math" in classification
        assert classification["math"] > 0.5
        
        self.logger.info("Task Vector Database test completed successfully")
    
    def _write_results(self, filename: str, results: Dict):
        """Write test results to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"testLib/logs/{timestamp}_{filename}"
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"Results written to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to write results to {filepath}: {str(e)}")
    
    def _generate_performance_report(self, results: Dict, completion_times=None):
        """Generate a performance report from test results."""
        selection_times = []
        model_usage = {}
        successful_completions = 0
        failed_completions = 0
        
        # Extract data for report
        for case_id, case_data in results.items():
            selection_times.append(case_data["selection_time"])
            
            for model_resp in case_data.get("model_responses", []):
                model_id = model_resp.get("model")
                if model_id:
                    model_usage[model_id] = model_usage.get(model_id, 0) + 1
                
                # Count successful and failed completions
                if "error" in model_resp:
                    failed_completions += 1
                else:
                    successful_completions += 1
        
        # Create report
        self.logger.info("\n\n" + "="*80)
        self.logger.info("SMART ROUTER PERFORMANCE REPORT")
        self.logger.info("="*80)
        
        # Selection performance
        self.logger.info("\nModel Selection Performance:")
        self.logger.info(f"  Average time: {mean(selection_times):.3f} seconds")
        self.logger.info(f"  Median time: {median(selection_times):.3f} seconds")
        if len(selection_times) > 1:
            self.logger.info(f"  Std deviation: {stdev(selection_times):.3f} seconds")
        
        # Completion performance
        if completion_times and len(completion_times) > 0:
            self.logger.info("\nModel Completion Performance:")
            self.logger.info(f"  Average time: {mean(completion_times):.3f} seconds")
            self.logger.info(f"  Median time: {median(completion_times):.3f} seconds")
            if len(completion_times) > 1:
                self.logger.info(f"  Std deviation: {stdev(completion_times):.3f} seconds")
            self.logger.info(f"  Min time: {min(completion_times):.3f} seconds")
            self.logger.info(f"  Max time: {max(completion_times):.3f} seconds")
        
            # Total response time
            avg_total = mean(selection_times) + mean(completion_times)
            self.logger.info("\nTotal Response Time (Selection + Completion):")
            self.logger.info(f"  Average total time: {avg_total:.3f} seconds")
            self.logger.info(f"  Successful completions: {successful_completions}")
            self.logger.info(f"  Failed completions: {failed_completions}")
        
        # Model usage statistics
        if model_usage:
            self.logger.info("\nModel Selection Statistics:")
            for model, count in sorted(model_usage.items(), key=lambda x: x[1], reverse=True):
                self.logger.info(f"  {model}: selected {count} times")
            
        self.logger.info("\n" + "="*80) 