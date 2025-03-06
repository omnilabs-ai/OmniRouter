"""
Comprehensive test of all model providers in the system.
This test file identifies all available providers from the models
and systematically tests each one.
"""

from .test_core import BaseTest
from .test_utils import test_logger

from serverRouter.core.datamodels import ModelProvider
import pytest
import time
import json

class TestAllProviders(BaseTest):
    """Test every provider available in the system."""
    
    def test_all_providers(self):
        """
        Test every provider by fetching all models, grouping by provider,
        and testing one model from each provider.
        """
        # Get all models
        response = self.client.get("/v1/models/chat")
        assert response.status_code == 200
        
        models_data = response.json()["models"]
        test_logger.info(f"Found {len(models_data)} total models")
        
        # Group models by provider
        provider_models = {}
        for model in models_data:
            provider = model["provider"]
            if provider not in provider_models:
                provider_models[provider] = []
            provider_models[provider].append(model["id"])
        
        test_logger.info(f"Found {len(provider_models)} unique providers: {list(provider_models.keys())}")
        
        # Test one model from each provider
        results = {}
        for provider, models in provider_models.items():
            test_logger.info(f"Testing provider: {provider}")
            
            # Choose the first model for this provider
            model_id = models[0]
            test_logger.info(f"Testing model: {model_id}")
            
            # Test the model
            result = self._test_provider_model(provider, model_id)
            results[provider] = result
            
            # Add a short delay between requests
            time.sleep(1)
        
        # Print a summary table
        print("\n===== Provider Test Results =====")
        print(f"{'Provider':<15} | {'Model':<25} | {'Status':<10} | {'Response Preview':<50}")
        print("-" * 105)
        
        for provider, result in results.items():
            model = result.get("model", "N/A")
            status = "Success" if result.get("success") else "Failed"
            preview = result.get("content", "")[:50] + "..." if result.get("content") else result.get("error", "N/A")
            print(f"{provider:<15} | {model:<25} | {status:<10} | {preview:<50}")
        
        # Ensure at least half of providers succeeded
        success_count = sum(1 for r in results.values() if r.get("success"))
        min_success = max(1, len(provider_models) // 2)  # At least 1 or half of providers
        
        assert success_count >= min_success, f"Only {success_count}/{len(provider_models)} providers succeeded, expected at least {min_success}"
    
    def _test_provider_model(self, provider: str, model_id: str) -> dict:
        """
        Test a specific model from a provider.
        
        Args:
            provider: Provider name
            model_id: Model ID to test
            
        Returns:
            Dictionary with test results
        """
        result = {
            "provider": provider,
            "model": model_id,
            "success": False
        }
        
        try:
            request_data = {
                "model": model_id,
                "messages": [
                    {
                        "role": "user", 
                        "content": "Say 'Hello, I am the test for " + provider + "'"
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 50
            }
            
            response = self.client.post(
                "/v1/chat/completions",
                json=request_data,
                timeout=30  # Allow longer timeout for slow providers
            )
            
            if response.status_code != 200:
                result["error"] = f"HTTP {response.status_code}: {response.text}"
                test_logger.error(f"Error for {provider}/{model_id}: {result['error']}")
                return result
            
            data = response.json()
            
            if "content" not in data or not data["content"]:
                result["error"] = "Empty response content"
                test_logger.error(f"Empty response for {provider}/{model_id}")
                return result
            
            result["content"] = data["content"]
            result["success"] = True
            test_logger.info(f"Success for {provider}/{model_id}")
            test_logger.info(f"Response: {data['content'][:100]}...")
            
            return result
            
        except Exception as e:
            result["error"] = str(e)
            test_logger.error(f"Exception testing {provider}/{model_id}: {str(e)}")
            return result
    
    def test_provider_capabilities(self):
        """Test various capabilities for each provider."""
        # Get all models
        response = self.client.get("/v1/models/chat")
        assert response.status_code == 200
        
        models_data = response.json()["models"]
        
        # Group models by provider
        provider_models = {}
        for model in models_data:
            provider = model["provider"]
            if provider not in provider_models:
                provider_models[provider] = []
            provider_models[provider].append(model["id"])
        
        # Define capabilities to test
        capabilities = [
            {
                "name": "Basic Response",
                "prompt": "Say hello in a friendly way",
                "validation": lambda text: "hello" in text.lower()
            },
            {
                "name": "Coding",
                "prompt": "Write a simple function that reverses a string in Python",
                "validation": lambda text: "def" in text and "return" in text and ("[::-1]" in text or "reversed" in text)
            },
            {
                "name": "Math",
                "prompt": "What is the square root of 144?",
                "validation": lambda text: "12" in text
            }
        ]
        
        results = {}
        
        # Test one model from each provider for each capability
        for provider, models in provider_models.items():
            model_id = models[0]  # Use the first model for each provider
            provider_results = {
                "model": model_id,
                "capabilities": {}
            }
            
            for capability in capabilities:
                test_logger.info(f"Testing {provider}/{model_id} for capability: {capability['name']}")
                
                try:
                    request_data = {
                        "model": model_id,
                        "messages": [
                            {
                                "role": "user", 
                                "content": capability['prompt']
                            }
                        ],
                        "temperature": 0.7,
                        "max_tokens": 500
                    }
                    
                    response = self.client.post(
                        "/v1/chat/completions",
                        json=request_data,
                        timeout=30
                    )
                    
                    if response.status_code != 200:
                        provider_results["capabilities"][capability["name"]] = {
                            "success": False,
                            "error": f"HTTP {response.status_code}"
                        }
                        continue
                    
                    data = response.json()
                    content = data.get("content", "")
                    
                    # Validate the response
                    success = capability["validation"](content)
                    
                    provider_results["capabilities"][capability["name"]] = {
                        "success": success,
                        "content_preview": content[:50] + "..." if len(content) > 50 else content
                    }
                    
                except Exception as e:
                    provider_results["capabilities"][capability["name"]] = {
                        "success": False,
                        "error": str(e)
                    }
                
                # Add a short delay between requests
                time.sleep(1)
            
            results[provider] = provider_results
        
        # Print capability results
        print("\n===== Provider Capability Test Results =====")
        for provider, result in results.items():
            print(f"\nProvider: {provider} (Model: {result['model']})")
            print("-" * 80)
            
            capability_results = result["capabilities"]
            for capability, cap_result in capability_results.items():
                status = "✓ Success" if cap_result.get("success") else "✗ Failed"
                preview = cap_result.get("content_preview", cap_result.get("error", "N/A"))
                print(f"  {capability}: {status}")
                print(f"    {preview}")
        
        # Ensure overall success
        overall_success = sum(
            1 for provider in results.values() 
            for cap in provider["capabilities"].values() 
            if cap.get("success")
        )
        
        min_success = len(provider_models)  # At least one successful capability per provider
        
        assert overall_success >= min_success, f"Only {overall_success} capabilities succeeded across all providers, expected at least {min_success}"