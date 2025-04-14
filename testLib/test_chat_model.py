from .test_core import BaseTest
import sys

class TestChatModels(BaseTest):
    def test_list_chat_models(self):
        self.logger.info("Testing List Chat Models")
        response = self.client.get("/v1/models/chat")
        assert response.status_code == 200, f"Failed to get models: Status {response.status_code}"
        
        models = response.json()["models"]
        assert len(models) > 0, "No chat models found"
        
        print(f"Found {len(models)} chat models")
        for i, model in enumerate(models[:3]):  # Show the first 3 models
            print(f"Model {i+1}: {model['id']} ({model['provider']})")
        
        errors = []
        for model in models:
            try:
                self._verify_model_fields(model)
                self._test_chat_completion(model)
            except AssertionError as e:
                self.logger.error(f"Test failed for model {model.get('id', 'unknown')}: {str(e)}")
                errors.append(str(e))
                print(f"ERROR: Test failed for model {model.get('id', 'unknown')}: {str(e)}")
            except Exception as e:
                self.logger.error(f"Unexpected error testing model {model.get('id', 'unknown')}: {str(e)}")
                errors.append(f"Unexpected error: {str(e)}")
                print(f"ERROR: Unexpected error testing model {model.get('id', 'unknown')}: {str(e)}")

        if not errors:
            self.logger.info("All chat models tested successfully")
            print("SUCCESS: All chat models tested successfully")
        else:
            self.logger.error(f"Tests failed for {len(errors)} models:\n" + "\n".join(errors))
            print(f"FAILED: Tests failed for {len(errors)} models:\n" + "\n".join(errors))
            assert False, f"Chat model tests failed:\n" + "\n".join(errors)
    
    def _verify_model_fields(self, model):
        """Helper method to verify all required fields in a model"""
        required_fields = [
            "id", "provider", "description", "max_tokens",
            "benchmarks", "tokenCost", "latency"
        ]
        for field in required_fields:
            assert field in model, f"Model missing '{field}' field: {model}"
        print(f"✓ Verified model fields for {model['id']}")
    
    def _test_chat_completion(self, model):
        """Helper method to test chat completion for a single model"""
        model_id = model["id"]
        self.logger.info(f"Testing Chat Completion for model: {model_id}")
        print(f"Testing Chat Completion for model: {model_id}")
        
        request_data = {
            "model": model_id,
            "messages": [{"role": "user", "content": "Say 'Hello, test!' in a friendly way."}],
            "temperature": 0.7,
            "max_tokens": 50
        }
        
        self.logger.debug(f"Sending chat completion request for {model_id}: {request_data}")
        print(f"Sending request to: {model_id}")
        
        chat_response = self.client.post(
            "/v1/chat/completions",
            json=request_data,
            timeout=15
        )
        # chat_response.raise_for_status()
        
        print(f"Response status: {chat_response.status_code}")
        
        self.logger.debug(f"Chat completion response for {model_id} (status {chat_response.status_code}): {chat_response.text}")
        
        if chat_response.status_code == 429:
            print(f"Rate limit exceeded for {model_id}, skipping verification")
            self.logger.warn(f"Rate limit exceeded for {model_id}, skipping verification")
            return
            
        response_data = chat_response.json()
        self._verify_chat_response(response_data, model)
    
    def _verify_chat_response(self, response_data, model):
        """Helper method to verify chat completion response"""
        model_id = model["id"]
        assert "content" in response_data, f"Response missing 'content' field for model {model_id}"
        assert "provider" in response_data, f"Response missing 'provider' field for model {model_id}"
        assert "usage" in response_data, f"Response missing 'usage' field for model {model_id}"
        assert len(response_data["content"]) > 0, f"Empty response content for model {model_id}"
        assert response_data["provider"] == model["provider"], f"Provider mismatch for model {model_id}"
        
        print(f"✓ Chat completion successful for {model_id}")
        print(f"   Content (first 50 chars): {response_data['content'][:50]}...")
        print(f"   Usage: {response_data.get('usage', {})}")
        
        self.logger.debug(f"Validated chat response for {model_id}: {response_data}")

# Direct test runner if run as a script
if __name__ == "__main__":
    # Print to stdout directly
    print("=== Starting Chat Model Test with Direct Output ===")
    
    # Create and run the test
    test = TestChatModels()
    test.setup_method()
    
    try:
        print("Testing chat models with output to console...")
        test.test_list_chat_models()
        print("All tests completed!")
    except Exception as e:
        print(f"Test failed: {str(e)}")
        sys.exit(1) 