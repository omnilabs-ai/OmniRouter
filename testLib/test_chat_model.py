from .test_core import BaseTest

class TestChatModels(BaseTest):
    def test_list_chat_models(self):
        self.logger.info("Testing List Chat Models")
        response = self.client.get("/v1/models/chat")
        assert response.status_code == 200
        
        models = response.json()["models"]
        assert len(models) > 0, "No chat models found"
        
        errors = []
        for model in models:
            try:
                self._verify_model_fields(model)
                self._test_chat_completion(model)
            except AssertionError as e:
                self.logger.error(f"Test failed for model {model.get('id', 'unknown')}: {str(e)}")
                errors.append(str(e))
            except Exception as e:
                self.logger.error(f"Unexpected error testing model {model.get('id', 'unknown')}: {str(e)}")
                errors.append(f"Unexpected error: {str(e)}")

        if not errors:
            self.logger.info("All chat models tested successfully")
        else:
            self.logger.error(f"Tests failed for {len(errors)} models:\n" + "\n".join(errors))
            assert False, f"Chat model tests failed:\n" + "\n".join(errors)
    
    def _verify_model_fields(self, model):
        """Helper method to verify all required fields in a model"""
        required_fields = [
            "id", "provider", "description", "max_tokens",
            "benchmarks", "tokenCost", "latency"
        ]
        for field in required_fields:
            assert field in model, f"Model missing '{field}' field: {model}"
    
    def _test_chat_completion(self, model):
        """Helper method to test chat completion for a single model"""
        model_id = model["id"]
        self.logger.info(f"Testing Chat Completion for model: {model_id}")
        
        request_data = {
            "model": model_id,
            "messages": [{"role": "user", "content": "Say 'Hello, test!' in a friendly way."}],
            "temperature": 0.7,
            "max_tokens": 50
        }
        
        self.logger.debug(f"Sending chat completion request for {model_id}: {request_data}")
        
        chat_response = self.client.post(
            "/v1/chat/completions",
            json=request_data,
            timeout=15
        )
        # chat_response.raise_for_status()
        
        self.logger.debug(f"Chat completion response for {model_id} (status {chat_response.status_code}): {chat_response.text}")
        
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
        self.logger.debug(f"Validated chat response for {model_id}: {response_data}") 