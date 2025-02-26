from .test_core import BaseTest
from typing import Optional

# python -m testLib.test_user meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo

class TestSingleModel(BaseTest):
    def test_specific_model(self, model_name: Optional[str] = None):
        """
        Test a specific model by name.
        Args:
            model_name: The ID of the model to test (e.g., 'gpt-4', 'claude-3-5-sonnet', etc.)
                       If None, will print available models and exit.
        """
        # Get available models
        self.logger.info("Fetching available models...")
        response = self.client.get("/v1/models")
        assert response.status_code == 200
        models = response.json()["models"]
        
        # If no model specified, print available models and return
        if not model_name:
            self.logger.info("Not a valid model name, available models:")
            for model in models:
                self.logger.info(f"\n- {model['id']} ({model['provider']})")
            return
        
        # Find specified model in the models list
        target_model = next((model for model in models if model["id"] == model_name), None)
        if not target_model:
            available_models = "\n".join([f"- {m['id']}" for m in models])
            raise ValueError(
                f"Model '{model_name}' not found. Available models:\n{available_models}"
            )
        
        self.logger.info(f"Testing model: {model_name}")
        self.logger.info(f"Provider: {target_model['provider']}")
        self.logger.info(f"Description: {target_model['description']}")
        
        # Verify model information
        assert "provider" in target_model
        assert "description" in target_model
        assert "max_tokens" in target_model
        
        # Test chat completion with the model
        test_message = "Say hello in a friendly way"
        chat_request = {
            "model": model_name,
            "messages": [{"role": "user", "content": test_message}],
            "temperature": 0.7,
            "max_tokens": 50
        }
        
        self.logger.info(f"Testing chat completion with message: '{test_message}'")

        try:
            chat_response = self.client.post("/v1/chat/completions", json=chat_request)
            assert chat_response.status_code == 200, f"Chat completion failed for model {model_name}: {chat_response.text}"
            
            response_data = chat_response.json()
            assert "content" in response_data, f"Response missing content field for model {model_name}"
            assert len(response_data["content"]) > 0, f"Response content is empty for model {model_name}"
            
            self.logger.info(f"Model response: {response_data['content']}")
            self.logger.info("Test completed successfully!")
        except Exception as e:
            self.logger.error(f"Test failed: {e}")
            assert False, f"Test failed: {e}"

if __name__ == "__main__":
    # This allows running the test directly with: python -m testLibV2.test_provider
    import sys
    model_to_test = sys.argv[1] if len(sys.argv) > 1 else None
    test = TestSingleModel()
    test.setup_method()
    test.test_specific_model(model_to_test)
