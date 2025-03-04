from .test_core import BaseTest
from typing import Optional
import os
import base64
from datetime import datetime
from pathlib import Path

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
        
        # Check if this is an image model
        is_image_model = False
        image_response = self.client.get("/v1/models/image")
        if image_response.status_code == 200:
            image_models = [m["id"] for m in image_response.json()["models"]]
            is_image_model = model_name in image_models
        
        try:
            if is_image_model:
                self._test_image_model(model_name)
            else:
                self._test_chat_model(model_name)
            self.logger.info("Test completed successfully!")
        except Exception as e:
            self.logger.error(f"Test failed: {e}")
            assert False, f"Test failed: {e}"

    def _test_chat_model(self, model_name: str):
        """Test a chat completion model"""
        test_message = "Say hello in a friendly way"
        self.logger.info(f"Testing chat completion with message: '{test_message}'")
        
        chat_request = {
            "model": model_name,
            "messages": [{"role": "user", "content": test_message}],
            "temperature": 0.7,
            "max_tokens": 50
        }
        
        chat_response = self.client.post("/v1/chat/completions", json=chat_request)
        assert chat_response.status_code == 200, f"Chat completion failed for model {model_name}: {chat_response.text}"
        
        response_data = chat_response.json()
        assert "content" in response_data, f"Response missing content field for model {model_name}"
        assert len(response_data["content"]) > 0, f"Response content is empty for model {model_name}"
        
        self.logger.info(f"Model response: {response_data['content']}")

    def _test_image_model(self, model_name: str):
        """Test an image generation model"""
        from serverRouter.core.datamodels import ImageSize
        
        test_prompt = "A serene landscape with mountains and a lake at sunset"
        self.logger.info(f"Testing image generation with prompt: '{test_prompt}'")
        
        image_request = {
            "model": model_name,
            "prompt": test_prompt,
            "size": ImageSize.LARGE.value,
            "quality": "standard",
            "n": 1
        }
        
        image_response = self.client.post("/v1/images/generate", json=image_request, timeout=30)
        assert image_response.status_code == 200, f"Image generation failed for model {model_name}: {image_response.text}"
        
        response_data = image_response.json()
        assert "urls" in response_data, f"Response missing urls field for model {model_name}"
        assert len(response_data["urls"]) > 0, f"No image URLs in response for model {model_name}"
        assert "provider" in response_data, f"Response missing provider field for model {model_name}"
        
        # Create logs directory if it doesn't exist
        logs_dir = Path("testLib\\logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Save images
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_paths = []
        
        for idx, url in enumerate(response_data["urls"]):
            assert not url.startswith("http"), f"HTTP URL found instead of base64 data for model {model_name}"
            assert url.startswith("data:"), f"Invalid URL format - expected data URL for model {model_name}"
            
            # Extract the image data and format
            header, encoded = url.split(",", 1)
            image_format = header.split(";")[0].split("/")[1]
            
            # Decode and save the image
            image_data = base64.b64decode(encoded)
            image_path = logs_dir / f"{model_name}_{timestamp}_{idx}.{image_format}"
            with open(image_path, "wb") as f:
                f.write(image_data)
            saved_paths.append(image_path)
            self.logger.info(f"Saved image to: {image_path}")
        
        self.logger.info(f"Generated and saved {len(saved_paths)} image(s) to logs directory")

if __name__ == "__main__":
    # This allows running the test directly with: python -m testLibV2.test_provider
    import sys
    model_to_test = sys.argv[1] if len(sys.argv) > 1 else None
    test = TestSingleModel()
    test.setup_method()
    test.test_specific_model(model_to_test)
