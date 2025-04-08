from .test_core import BaseTest
from serverRouter.core.datamodels import ImageSize

class TestDeepAIProvider(BaseTest):
    def test_deepai_image_generation(self):
        """Test DeepAI image generation with different models."""
        self.logger.info("Testing DeepAI Image Generation")
        
        # Test models
        models = [
            "deepai-standard",
            "deepai-hd",
            "deepai-genius",
            "deepai-genius-anime",
            "deepai-genius-photography",
            "deepai-genius-graphic"
        ]
        
        # Simple test prompt for all models
        prompt = "A beautiful sunset over a mountain lake with reflections"
        
        for model_id in models:
            self.logger.info(f"Testing DeepAI image generation with model: {model_id}")
            
            # Create image generation request
            request_data = {
                "prompt": prompt,
                "model": model_id,
                "size": ImageSize.MEDIUM.value,
                "quality": "standard",
                "n": 1
            }
            
            try:
                # Send request to API
                response = self.client.post(
                    "/v1/images/generate",
                    json=request_data,
                    timeout=30  # Image generation might take longer
                )
                
                # Check response
                self.logger.info(f"Response status code: {response.status_code}")
                
                if response.status_code == 200:
                    response_data = response.json()
                    assert "urls" in response_data, f"Response missing 'urls' field for model {model_id}"
                    assert len(response_data["urls"]) > 0, f"No image URLs returned for model {model_id}"
                    assert response_data["provider"] == "deepai", f"Provider mismatch for model {model_id}"
                    
                    # Verify URL is a data URL (base64), not an HTTP URL
                    for url in response_data["urls"]:
                        assert url.startswith("data:"), f"Invalid URL format for model {model_id}: {url}"
                    
                    self.logger.info(f"Successfully generated image with model: {model_id}")
                else:
                    self.logger.warning(f"Image generation for model {model_id} failed: {response.text}")
                    assert False, f"Image generation failed for model {model_id}: {response.text}"
            
            except Exception as e:
                self.logger.error(f"Error during image generation test for {model_id}: {str(e)}")
                assert False, f"Test failed for {model_id}: {str(e)}"
        
        self.logger.info("All DeepAI image generation tests completed successfully") 