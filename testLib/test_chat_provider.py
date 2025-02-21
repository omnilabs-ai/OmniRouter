from .test_core import BaseTest

# Built to test all the ways a provider can be called and make sure they all work

class TestProviders(BaseTest):
    def test_list_providers(self):
        self.logger.info("Testing List Providers with Sample Models")
        response = self.client.get("/v1/models/chat")
        assert response.status_code == 200
        
        models = response.json()["models"]
        assert len(models) > 0, "No models found"
        
        provider_models = {}
        for model in models:
            provider = model["provider"]
            if provider not in provider_models:
                provider_models[provider] = model["id"]
        
        assert len(provider_models) > 0, "No providers found"
        self.logger.info(f"Found {len(provider_models)} unique providers")

        errors = []
        for provider, model_id in provider_models.items():
            self.logger.debug(f"Testing Provider: {provider}, Model: {model_id}")
            try:
                response = self._call_provider_simple(model_id)
                self.logger.debug(f"Response: {response.json()}")
                if response.status_code != 200:
                    errors.append(f"Provider {provider} with model {model_id} failed with status {response.status_code}")
            except Exception as e:
                self.logger.error(f"Error testing provider {provider} with model {model_id}: {str(e)}")
                errors.append(f"Provider {provider} with model {model_id} failed with error: {str(e)}")

        if errors:
            self.logger.error(f"Tests failed for {len(errors)} providers:\n" + "\n".join(errors))
            assert False, f"Provider tests failed:\n" + "\n".join(errors)
        else:
            self.logger.info("All providers tested successfully")

    def _call_provider_simple(self, model_id):
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": model_id, 
                "messages": [{"role": "user", "content": "Say 'Hello, test!' in a friendly way."}]
                },
        )
        return response
        
