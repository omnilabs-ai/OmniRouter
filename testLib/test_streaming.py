from .test_core import BaseTest
from serverRouter.core.datamodels import ChatMessage, ChatCompletionRequest, SmartRouterRequest
import json

class TestStreaming(BaseTest):

    def get_providers(self):
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
        return provider_models

    # def test_streaming(self):
    #     # Create a simple chat completion request
    #     provider_models = self.get_providers()

    #     for provider, model_id in provider_models.items():
    #         self.logger.info(f"Testing Provider: {provider}, Model: {model_id}")
    #         request = ChatCompletionRequest(
    #             model=model_id,
    #             messages=[
    #                 ChatMessage(role="user", content="Write me a 3 sentence story about a cat")
    #             ],
    #         )
            
    #         # Make the streaming request
    #         response = self.client.post(
    #             "/v1/chat/completions/stream",
    #             json=request.model_dump()
    #         )
    #         # Verify response status
    #         assert response.status_code == 200
            
    #         # Read and verify streaming chunks
    #         content = ""
    #         chunks = []
            
    #         # Process SSE stream
    #         for line in response.iter_lines():
    #             if line:
    #                 line = line.decode() if isinstance(line, bytes) else line
    #                 if line.startswith('data: '):
    #                     data = line[6:]  # Remove 'data: ' prefix
    #                     if data == '[DONE]':
    #                         break
    #                     try:
    #                         chunk_data = json.loads(data)
    #                         chunk_content = chunk_data.get('content', '')
    #                         content += chunk_content
    #                         chunks.append(chunk_content)
    #                     except json.JSONDecodeError:
    #                         self.logger.error(f"Failed to parse chunk: {data}")
    #                         continue

    #         # Verify we got some content
    #         self.logger.info(f"Final content: {content}")
    #         self.logger.info(f"Number of chunks received: {len(chunks)}")
    #         assert len(content) > 0
    #         assert len(chunks) > 1
        
    def test_smart_streaming(self):
        # Create a simple chat completion request
        self.logger.info(f"Testing Smart Router")
        request = SmartRouterRequest(
            messages=[
                ChatMessage(role="user", content="Write me a 3 sentence story about a cat")
            ]
        )
        
        # Make the streaming request
        response = self.client.post(
            "/v1/router/smart_select/stream",
            json=request.model_dump()
        )
        # Verify response status
        assert response.status_code == 200
        
        # Read and verify streaming chunks
        content = ""
        chunks = []
        
        # Process SSE stream
        for line in response.iter_lines():
            if line:
                line = line.decode() if isinstance(line, bytes) else line
                if line.startswith('data: '):
                    data = line[6:]  # Remove 'data: ' prefix
                    if data == '[DONE]':
                        break
                    try:
                        chunk_data = json.loads(data)
                        chunk_content = chunk_data.get('content', '')
                        content += chunk_content
                        chunks.append(chunk_content)
                    except json.JSONDecodeError:
                        self.logger.error(f"Failed to parse chunk: {data}")
                        continue

        # Verify we got some content
        self.logger.info(f"Final content: {content}")
        self.logger.info(f"Number of chunks received: {len(chunks)}")
        assert len(content) > 0
        assert len(chunks) > 1


