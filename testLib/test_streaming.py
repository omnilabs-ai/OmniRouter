# from .test_core import BaseTest
# from serverRouter.core.datamodels import ChatMessage, ChatCompletionRequest
# import json
# import time
# class TestStreaming(BaseTest):
    
#     def get_providers(self):
#         self.logger.info("Testing List Providers with Sample Models")
#         response = self.client.get("/v1/models/chat")
#         assert response.status_code == 200

#         models = response.json()["models"]
#         assert len(models) > 0, "No models found"

#         provider_models = {}
#         for model in models:
#             provider = model["provider"]
#             if provider not in provider_models:
#                 provider_models[provider] = model["id"]

#         assert len(provider_models) > 0, "No providers found"
#         self.logger.info(f"Found {len(provider_models)} unique providers")
#         return provider_models

#     def _test_streaming(self, model_id: str):
#         self.logger.info(f"Testing streaming with model: {model_id}")
#         request = ChatCompletionRequest(
#             model=model_id,
#             messages=[
#                 ChatMessage(role="user", content="Write me a 3 sentence story about a cat")
#             ],
#             max_tokens=100
#         )

#         # Stream response using `stream=True`
#         with self.client.stream(
#             "POST",
#             "/v1/chat/completions/stream",
#             json=request.model_dump()
#         ) as response:
#             assert response.status_code == 200

#             content = ""
#             chunks = []
#             start_time = time.time()
#             for line in response.iter_lines():
#                 if line:
#                     decoded = line.decode() if isinstance(line, bytes) else line
#                     self.logger.info(f"Received line [{time.time() - start_time:.2f}]: {decoded}")
#                     content += decoded
#                     chunks.append(decoded)
#             end_time = time.time()
#             total_time = end_time - start_time
#             self.logger.info(f"Total streaming time: {total_time:.2f} seconds")
#             assert content.strip() != "", "No content received in stream"
#             assert len(chunks) > 1, "No chunks received in stream"
    
#     def test_streaming(self):
#         self._test_streaming("gemini-2.0-flash-lite")
