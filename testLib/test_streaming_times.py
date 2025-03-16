from .test_core import BaseTest
from serverRouter.core.datamodels import ChatMessage, ChatCompletionRequest, SmartRouterRequest
import json
from datetime import datetime



class TestStreaming(BaseTest):
    provider = "openai"
    model_id = "gpt-3.5-turbo"
    
    test_request = ChatCompletionRequest(
        model=model_id,
        messages=[
            ChatMessage(role="user", content="Write me a 3 sentence story about a cat")
        ],
    )

    def streaming_response(self):
        # Make the streaming request
        start_time = datetime.now()
        self.logger.info(f"[+0.000000s] Sending Stream request: {self.test_request}")
        response = self.client.post(
            "/v1/chat/completions/stream",
            json=self.test_request.model_dump()
        )
        # Verify response status
        assert response.status_code == 200
                
        # Read and verify streaming chunks
        content = ""
        chunks = []
        first_chunk_time = None
        
        # Process SSE stream
        for line in response.iter_lines():
            if line:
                line = line.decode() if isinstance(line, bytes) else line
                if line.startswith('data: '):
                    data = line[6:]  # Remove 'data: ' prefix
                    if data == '[DONE]':
                        break
                    try:
                        current_time = datetime.now()
                        if not first_chunk_time:
                            first_chunk_time = current_time
                        chunk_data = json.loads(data)
                        chunk_content = chunk_data.get('content', '')
                        content += chunk_content
                        elapsed = (current_time - start_time).total_seconds()
                        self.logger.info(f"[+{elapsed:.6f}s] Chunk content: {chunk_content}")
                        chunks.append(chunk_content)
                    except json.JSONDecodeError:
                        self.logger.error(f"Failed to parse chunk: {data}")
                        continue

        end_time = datetime.now()
        
        # Calculate timing metrics
        time_to_first_chunk = (first_chunk_time - start_time).total_seconds() if first_chunk_time else 0
        total_time = (end_time - start_time).total_seconds()
        
        # Log timing information
        self.logger.info(f"\nStreaming metrics:")
        self.logger.info(f"Time to first chunk: +{time_to_first_chunk:.6f}s")
        self.logger.info(f"Total time: +{total_time:.6f}s")
        self.logger.info(f"Final content: {content}")
        self.logger.info(f"Number of chunks received: {len(chunks)}")
        
        # Verify we got some content
        assert len(content) > 0
        assert len(chunks) > 1
        return total_time
        
    def baseline_response(self):
        start_time = datetime.now()
        self.logger.info(f"[+0.000000s] Sending static request: {self.test_request}")
        response = self.client.post(
            "/v1/chat/completions",
            json=self.test_request.model_dump()
        )
        end_time = datetime.now()
        assert response.status_code == 200
        
        response_data = response.json()
        total_time = (end_time - start_time).total_seconds()
        
        # Log timing information
        self.logger.info(f"\nStatic response metrics:")
        self.logger.info(f"Total time: +{total_time:.6f}s")
        self.logger.info(f"Response content: {response_data.get('content', '')}")
        
        return total_time
        
    def test_compare_response_times(self):
        """Compare the response times between streaming and static responses"""
        
        streaming_time = self.streaming_response()
        static_time = self.baseline_response()
        
        self.logger.info("\nResponse Time Comparison:")
        self.logger.info(f"Streaming total time: {streaming_time:.6f}s")
        self.logger.info(f"Static total time: {static_time:.6f}s")
        self.logger.info(f"Difference: {static_time - streaming_time:+.6f}s")