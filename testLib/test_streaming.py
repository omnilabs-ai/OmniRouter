from .test_core import BaseTest
from serverRouter.core.datamodels import ChatMessage, ChatCompletionRequest, ChatReasoningRequest, ReasoningEffort
import json
import time
import sys

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

    def get_reasoning_models(self):
        self.logger.info("Getting reasoning models")
        response = self.client.get("/v1/models/reasoning")
        assert response.status_code == 200

        models = response.json()["models"]
        assert len(models) > 0, "No reasoning models found"

        model_by_provider = {}
        for model in models:
            provider = model["provider"]
            model_by_provider[provider] = model["id"]
        
        self.logger.info(f"Found reasoning models for providers: {list(model_by_provider.keys())}")
        return model_by_provider

    def _test_chat_streaming(self, model_id: str):
        self.logger.info(f"Testing chat streaming with model: {model_id}")
        request = ChatCompletionRequest(
            model=model_id,
            messages=[
                ChatMessage(role="user", content="Write me a 3 sentence story about a cat")
            ],
            max_tokens=100
        )

        # Stream response using `stream=True`
        with self.client.stream(
            "POST",
            "/v1/chat/completions/stream",
            json=request.model_dump()
        ) as response:
            assert response.status_code == 200

            content = ""
            chunks = []
            start_time = time.time()
            for line in response.iter_lines():
                if line:
                    decoded = line.decode() if isinstance(line, bytes) else line
                    self.logger.info(f"Received line [{time.time() - start_time:.2f}]: {decoded}")
                    content += decoded
                    chunks.append(decoded)
            end_time = time.time()
            total_time = end_time - start_time
            self.logger.info(f"Total streaming time: {total_time:.2f} seconds")
            assert content.strip() != "", "No content received in stream"
            assert len(chunks) > 1, "No chunks received in stream"
    
    def _test_reasoning_streaming(self, model_id: str):
        """Test streaming for a reasoning model"""
        self.logger.info(f"Testing reasoning streaming with model: {model_id}")
        
        # Create a reasoning request
        request = ChatReasoningRequest(
            model=model_id,
            messages=[
                ChatMessage(role="user", content="What is the sum of the first 5 prime numbers? Explain your reasoning.")
            ],
            reasoning_effort=ReasoningEffort.LOW,  # Use low for faster test
            max_tokens=200,
            temperature=1.0
        )
        
        # Make the streaming request
        response = self.client.post(
            "/v1/reason/completions/stream",
            json=request.model_dump(),
            timeout=60,  # Longer timeout for reasoning
            stream=True  # Enable streaming
        )
        
        # Check status code
        assert response.status_code == 200, f"Expected 200 status but got {response.status_code}: {response.text}"
        
        # Process and analyze the streamed response
        events = {}  # Track events by type
        current_event = None  # Current event being processed
        event_sequence = []  # Sequence of event types
        
        # Process each line of the response
        for line in response.iter_lines():
            if not line:
                continue
                
            line_text = line.decode('utf-8') if isinstance(line, bytes) else line
            line_text = line_text.strip()
            
            # Extract event type and data
            if line_text.startswith('event:'):
                current_event = line_text.replace('event:', '').strip()
                event_sequence.append(current_event)
                if current_event not in events:
                    events[current_event] = []
                self.logger.info(f"EVENT: {current_event}")
                    
            elif line_text.startswith('data:') and current_event:
                data_text = line_text.replace('data:', '').strip()
                
                # Try to parse as JSON for structured analysis
                try:
                    data_json = json.loads(data_text)
                    events[current_event].append(data_json)
                    
                    # Log interesting events with content
                    if current_event in ['content', 'reasoning'] and 'content' in data_json:
                        content_text = data_json['content']
                        self.logger.info(f"  {current_event}: {content_text[:50]}...")
                except Exception:
                    # If not JSON, store as plain text
                    events[current_event].append(data_text)
                    self.logger.info(f"  Raw data: {data_text[:50]}...")
                    
        # Analyze the response for proper streaming behavior
        self.logger.info("Analyzing reasoning stream response:")
        self.logger.info(f"Event types: {list(events.keys())}")
        self.logger.info(f"Event sequence: {event_sequence[:5]}... ({len(event_sequence)} total)")
        
        # Check for important event types
        required_events = ['metadata', 'thinking_start']
        for event in required_events:
            assert event in events, f"Missing required event: {event}"
        
        # Check that we got reasoning or content events
        assert 'reasoning' in events or 'content' in events, "No reasoning or content events received"
        
        # Check block markers if present
        if 'block_start' in events:
            assert 'block_stop' in events, "Found block_start but missing block_stop"
            
        # Verify final usage token stats are provided
        assert 'usage' in events, "Missing usage event with token stats"
            
        self.logger.info(f"Reasoning streaming test for {model_id} completed successfully")
            
    def test_xai_reasoning_streaming(self):
        """Test specifically for XAI reasoning models streaming"""
        reasoning_models = self.get_reasoning_models()
        
        # Find XAI reasoning models
        xai_models = {model_id: provider for provider, model_id in reasoning_models.items() if provider == "xai"}
        
        if not xai_models:
            self.logger.info("No XAI reasoning models found, skipping test")
            return
            
        # Test each XAI reasoning model
        for model_id in xai_models.values():
            self.logger.info(f"Testing XAI reasoning streaming with model: {model_id}")
            self._test_reasoning_streaming(model_id)
    
    def test_chat_streaming(self):
        """Test basic chat completion streaming"""
        self._test_chat_streaming("gemini-2.0-flash-lite")
        
# Direct test runner
if __name__ == "__main__":
    # Print to stdout directly
    print("=== Starting Streaming Tests with Direct Output ===")
    
    # Create and run the test
    test = TestStreaming()
    test.setup_method()
    
    try:
        print("Testing XAI streaming...")
        test.test_xai_reasoning_streaming()
        print("All XAI streaming tests passed!")
    except Exception as e:
        print(f"Test failed: {str(e)}")
        sys.exit(1)
