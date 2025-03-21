from .test_core import BaseTest
import time
class TestSmartRouter(BaseTest):

    test_cases = [
        {
            "messages": [{"role": "user", "content": "Write a Python function to calculate factorial"}],
            "expected_task": "coding"
        },
        {
            "messages": [{"role": "user", "content": "Solve this equation: 2x^2 + 3x - 5 = 0"}],
            "expected_task": "math"
        },
        {
            "messages": [{"role": "user", "content": "Analyze the logical fallacies in this argument"}],
            "expected_task": "reasoning"
        },
        {
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
            "expected_task": "general_knowledge"
        }
    ]

    def test_smart_router(self):
        self.logger.info("Testing Smart Router")

        for test_case in self.test_cases:
            self.logger.info(f"Testing test case: {test_case}")
            response = self.client.post("/v1/smartRouter", 
                json={"messages": test_case["messages"],
                      "max_latency": "balanced",
                      "max_cost": "balanced",
                      "model_list": []
                })
            assert response.status_code == 200
            self.logger.info(f"Response: {response.json()}")

    def test_smart_router_stream(self):
        self.logger.info("Testing Smart Router Stream")

        test_case = self.test_cases[0]
        self.logger.info(f"Testing test case: {test_case}")
        
        with self.client.stream(
            "POST",
            "/v1/smartRouterStream",
            json={
                "messages": test_case["messages"],
                "max_latency": "balanced", 
                "max_cost": "balanced",
                "model_list": []
            }
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
            
            # Verify we received all expected event types
            events = [chunk for chunk in chunks if chunk.startswith("event:")]
            assert "event: metadata" in events, "Missing metadata events"
            assert "event: return" in events, "Missing return event"