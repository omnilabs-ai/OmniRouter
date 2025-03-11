from .test_core import BaseTest

class TestSmartRouter(BaseTest):

    test_cases = [
        # {
        #     "messages": [{"role": "user", "content": "Write a Python function to calculate factorial"}],
        #     "expected_task": "coding"
        # },
        {
            "messages": [{"role": "user", "content": "Solve this equation: 2x^2 + 3x - 5 = 0"}],
            "expected_task": "math"
        },
        # {
        #     "messages": [{"role": "user", "content": "Explain quantum mechanics and its applications"}],
        #     "expected_task": "science"
        # },
        # {
        #     "messages": [{"role": "user", "content": "Analyze the logical fallacies in this argument"}],
        #     "expected_task": "reasoning"
        # },
        # {
        #     "messages": [{"role": "user", "content": "What is the capital of France?"}],
        #     "expected_task": "general_knowledge"
        # },
        # {
        #     "messages": [{"role": "user", "content": "Write a short story about a detective"}],
        #     "expected_task": "creative_writing"
        # }
    ]

    def test_smart_router(self):
        self.logger.info("Testing Smart Router")

        for test_case in self.test_cases:
            self.logger.info(f"Testing test case: {test_case}")
            response = self.client.post("/v1/router/smart_select", 
                json={"messages": test_case["messages"],
                    "k": 3,
                    "rel_cost": 0,
                    "rel_latency": 1,
                    "rel_accuracy": 1
                })
            assert response.status_code == 200
            self.logger.info(f"Response: {response.json()}")