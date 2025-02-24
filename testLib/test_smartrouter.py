from .test_core import BaseTest

class TestSmartRouter(BaseTest):

    def test_prompt_based_routing(self):
        # Test with a math-heavy prompt
        math_messages = [
            {
                "role": "user",
                "content": """Solve the following calculus problem:
                Find the derivative of f(x) = 3x^4 + 2x^3 - 5x^2 + 7x - 9
                Show your step-by-step work and explain the power rule."""
            }
        ]
        
        # Test with a creative/poetic prompt
        poetry_messages = [
            {
                "role": "user",
                "content": """Write a beautiful poem about the sunset over the ocean,
                using vivid imagery and metaphors to capture the colors
                and emotions of the scene."""
            }
        ]
        
        # Get routing decisions for both prompts using the API endpoint
        math_response = self.client.post(
            "/v1/router/select-model",
            json={
                "messages": math_messages,
                "k": 3,
                "rel_accuracy": 0.8,
                "rel_cost": 0.2,
                "rel_latency": 0.0,
                "verbose": True
            }
        )
        
        # Test with minimal parameters (all optional params omitted)
        poetry_response = self.client.post(
            "/v1/router/select-model",
            json={
                "messages": poetry_messages
            }
        )
        
        # Assert successful responses
        assert math_response.status_code == 200, f"Math routing failed: {math_response.text}"
        assert poetry_response.status_code == 200, f"Poetry routing failed: {poetry_response.text}"
        
        # Extract results
        math_result = math_response.json()
        poetry_result = poetry_response.json()
        
        # Extract chosen models
        math_model = math_result["model"]
        poetry_model = poetry_result["model"]

        # Log the responses
        self.logger.info(f"Math response: {math_result}")
        self.logger.info(f"Poetry response: {poetry_result}")
        
        # Log the detailed decision process
        # self.logger.info(f"Math prompt routing explanation:\n{math_result['explanation']}")
        # self.logger.info(f"Poetry prompt routing explanation:\n{poetry_result['explanation']}")
        
        # Assert we got different models for different tasks
        # assert math_model != poetry_model, \
        #     f"Expected different models for math vs poetry, but got {math_model} for both"
        
        # Log the final choices
        self.logger.info(f"Math prompt routed to: {math_model}")
        self.logger.info(f"Poetry prompt routed to: {poetry_model}")
