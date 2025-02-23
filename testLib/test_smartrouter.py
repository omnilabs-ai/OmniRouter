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
                "content": "What is the derivative of x^2 + 3x - 4"
            },
            {
                "role": "assistant",
                "content": "The derivative of x^2 + 3x - 4 is 2x + 3.\n\nHere's how to find it using the power rule and the sum/difference rule of differentiation:\n\n*   **Power Rule:** The derivative of x^n is n * x^(n-1)\n\n*   **Sum/Difference Rule:** The derivative of a sum or difference of terms is the sum or difference of the derivatives of each term.\n\n1.  **Derivative of x^2:** Applying the power rule, the derivative of x^2 is 2 * x^(2-1) = 2x.\n2.  **Derivative of 3x:**  This can be thought of as 3 * x^1. Applying the power rule and the constant multiple rule (the derivative of c*f(x) is c*f'(x)), the derivative is 3 * 1 * x^(1-1) = 3 * x^0 = 3 * 1 = 3.\n3.  **Derivative of -4:** The derivative of a constant is always 0.\n\n4.  **Combine:** Using the sum/difference rule, the derivative of x^2 + 3x - 4 is 2x + 3 - 0 = 2x + 3.\n\nTherefore, the derivative is **2x + 3**.\n"
            },
            {
                "role": "user",
                "content": "What is the derivative of x^2 + 3x - 4\n"
            }
        ]       
        # [
        #     {
        #         "role": "user",
        #         "content": """Write a beautiful poem about the sunset over the ocean,
        #         using vivid imagery and metaphors to capture the colors
        #         and emotions of the scene."""
        #     }
        # ]
        
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
