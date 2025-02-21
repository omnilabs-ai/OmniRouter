from fastapi.testclient import TestClient
from serverRouter.router import app
from .test_utils import test_logger

class BaseTest:
    def setup_method(self):
        self.client = TestClient(app)
        self.logger = test_logger
        self.client.headers = {
            "Authorization": "Bearer test-sk1o83e"
        }

class TestBasicEndpoints(BaseTest):
    def test_root(self):
        self.logger.info("Test Root")
        response = self.client.get("/")
        self.logger.debug(f"Test Root: {response.json()}")
        assert response.status_code == 200
        assert response.json() == {"message": "Welcome to OmniLLM!"}
        self.logger.info("Root endpoint test completed successfully")

class TestModelEndpoints(BaseTest):
    def test_list_models(self):
        self.logger.info("Test List Models")
        response = self.client.get("/v1/models")
        self.logger.debug(f"Test List Models: {response.json()}")
        assert response.status_code == 200
        assert "models" in response.json()
        self.logger.info("List models endpoint test completed successfully")
