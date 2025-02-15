import os
import requests
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv

class APIClient:
    def __init__(self, api_key: str = None):

        load_dotenv()
        
        if api_key is None:
            # Try to get from environment variable
            api_key = os.getenv('OMNI_API_KEY')

        if not api_key:
            raise ValueError(
                "No API key provided. Pass it when initializing the client or "
                "set the OMNI_API_KEY environment variable."
            )
            
        self._api_key = api_key
        self._base_url =  "http://localhost:8000"
        

    def _make_request(self, endpoint: str, method: str = 'GET', **kwargs):
        headers = {
            'Authorization': f'Bearer {self._api_key}',
            'Content-Type': 'application/json'
        }
        
        response = requests.request(
            method=method,
            url=f"{self._base_url}/{endpoint.lstrip('/')}",
            headers=headers,
            **kwargs
        )
        
        response.raise_for_status()
        return response.json()
        
    def chat(self, messages, model: str, temperature: float = 0.7, max_tokens: int = 100) -> dict:
        """
        Send a chat completion request.
        
        Args:
            messages (list): List of message dictionaries with 'role' and 'content'
            model (str): The model ID to use
            temperature (float, optional): Sampling temperature. Defaults to 0.7
            max_tokens (int, optional): Maximum tokens to generate. Defaults to 100
            
        Returns:
            dict: The API response containing the chat completion
        """
        request_data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        return self._make_request(
            endpoint="/v1/chat/completions",
            method="POST",
            json=request_data
        )

    def generate_image(self, prompt: str, model: str, n: int = 1, size: str = "1024x1024", google_cloud_project_id: Optional[str] = None, google_cloud_location: Optional[str] = None) -> dict:
        """
        Generate images from a text prompt.
        
        Args:
            prompt (str): The image generation prompt
            model (str): The model ID to use
            n (int, optional): Number of images to generate. Defaults to 1
            size (str, optional): Image size. Defaults to "1024x1024"
            
        Returns:
            dict: The API response containing image URLs
        """
        request_data = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "size": size,
            "google_cloud_project_id": google_cloud_project_id, #Added
            "google_cloud_location": google_cloud_location #Added
        }
        
        return self._make_request(
            endpoint="/v1/images/generate",
            method="POST",
            json=request_data
        )

    def get_available_models(self, model_type: str = None) -> List[Dict[str, Any]]:
        """
        Get list of available models from the API
        
        Args:
            model_type (str, optional): Type of models to get ('chat' or 'image'). 
                                      If None, returns all models.
        
        Returns:
            List[Dict[str, Any]]: List of available models and their information
        """
        if model_type == 'chat':
            return self._make_request("v1/models/chat")["models"]
        elif model_type == 'image':
            return self._make_request("v1/models/image")["models"]
        else:
            # Get all models if no specific type is requested
            chat_models = self._make_request("v1/models/chat")["models"]
            image_models = self._make_request("v1/models/image")["models"]
            return chat_models + image_models

    def select_model(
        self,
        query: str,
        k: int = 5,
        model_names: Optional[List[str]] = None,
        rel_cost: float = 0.5,
        rel_latency: float = 0.0,
        rel_accuracy: float = 0.5,
        verbose: bool = False
    ) -> Union[str, Dict[str, str]]:
        """
        Get model recommendation based on query and preferences.
        
        Args:
            query (str): The query text to analyze for model selection
            k (int, optional): Number of top models to consider. Defaults to 5.
            model_names (List[str], optional): Specific models to select from. Defaults to None.
            rel_cost (float, optional): Relative importance of cost (0-1). Defaults to 0.5.
            rel_latency (float, optional): Relative importance of latency (0-1). Defaults to 0.0.
            rel_accuracy (float, optional): Relative importance of accuracy (0-1). Defaults to 0.5.
            verbose (bool, optional): Whether to return detailed explanation. Defaults to False.
            
        Returns:
            If verbose=False: str - Name of the recommended model
            If verbose=True: dict - Contains model name and detailed explanation
        """
        request_data = {
            "query": query,
            "k": k,
            "model_names": model_names,
            "rel_cost": rel_cost,
            "rel_latency": rel_latency,
            "rel_accuracy": rel_accuracy,
            "verbose": verbose
        }
        
        return self._make_request(
            endpoint="/v1/router/select-model",
            method="POST",
            json=request_data
        )

# Usage example:
# client = APIClient(api_key='your-api-key-here')
# Or using environment variable:
# client = APIClient()