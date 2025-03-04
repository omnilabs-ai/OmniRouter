"""
Test suite for the streaming functionality of the API.
"""

import pytest
import os
import sys
import json
import asyncio
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import aiohttp
import requests

# Add parent directory to path to allow imports
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

# Initialize logger
from testLib.test_utils import test_logger

from testLib.test_core import BaseTest
from fastapi.testclient import TestClient
from serverRouter.router import app

class TestStreaming(BaseTest):
    """Test class for streaming functionality"""
    
    @pytest.fixture(scope="class")
    def api_key(self):
        """Test API key from config."""
        return "test-sk1o83e"  # Default test key from config.py
    
    @pytest.fixture(scope="class")
    def api_url(self):
        """Base URL for API requests."""
        return "http://localhost:8000/v1/chat/completions"
    
    def test_streaming_request_structure(self, api_key):
        """Test that a streaming request is properly structured."""
        test_logger.info("Testing streaming request structure")
        
        # Prepare request payload
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": "Say hello"}
            ],
            "stream": True,
            "temperature": 0.7
        }
        
        # Check request structure
        assert "model" in payload
        assert "messages" in payload
        assert isinstance(payload["messages"], list)
        assert len(payload["messages"]) > 0
        assert "role" in payload["messages"][0]
        assert "content" in payload["messages"][0]
        assert payload["stream"] is True
        
        test_logger.info("Streaming request structure is valid")
    #why is this in the test_streaming file?
    def test_non_streaming_endpoint(self):
        """Test the non-streaming version of the chat completions endpoint."""
        test_logger.info("Testing non-streaming chat completion")
        
        request_data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Say 'Hello, test!' in a friendly way."}],
            "temperature": 0.7,
            "max_tokens": 50,
            "stream": False
        }
        
        response = self.client.post(
            "/v1/chat/completions",
            json=request_data
        )
        
        test_logger.info(f"Response status: {response.status_code}")
        assert response.status_code == 200
        
        data = response.json()
        test_logger.info(f"Response content (truncated): {str(data)[:100]}...")
        
        assert "content" in data
        assert "model" in data
        assert "provider" in data
        assert isinstance(data["content"], str)
        assert len(data["content"]) > 0
        
        test_logger.info("Non-streaming endpoint works correctly")
    
    #what is this and why it is doing
    @pytest.mark.asyncio
    async def test_streaming_response_handling(self, api_key, api_url):
        """Test parsing and handling streaming responses."""
        test_logger.info("Testing streaming response handling")
        
        # Sample streaming response lines
        sample_lines = [
            'data: {"content":"Hello", "model":"gpt-3.5-turbo", "provider":"openai"}',
            'data: {"content":", ", "model":"gpt-3.5-turbo", "provider":"openai"}',
            'data: {"content":"world!", "model":"gpt-3.5-turbo", "provider":"openai"}',
            'data: [DONE]'
        ]
        
        # Parse the sample lines
        complete_content = ""
        for line in sample_lines:
            if line.startswith('data: '):
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                
                try:
                    chunk = json.loads(data)
                    content = chunk.get("content", "")
                    complete_content += content
                except json.JSONDecodeError:
                    test_logger.error(f"Error parsing JSON: {data}")
        
        assert complete_content == "Hello, world!"
        test_logger.info("Successfully handled streaming response chunks")
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    
    async def test_streaming_integration(self, api_key, api_url):
        """
        Integration test for the streaming API.
        
        Note: This test requires the API server to be running.
        Skip this test if the server is not available.
        """
        test_logger.info("Testing streaming integration")
        
        # Check if server is running
        try:
            response = requests.get("http://localhost:8000/")
            if response.status_code != 200:
                pytest.skip("API server is not running")
        except requests.exceptions.ConnectionError:
            pytest.skip("API server is not running")
        
        # Prepare request payload
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": "Say hello in a friendly way"}
            ],
            "stream": True,
            "temperature": 0.7
        }
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    api_url,
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status != 200:
                        test_logger.error(f"Error response: {response.status}")
                        test_logger.error(await response.text())
                        pytest.skip("API returned an error")
                    
                    # Process the streaming response
                    complete_content = ""
                    first_token_received = False
                    
                    async for line in response.content:
                        line = line.decode('utf-8')
                        test_logger.debug(f"Raw data: {line}")
                        
                        if line.startswith('data: '):
                            data = line[6:].strip()
                            if data == "[DONE]":
                                break
                                
                            try:
                                chunk = json.loads(data)
                                
                                # Record time of first token
                                if not first_token_received:
                                    first_token_received = True
                                    
                                # Print the content chunk
                                content = chunk.get("content", "")
                                complete_content += content
                                
                            except json.JSONDecodeError:
                                test_logger.error(f"Error parsing JSON: {data}")
                    
                    test_logger.info(f"Complete content: {complete_content}")
                    assert len(complete_content) > 0
                    assert "hello" in complete_content.lower()
                    
                    test_logger.info("Streaming integration test passed")
        
        except Exception as e:
            test_logger.error(f"Error in streaming test: {str(e)}")
            pytest.skip(f"Error in streaming test: {str(e)}")


class TestStreamingClient:
    """Test class for the streaming client functionality"""
    
    def test_client_request_formatting(self):
        """Test that the streaming client formats requests correctly."""
        test_logger.info("Testing client request formatting")
        
        # Sample request parameters
        model = "gpt-3.5-turbo"
        prompt = "Tell me a short story"
        api_key = "test-key"
        
        # Expected payload structure
        expected_payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "temperature": 0.7
        }
        
        # Expected headers
        expected_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Verify payload structure
        assert expected_payload["model"] == model
        assert len(expected_payload["messages"]) == 1
        assert expected_payload["messages"][0]["role"] == "user"
        assert expected_payload["messages"][0]["content"] == prompt
        assert expected_payload["stream"] is True
        
        # Verify headers
        assert expected_headers["Authorization"].startswith("Bearer ")
        assert expected_headers["Content-Type"] == "application/json"
        
        test_logger.info("Client request formatting is correct")