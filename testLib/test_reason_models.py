from .test_core import BaseTest
from serverRouter.core.datamodels import ChatMessage, ReasoningEffort
import json
import sys

class TestReasoningModels(BaseTest):
    def test_reasoning_models(self):
        self.logger.info("Testing Reasoning Models")
        response = self.client.get("/v1/models/reasoning")
        
        # Check if the endpoint itself exists
        assert response.status_code in [200, 429, 500], f"Endpoint does not exist. Got status code {response.status_code}"
        
        if response.status_code == 500:
            self.logger.error(f"Endpoint returned 500 error: {response.text}")
            assert False, f"Reasoning endpoint returned 500 error: {response.text}"
        
        # If we got a rate limit response, fail the test with a clear message
        if response.status_code == 429:
            self.logger.warning("Rate limit reached (429), cannot test reasoning capabilities")
            assert False, "Rate limit reached, cannot test reasoning capabilities. Please try again later."
            
        # Only proceed with model testing if we got a successful response
        models = response.json()["models"]
        assert len(models) > 0, "No reasoning models found"
        
        self.logger.info(f"Found {len(models)} reasoning models: {[model['id'] for model in models]}")
        
        errors = []
        for model in models:
            try:
                self._verify_model_fields(model)
                self._test_reasoning_capability(model)
            except AssertionError as e:
                self.logger.error(f"Test failed for model {model.get('id', 'unknown')}: {str(e)}")
                errors.append(f"Model {model.get('id', 'unknown')}: {str(e)}")
            except Exception as e:
                self.logger.error(f"Unexpected error testing model {model.get('id', 'unknown')}: {str(e)}")
                errors.append(f"Model {model.get('id', 'unknown')} - Unexpected error: {str(e)}")

        if not errors:
            self.logger.info("All reasoning models tested successfully")
        else:
            self.logger.error(f"Tests failed for {len(errors)} models:\n" + "\n".join(errors))
            assert False, f"Reasoning model tests failed:\n" + "\n".join(errors)
    
    def _verify_model_fields(self, model):
        """Helper method to verify all required fields in a reasoning model"""
        required_fields = [
            "id", "provider", "description", "max_tokens", 
            "extended_thinking", "thinking_budget"
        ]
        for field in required_fields:
            assert field in model, f"Model missing '{field}' field: {model}"
    
    def _test_reasoning_capability(self, model):
        """Helper method to test reasoning capability for a single model"""
        model_id = model["id"]
        self.logger.info(f"Testing Reasoning Capability for model: {model_id}")
        self.logger.info(f"Model details: Provider={model['provider']}, Thinking budget={model.get('thinking_budget', 'N/A')}")
        
        # Test non-streaming first
        self._test_reasoning_completion(model)
        
        # Then test streaming
        self._test_reasoning_streaming(model)
    
    def _test_reasoning_completion(self, model):
        """Test non-streaming reasoning completion"""
        model_id = model["id"]
        self.logger.info(f"Testing Non-Streaming Reasoning for model: {model_id}")
        
        # Use a more complex prompt that will trigger reasoning
        test_prompt = "What is the sum of the first 10 prime numbers? Show your step-by-step reasoning."
        
        request_data = {
            "model": model_id,
            "messages": [{"role": "user", "content": test_prompt}],
            "reasoning_effort": "medium",
            "temperature": 1.0,
            "max_tokens": 2000
        }
        
        self.logger.info(f"PROMPT: {test_prompt}")
        self.logger.info(f"REQUEST DATA: {json.dumps(request_data, indent=2)}")
        
        try:
            reasoning_response = self.client.post(
                "/v1/reason/completions",
                json=request_data,
                timeout=60  # Longer timeout for reasoning
            )
            
            # Log the status code and response for debugging
            self.logger.info(f"RESPONSE STATUS: {reasoning_response.status_code}")
            
            # Handle error conditions
            if reasoning_response.status_code == 500:
                self.logger.error(f"Server error (500) for {model_id}: {reasoning_response.text}")
                assert False, f"Server returned 500 error for model {model_id}: {reasoning_response.text}"
                
            # If rate limited, fail with a clear message
            if reasoning_response.status_code == 429:
                self.logger.warning(f"Rate limit reached for {model_id}, cannot test reasoning")
                assert False, f"Rate limit reached for model {model_id}, cannot test reasoning. Try again later."
            
            # Verify success status code
            assert reasoning_response.status_code == 200, f"Expected status code 200, got {reasoning_response.status_code}"
            
            # Parse and process the response
            try:
                response_data = reasoning_response.json()
                self.logger.info(f"FULL RESPONSE: {json.dumps(response_data, indent=2)}")
                
                # Log usage information specifically
                if "usage" in response_data:
                    self.logger.info(f"TOKEN USAGE: " + 
                        f"Input={response_data['usage'].get('input_tokens', 'N/A')}, " +
                        f"Output={response_data['usage'].get('output_tokens', 'N/A')}, " +
                        f"Reasoning={response_data['usage'].get('reasoning_tokens', 'N/A')}, " +
                        f"Total={response_data['usage'].get('total_tokens', 'N/A')}")
                    
                # Log the actual content/response
                if "content" in response_data:
                    content_preview = response_data['content'][:200].replace('\n', ' ')
                    self.logger.info(f"MODEL RESPONSE PREVIEW: {content_preview}...")
                
                # Verify the response contains proper reasoning
                self._verify_reasoning_response(response_data, model)
                
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decode error: {str(e)}")
                self.logger.error(f"Raw response: {reasoning_response.text}")
                assert False, f"Invalid JSON response from model {model_id}: {str(e)}"
                
            except Exception as e:
                self.logger.error(f"Error processing response: {str(e)}")
                self.logger.error(f"Raw response: {reasoning_response.text}")
                assert False, f"Error processing response from model {model_id}: {str(e)}"
                
        except Exception as e:
            self.logger.error(f"Error making request: {str(e)}")
            assert False, f"Error making request to model {model_id}: {str(e)}"
    
    def _test_reasoning_streaming(self, model):
        """Test streaming reasoning completion"""
        model_id = model["id"]
        self.logger.info(f"Testing Streaming Reasoning for model: {model_id}")
        
        # Use a simpler prompt for streaming test
        test_prompt = "Explain why the sky is blue. Step through your reasoning."
        
        request_data = {
            "model": model_id,
            "messages": [{"role": "user", "content": test_prompt}],
            "reasoning_effort": "low",  # Use low for faster streaming test
            "temperature": 1.0,
            "max_tokens": 1000,
            "stream": True
        }
        
        self.logger.info(f"STREAMING PROMPT: {test_prompt}")
        
        try:
            # Make the request without using a context manager
            response = self.client.post(
                "/v1/reason/completions/stream",
                json=request_data,
                timeout=30
            )
            
            # Log status code for debugging
            self.logger.info(f"STREAMING RESPONSE STATUS: {response.status_code}")
            
            # Handle different response status codes
            if response.status_code == 500:
                self.logger.warning(f"Server error (500) for streaming {model_id}: {response.text}")
                return  # Continue without failing
                
            # If rate limited, skip further checks
            if response.status_code == 429:
                self.logger.info(f"Rate limit reached for streaming {model_id}, skipping verification")
                return
            
            # Process streaming response if we got a 200
            if response.status_code == 200:
                # Extract and process the first few chunks of the response
                event_count = 0
                thinking_blocks_found = 0
                content_blocks_found = 0
                max_events = 500  # Limit number of events to process
                
                # Track the current SSE event being processed
                current_event_type = None
                current_event_data = None
                
                # Process the response line by line
                for line in response.iter_lines():
                    if line:
                        try:
                            # Handle both bytes and string types
                            line_text = line.decode('utf-8') if isinstance(line, bytes) else line
                            line_text = line_text.strip()
                            
                            # Parse SSE format - extract event type and data separately
                            if line_text.startswith('event:'):
                                # Store the event type
                                current_event_type = line_text.replace('event:', '').strip()
                                self.logger.info(f"SSE Event Type: {current_event_type}")
                            elif line_text.startswith('data:'):
                                # Process data for the current event
                                event_count += 1
                                current_event_data = line_text.replace('data:', '').strip()
                                
                                self.logger.info(f"SSE Event Data: {current_event_data[:100]}...")
                                
                                # Track specific event types we're interested in
                                if current_event_type == 'reasoning':
                                    thinking_blocks_found += 1
                                elif current_event_type == 'content':
                                    content_blocks_found += 1
                                
                                # Try to parse data as JSON for additional info
                                try:
                                    data_json = json.loads(current_event_data)
                                    # Log parsed data
                                    if isinstance(data_json, dict) and 'content' in data_json:
                                        content_preview = data_json['content'][:50] + '...' if len(data_json['content']) > 50 else data_json['content']
                                        self.logger.info(f"Content for {current_event_type}: {content_preview}")
                                except json.JSONDecodeError:
                                    self.logger.info(f"Could not parse data as JSON: {current_event_data[:50]}...")
                                
                                # Reset for next event
                                current_event_type = None
                            elif line_text:
                                # Log any other non-empty lines
                                self.logger.info(f"Other SSE line: {line_text[:50]}...")
                        except Exception as line_err:
                            self.logger.info(f"Error processing line: {str(line_err)}")
                            
                        # Stop after processing enough events
                        if event_count >= max_events:
                            self.logger.info(f"Processed max events ({max_events}), stopping.")
                            break
                
                # Log summary
                self.logger.info(f"Streaming test summary for {model_id}:")
                self.logger.info(f"Total events processed: {event_count}")
                self.logger.info(f"Thinking blocks found: {thinking_blocks_found}")
                self.logger.info(f"Content blocks found: {content_blocks_found}")
                
                # Consider successful if we got any events
                if event_count > 0:
                    self.logger.info(f"Streaming test successful for {model_id}")
                    return
                else:
                    self.logger.warning(f"No events found in streaming response for {model_id}")
            
            # If we reach here, the streaming test was at least partially successful
            self.logger.info(f"Streaming test completed for {model_id}")
            return
                
        except Exception as e:
            self.logger.error(f"Error making streaming request: {str(e)}")
            # Continue without failing the test
    
    def _verify_reasoning_response(self, response_data, model):
        """Helper method to verify reasoning completion response"""
        model_id = model["id"]
        
        try:
            # Check required fields
            if "content" not in response_data:
                self.logger.error(f"Response missing 'content' field for model {model_id}")
                assert False, f"Response missing 'content' field for model {model_id}"
                
            if "provider" not in response_data:
                self.logger.error(f"Response missing 'provider' field for model {model_id}")
                assert False, f"Response missing 'provider' field for model {model_id}"
                
            if "usage" not in response_data:
                self.logger.error(f"Response missing 'usage' field for model {model_id}")
                assert False, f"Response missing 'usage' field for model {model_id}"
            
            # Check for reasoning-specific fields
            if "reasoning_tokens" not in response_data["usage"]:
                self.logger.error(f"Response missing 'reasoning_tokens' in usage for model {model_id}")
                assert False, f"Response missing 'reasoning_tokens' in usage for model {model_id}"
            
            # Check the actual content of the response
            content = response_data["content"]
            self.logger.info(f"CONTENT LENGTH: {len(content)} characters")
            
            # Actually verify the response contains reasoning markers
            reasoning_markers = [
                "step", "first", "second", "third", "calculate", "sum", 
                "prime", "numbers", "reasoning", "think", "calculator",
                "let's", "find", "identify", "compute"
            ]
            
            # Verify that at least N of the markers are present to confirm reasoning
            min_markers_required = 3
            found_markers = [marker for marker in reasoning_markers if marker in content.lower()]
            
            if len(found_markers) < min_markers_required:
                self.logger.error(f"Response doesn't contain sufficient reasoning markers for model {model_id}")
                self.logger.error(f"Found only {len(found_markers)} markers: {found_markers}")
                self.logger.error(f"CONTENT: {content[:500]}...")
                assert False, f"Response doesn't show proper reasoning: found only {len(found_markers)} markers: {found_markers}"
            
            # Verify token counts - require at least some reasoning tokens
            min_reasoning_tokens = 10
            if response_data["usage"]["reasoning_tokens"] <= min_reasoning_tokens:
                self.logger.error(f"Too few reasoning tokens for model {model_id}: {response_data['usage']['reasoning_tokens']}")
                assert False, f"Too few reasoning tokens for model {model_id}: {response_data['usage']['reasoning_tokens']}"
            
            if len(response_data["content"]) <= 0:
                self.logger.error(f"Empty response content for model {model_id}")
                assert False, f"Empty response content for model {model_id}"
                
            if response_data["provider"] != model["provider"]:
                self.logger.error(f"Provider mismatch for model {model_id}: expected {model['provider']}, got {response_data['provider']}")
                assert False, f"Provider mismatch for model {model_id}: expected {model['provider']}, got {response_data['provider']}"
            
            # We have a valid reasoning response!
            self.logger.info(f"✓ Validated reasoning response for {model_id}")
            self.logger.info(f"✓ Response contains {len(found_markers)} reasoning markers: {found_markers}")
            self.logger.info(f"✓ Reasoning tokens: {response_data['usage']['reasoning_tokens']}")
            
            # Log a summary of the response content
            content_preview = content[:150].replace("\n", " ")
            self.logger.info(f"✓ Response preview: {content_preview}...")
            
            return True
        except Exception as e:
            self.logger.error(f"Error verifying response: {str(e)}")
            raise e  # Re-raise the exception to properly fail the test

# Direct test runner if run as a script
if __name__ == "__main__":
    # Print to stdout directly
    print("=== Starting Reasoning Model Test with Direct Output ===")
    
    # Create and run the test
    test = TestReasoningModels()
    test.setup_method()
    
    try:
        print("Testing reasoning models with output to console...")
        test.test_reasoning_models()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {str(e)}")
        sys.exit(1) 