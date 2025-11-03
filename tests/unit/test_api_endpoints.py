"""
Unit tests for Claude-Bedrock Proxy API endpoints.

This module contains unit tests for:
- FastAPI endpoint functionality
- Request/response validation  
- Error handling scenarios
- Mocked Bedrock API interactions

Tests use FastAPI TestClient with mocked AWS Bedrock calls to ensure
reliable testing without external dependencies.
"""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from claudecodex.server import app


# Initialize test client
client = TestClient(app)

# Mock Bedrock response for successful API calls
mock_bedrock_response = {
    'output': {
        'message': {
            'content': [
                {'text': 'Hello! How can I help you today?'}
            ]
        },
        'stopReason': 'end_turn'
    },
    'usage': {
        'inputTokens': 10,
        'outputTokens': 8
    }
}


@patch('claudecodex.bedrock.get_bedrock_client')
def test_create_message_success(mock_get_client):
    """
    Test successful message creation with all parameters.
    
    Verifies:
    - Request processing with full parameter set
    - Proper response structure
    - Token usage information
    - Claude API compatibility
    """
    # Setup mock Bedrock client
    mock_client = mock_get_client.return_value
    mock_client.converse.return_value = mock_bedrock_response
    
    # Test request with comprehensive parameters
    request_data = {
        "model": "claude-3-5-sonnet",
        "max_tokens": 1000,
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "system": "You are a helpful assistant",
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "stop_sequences": ["END"],
        "stream": False,
        "metadata": {"user_id": "123"}
    }
    
    response = client.post("/v1/messages", json=request_data)
    
    # Verify successful response
    assert response.status_code == 200
    data = response.json()
    
    # Check Claude API response structure
    assert "id" in data
    assert data["role"] == "assistant"
    assert data["type"] == "message"
    assert len(data["content"]) == 1
    assert data["content"][0]["type"] == "text"
    assert data["content"][0]["text"] == "Hello! How can I help you today?"
    assert data["stop_reason"] == "end_turn"
    
    # Check usage information
    assert "usage" in data
    assert "input_tokens" in data["usage"]
    assert "output_tokens" in data["usage"]
    assert data["usage"]["input_tokens"] == 10
    assert data["usage"]["output_tokens"] == 8


@patch('claudecodex.bedrock.get_bedrock_client')
def test_bedrock_error_handling(mock_get_client):
    """
    Test error handling when AWS Bedrock API fails.
    
    Simulates various Bedrock API errors and verifies proper
    error propagation and HTTP status codes.
    """
    from botocore.exceptions import ClientError
    
    # Setup mock client to raise ClientError
    mock_client = mock_get_client.return_value
    error_response = {
        'Error': {
            'Code': 'ValidationException',
            'Message': 'Invalid model ID'
        }
    }
    mock_client.converse.side_effect = ClientError(error_response, 'Converse')
    
    request_data = {
        "model": "invalid-model",
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": "Hello"}]
    }
    
    response = client.post("/v1/messages", json=request_data)
    
    # Verify error response
    assert response.status_code == 500
    assert "Bedrock error" in response.json()["detail"]


def test_invalid_request_format():
    """
    Test validation of invalid request formats.
    
    Verifies that FastAPI properly validates request structure
    and returns appropriate HTTP 422 for validation errors.
    """
    request_data = {
        "model": "claude-3-5-sonnet",
        # Missing required max_tokens field
        "messages": [{"role": "user", "content": "Hello"}]
    }
    
    response = client.post("/v1/messages", json=request_data)
    assert response.status_code == 422  # Validation error


def test_token_counting_endpoint():
    """
    Test token counting endpoint functionality.
    
    Verifies:
    - Token estimation without API calls
    - Proper response structure
    - Non-zero token counts for content
    """
    request_data = {
        "model": "claude-3-5-sonnet",
        "messages": [
            {"role": "user", "content": "Hello world"}
        ],
        "system": "You are helpful"
    }
    
    response = client.post("/v1/messages/count_tokens", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    assert "input_tokens" in data
    assert data["input_tokens"] > 0


def test_health_endpoint():
    """
    Test health check endpoint for monitoring.
    
    Verifies health endpoint returns proper status and model information.
    """
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "model" in data


def test_root_endpoint():
    """
    Test root endpoint for server information.
    
    Verifies root endpoint provides server identification and model info.
    """
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "model" in data


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])