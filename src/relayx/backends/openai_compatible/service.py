"""
Core service for OpenAI-compatible API integration.

This module provides the main business logic for any provider that implements
the OpenAI Chat Completions API standard, including:
- OpenAI (api.openai.com)
- Google Gemini (generativelanguage.googleapis.com)
- Azure OpenAI
- Local models (Ollama, LM Studio, etc.)
- Other OpenAI-compatible providers

Features:
- Making calls to OpenAI-compatible Chat Completions APIs
- Parameter translation and validation
- Error handling and logging
- Response processing and formatting
"""

import json
import logging
from requests.exceptions import RequestException, Timeout
from fastapi import HTTPException

from ...models import MessagesRequest, MessagesResponse, TokenCountRequest, TokenCountResponse
from .translator import (
    convert_to_openai_messages, convert_tools_to_openai,
    count_tokens_from_messages_openai, create_claude_response_from_openai
)
from .client import get_openai_compatible_client, get_openai_compatible_model


logger = logging.getLogger(__name__)


def call_openai_compatible_chat(request: MessagesRequest) -> MessagesResponse:
    """Execute Claude API request via OpenAI-compatible provider."""
    try:
        # Get OpenAI client and model
        openai_client = get_openai_compatible_client()
        model_id = get_openai_compatible_model()
        
        # Convert messages to OpenAI format
        openai_messages = convert_to_openai_messages(request)
        
        # Build request payload
        payload = {
            "model": model_id,
            "messages": openai_messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature or 0.7
        }
        
        # Add optional parameters
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        
        # Handle tool configuration
        if request.tools:
            openai_tools = convert_tools_to_openai(request.tools)
            payload["tools"] = openai_tools
            
            # Handle tool choice
            if request.tool_choice:
                if request.tool_choice.get("type") == "tool":
                    payload["tool_choice"] = {
                        "type": "function",
                        "function": {"name": request.tool_choice["name"]}
                    }
                elif request.tool_choice.get("type") == "auto":
                    payload["tool_choice"] = "auto"
                elif request.tool_choice.get("type") == "any":
                    payload["tool_choice"] = "required"  # OpenAI's equivalent
        
        # Handle stop sequences
        if request.stop_sequences:
            payload["stop"] = request.stop_sequences
        
        # Log request details (similar to bedrock_service pattern)
        logger.debug(f"OpenAI request: model={model_id}, messages={len(openai_messages)}")
        if request.tools:
            logger.debug(f"Tools: {len(request.tools)} functions")
        if request.stream:
            logger.warning("Streaming requested but not yet implemented for OpenAI")
        
        # Make the API call
        response = openai_client.post(
            f"{openai_client.base_url}/chat/completions",
            json=payload,
            timeout=120
        )
        
        # Check for HTTP errors
        response.raise_for_status()
        
        # Parse response
        response_data = response.json()
        
        # Convert response back to Claude format
        return create_claude_response_from_openai(response_data, model_id)
        
    except RequestException as e:
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_data = e.response.json()
                
                # Handle different error response formats
                # Gemini returns errors as a list: [{"error": {...}}]
                # OpenAI returns errors as a dict: {"error": {...}}
                if isinstance(error_data, list) and len(error_data) > 0:
                    error_obj = error_data[0].get('error', {})
                else:
                    error_obj = error_data.get('error', {})
                
                error_message = error_obj.get('message', str(e))
                status_code = e.response.status_code
                
                # Map OpenAI error codes to appropriate HTTP status codes
                if status_code == 401:
                    error_message = f"OpenAI authentication failed: {error_message}"
                elif status_code == 403:
                    error_message = f"OpenAI access forbidden: {error_message}"
                elif status_code == 429:
                    error_message = f"OpenAI rate limit exceeded: {error_message}"
                elif status_code >= 500:
                    error_message = f"OpenAI server error: {error_message}"
                else:
                    error_message = f"OpenAI API error: {error_message}"
                
                logger.error(f"OpenAI API error ({status_code}): {error_message}")
                raise HTTPException(status_code=status_code, detail=error_message)
                
            except json.JSONDecodeError:
                error_message = f"OpenAI API error: {e.response.text[:200]}"
                logger.error(error_message)
                raise HTTPException(status_code=e.response.status_code, detail=error_message)
        else:
            error_message = f"OpenAI connection error: {str(e)}"
            logger.error(error_message)
            raise HTTPException(status_code=500, detail=error_message)
    
    except json.JSONDecodeError as e:
        error_message = f"Failed to parse OpenAI response: {str(e)}"
        logger.error(error_message)
        raise HTTPException(status_code=500, detail=error_message)
    
    except Exception as e:
        logger.error(f"Unexpected error calling OpenAI: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


def count_openai_tokens(request: TokenCountRequest) -> TokenCountResponse:
    """Count tokens for OpenAI-compatible models."""
    try:
        token_count = count_tokens_from_messages_openai(request.messages, request.system)
        return TokenCountResponse(input_tokens=token_count)
        
    except Exception as e:
        logger.error(f"Error counting tokens for OpenAI: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error counting tokens: {str(e)}")