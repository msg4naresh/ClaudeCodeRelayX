"""
Core service for AWS Bedrock Converse API integration.

This module provides the main business logic for:
- Making calls to AWS Bedrock Converse API  
- Parameter translation and validation
- Error handling and logging
- Response processing and formatting
"""

import logging
from botocore.exceptions import ClientError
from fastapi import HTTPException

from ...models import MessagesRequest, MessagesResponse, TokenCountRequest, TokenCountResponse
from .translator import (
    convert_to_bedrock_messages, extract_system_message, 
    count_tokens_from_messages, create_claude_response
)
from .client import get_bedrock_client, get_model_id


logger = logging.getLogger(__name__)


def call_bedrock_converse(request: MessagesRequest) -> MessagesResponse:
    """Execute Claude API request via AWS Bedrock."""
    try:
        # Get Bedrock client and model ID
        bedrock_client = get_bedrock_client()
        model_id = get_model_id()
        
        # Convert messages and extract system
        bedrock_messages = convert_to_bedrock_messages(request)
        system_message = extract_system_message(request)
        
        # Build inference configuration
        inference_config = {
            "temperature": request.temperature or 0.7,
            "maxTokens": request.max_tokens
        }
        
        # Add optional parameters
        if request.top_p:
            inference_config["topP"] = request.top_p
        if request.top_k:
            inference_config["topK"] = request.top_k
            
        # Handle tool configuration
        if request.tools:
            tool_config = {
                "tools": []
            }
            
            for tool in request.tools:
                bedrock_tool = {
                    "toolSpec": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "inputSchema": {
                            "json": tool.input_schema
                        }
                    }
                }
                tool_config["tools"].append(bedrock_tool)
            
            # Handle tool choice
            if request.tool_choice:
                if request.tool_choice.get("type") == "tool":
                    tool_config["toolChoice"] = {
                        "tool": {
                            "name": request.tool_choice["name"]
                        }
                    }
                elif request.tool_choice.get("type") == "auto":
                    tool_config["toolChoice"] = {"auto": {}}
                elif request.tool_choice.get("type") == "any":
                    tool_config["toolChoice"] = {"any": {}}
        
        # Prepare Bedrock Converse request
        converse_params = {
            "modelId": model_id,
            "messages": bedrock_messages,
            "inferenceConfig": inference_config
        }
        
        # Add system message if present
        if system_message:
            converse_params["system"] = [{"text": system_message}]
            
        # Add tool configuration if present
        if request.tools:
            converse_params["toolConfig"] = tool_config
        
        # Make the API call
        response = bedrock_client.converse(**converse_params)
        
        # Convert response back to Claude format
        return create_claude_response(response, model_id)
        
    except ClientError as e:
        error_message = e.response.get('Error', {}).get('Message', str(e))
        logger.error(f"Bedrock error: {error_message}")
        raise HTTPException(status_code=500, detail=f"Bedrock error: {error_message}")
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


def count_request_tokens(request: TokenCountRequest) -> TokenCountResponse:
    """Count tokens for Bedrock models."""
    try:
        token_count = count_tokens_from_messages(request.messages, request.system)
        return TokenCountResponse(input_tokens=token_count)
        
    except Exception as e:
        logger.error(f"Error counting tokens: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error counting tokens: {str(e)}")