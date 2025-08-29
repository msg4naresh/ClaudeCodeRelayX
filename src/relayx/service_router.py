"""
Backend service router for LLM API calls.

This module provides backend-agnostic service routing:
- Route requests to Bedrock or OpenAI-compatible backends based on configuration
- Maintain consistent interface for API server
- Handle backend-specific error handling and logging
- Support runtime backend switching via environment variables

Supported Backends:
- bedrock: AWS Bedrock Claude models (native Claude API format)
- openai_compatible: Any provider implementing OpenAI Chat Completions API
  (OpenAI, Gemini, Azure OpenAI, local models, etc.)
"""

import os
import logging
from typing import Literal

from .models import MessagesRequest, MessagesResponse, TokenCountRequest, TokenCountResponse
from .backends.bedrock.service import call_bedrock_converse, count_request_tokens
from .backends.openai_compatible.service import call_openai_compatible_chat, count_openai_tokens


logger = logging.getLogger(__name__)

BackendType = Literal["bedrock", "openai_compatible"]


def get_backend_type() -> BackendType:
    """Get configured backend type from environment variables."""
    # Check for explicit backend selection first
    explicit_backend = os.environ.get("LLM_BACKEND")
    if explicit_backend:
        backend = explicit_backend.lower()
        if backend not in ["bedrock", "openai_compatible"]:
            raise ValueError(f"Unsupported LLM backend: {backend}. Must be 'bedrock' or 'openai_compatible'")
        return backend
    
    # Auto-detect backend based on available API keys (prioritize OpenAI-compatible/Gemini)
    if os.environ.get("OPENAI_API_KEY"):
        return "openai_compatible"
    else:
        return "bedrock"


def call_llm_service(request: MessagesRequest) -> MessagesResponse:
    """Route request to appropriate backend service."""
    backend = get_backend_type()
    
    logger.debug(f"Routing request to {backend} backend")
    
    if backend == "bedrock":
        return call_bedrock_converse(request)
    elif backend == "openai_compatible":
        return call_openai_compatible_chat(request)
    else:
        # This should not happen due to validation in get_backend_type()
        raise ValueError(f"Unsupported backend: {backend}")


def count_llm_tokens(request: TokenCountRequest) -> TokenCountResponse:
    """Route token counting to appropriate backend."""
    backend = get_backend_type()
    
    logger.debug(f"Routing token count request to {backend} backend")
    
    if backend == "bedrock":
        return count_request_tokens(request)
    elif backend == "openai_compatible":
        return count_openai_tokens(request)
    else:
        # This should not happen due to validation in get_backend_type()
        raise ValueError(f"Unsupported backend: {backend}")


def get_backend_info() -> dict:
    """Get current backend configuration info."""
    backend = get_backend_type()
    
    if backend == "bedrock":
        from .backends.bedrock.client import get_model_id
        return {
            "backend": "bedrock",
            "model": get_model_id(),
            "region": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            "profile": os.environ.get("AWS_PROFILE", "saml")
        }
    elif backend == "openai_compatible":
        from .backends.openai_compatible.client import get_openai_compatible_model, get_openai_compatible_base_url
        return {
            "backend": "openai_compatible", 
            "model": get_openai_compatible_model(),
            "base_url": get_openai_compatible_base_url(),
            "api_key_configured": bool(os.environ.get("OPENAI_API_KEY"))
        }
    else:
        return {"backend": "unknown", "error": f"Unsupported backend: {backend}"}


def validate_backend_config() -> bool:
    """Validate current backend configuration is complete."""
    try:
        backend = get_backend_type()

        if backend == "openai_compatible":
            # Check OpenAI-compatible configuration
            api_key = os.environ.get("OPENAI_API_KEY")
            return bool(api_key)  # API key is required

        if backend == "bedrock":
            # Check AWS configuration
            aws_profile = os.environ.get("AWS_PROFILE")
            aws_region = os.environ.get("AWS_DEFAULT_REGION")
            # Note: We don't validate actual AWS credentials here as that would require a test call
            return True  # Basic env vars are optional due to AWS credential chain
            
        return False
        
    except Exception as e:
        logger.error(f"Backend config validation failed: {str(e)}")
        return False