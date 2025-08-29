"""
OpenAI-compatible API client configuration and management.

This module handles HTTP client initialization for any provider that implements
the OpenAI Chat Completions API standard, including:
- OpenAI (api.openai.com)
- Google Gemini (generativelanguage.googleapis.com)
- Azure OpenAI
- Local models (Ollama, LM Studio, etc.)
- Other OpenAI-compatible providers

Features:
- HTTP client initialization using requests library
- API authentication and configuration
- Model and endpoint configuration
- Request session management with retry policies
"""

import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from fastapi import HTTPException


def get_openai_compatible_client():
    """Get configured HTTP session for OpenAI-compatible providers."""
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="OPENAI_API_KEY environment variable is required"
            )
        
        base_url = os.environ.get("OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
        
        # Create session with retry strategy
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set default headers
        session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "claude-bedrock-proxy/1.0.0"
        })
        
        # Store base URL for later use
        session.base_url = base_url
        
        return session
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create OpenAI-compatible client: {str(e)}"
        )


def get_openai_compatible_model() -> str:
    """Get model name from environment variables."""
    return os.environ.get("OPENAI_MODEL", "gemini-2.0-flash")


def get_openai_compatible_base_url() -> str:
    """Get base URL from environment variables."""
    return os.environ.get("OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")