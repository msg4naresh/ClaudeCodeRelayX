#!/usr/bin/env python3
"""
Main entry point for ClaudeCodeRelayX.

This script starts the FastAPI server that provides Claude API compatibility
for multiple LLM backends (AWS Bedrock, OpenAI). It can be run directly or imported as a module.

Usage:
    python main.py
    
The server will start on http://0.0.0.0:8082 by default.

    
API Endpoints:
    POST /v1/messages - Claude API compatible chat completions
    POST /v1/messages/count_tokens - Token counting
    GET /health - Health check
    GET / - Server info
"""

import uvicorn
import logging
import os
from dotenv import load_dotenv

from src.relayx.proxy_server import app
from src.relayx.service_router import get_backend_info


def main():
    """
    Start the ClaudeCodeRelayX proxy server.
    
    Initializes logging, displays startup information, and starts the FastAPI
    server with uvicorn on all interfaces at port 8082.
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Get port from environment variable, fallback to 8082
    port = int(os.getenv("SERVER_PORT", 8082))
    
    # Setup logging for main entry point
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Display minimal startup information
    backend_info = get_backend_info()
    print(f"ClaudeCodeRelayX starting on http://localhost:{port}")
    print(f"Backend: {backend_info['backend']} | Model: {backend_info['model']}")
    
    # Start the server
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()