"""
FastAPI server providing Claude API compatibility for multiple LLM backends.

This module contains:
- FastAPI application setup and configuration
- API endpoint definitions (/v1/messages, /v1/messages/count_tokens)  
- Request/response middleware for logging
- Health check and status endpoints
- Backend-agnostic LLM service routing (OpenAI API Compatible, Bedrock)
- Entry point for running the server
"""

import json
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .models import MessagesRequest, TokenCountRequest
from .service_router import call_llm_service, count_llm_tokens, get_backend_info
from .logging_config import setup_logging, log_request_response


# Initialize logging
logger, request_logger = setup_logging()

# Create FastAPI application
app = FastAPI(
    title="Claude Multi-Backend Proxy Server",
    description="Claude API compatible server supporting multiple LLM backends (Bedrock, OpenAI)",
    version="1.0.0"
)

# Add CORS middleware for web client support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log HTTP requests and responses."""
    start_time = time.time()
    
    # Capture request details for POST requests
    request_body = None
    if request.method == "POST":
        body = await request.body()
        if body:
            try:
                request_body = json.loads(body.decode())
            except json.JSONDecodeError:
                request_body = {"raw": body.decode()[:200]}
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration = time.time() - start_time
    
    # Logging is handled in individual endpoints to avoid duplication
    
    return response


@app.post("/v1/messages")
async def create_message(request: MessagesRequest):
    """Handle /v1/messages endpoint - main Claude API compatibility."""
    start_time = time.time()
    
    # Request processing (verbose logging removed for clean output)
    
    try:
        # Call configured LLM service
        result = call_llm_service(request)
        duration = time.time() - start_time
        
        # Log successful request/response
        log_request_response(
            request_logger=request_logger,
            main_logger=logger,
            endpoint="/v1/messages",
            request_data=request.model_dump(),
            response_data=result.model_dump(),
            duration=duration,
            status_code=200
        )
        
        return result
        
    except Exception as e:
        duration = time.time() - start_time
        error_msg = str(e)
        
        # Log failed request
        log_request_response(
            request_logger=request_logger, 
            main_logger=logger,
            endpoint="/v1/messages",
            request_data=request.model_dump(),
            response_data={},
            duration=duration,
            status_code=500,
            error=error_msg
        )
        
        raise


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: TokenCountRequest):
    """Handle /v1/messages/count_tokens endpoint."""
    start_time = time.time()
    
    try:
        result = count_llm_tokens(request)
        duration = time.time() - start_time
        
        # Log successful token count
        log_request_response(
            request_logger=request_logger,
            main_logger=logger,
            endpoint="/v1/messages/count_tokens",
            request_data=request.model_dump(),
            response_data={"input_tokens": result.input_tokens},
            duration=duration,
            status_code=200
        )
        
        return result
        
    except Exception as e:
        duration = time.time() - start_time
        error_msg = str(e)
        
        # Log failed token count
        log_request_response(
            request_logger=request_logger,
            main_logger=logger,
            endpoint="/v1/messages/count_tokens", 
            request_data=request.model_dump(),
            response_data={},
            duration=duration,
            status_code=500,
            error=error_msg
        )
        
        raise


@app.get("/")
async def root():
    """Root endpoint with server info."""
    backend_info = get_backend_info()
    return {
        "message": "Claude Code-Compatible Multi-Backend Server",
        "backend": backend_info["backend"],
        "model": backend_info["model"]
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    backend_info = get_backend_info()
    return {
        "status": "healthy",
        "backend": backend_info["backend"],
        "model": backend_info["model"]
    }