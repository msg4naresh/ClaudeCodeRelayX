# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python main.py

# Run all tests  
pytest tests/ -v

# Run unit tests only
pytest tests/unit/ -v

# Run integration tests (requires API keys)
pytest tests/integration/ -v

# Start server with specific backend
export LLM_BACKEND=bedrock  # or openai_compatible
python main.py
```

## Architecture Overview

This is a multi-backend LLM proxy (`relayx`) that provides Claude API compatibility for various LLM providers.

### Core Components

- **`proxy_server.py`** - FastAPI application with `/v1/messages`, `/v1/messages/count_tokens`, and `/health` endpoints
- **`service_router.py`** - Backend routing logic that determines which backend to use based on environment variables
- **`models.py`** - Pydantic models defining Claude API request/response structures
- **`backends/`** - Backend-specific implementations:
  - `bedrock/` - AWS Bedrock Claude models (native Claude API format)
  - `openai_compatible/` - OpenAI Chat Completions API format (supports OpenAI, Gemini, local models)

### Backend Selection Logic

1. If `LLM_BACKEND` environment variable is set, use that backend
2. Otherwise, auto-detect: if `OPENAI_API_KEY` exists, use `openai_compatible`; else use `bedrock`

### Translation Layer

Each backend has translator modules that convert between Claude API format and the backend's native format:
- `bedrock/translator.py` - Claude ↔ AWS Bedrock Converse API
- `openai_compatible/translator.py` - Claude ↔ OpenAI Chat Completions API

### Key Environment Variables

- `LLM_BACKEND` - Force backend selection (bedrock/openai_compatible)
- `SERVER_PORT` - Server port (default: 8082)
- `AWS_PROFILE`, `AWS_DEFAULT_REGION` - AWS Bedrock configuration
- `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` - OpenAI-compatible backend configuration

### Testing Structure

- `tests/unit/` - Fast tests with mocks for API endpoints and backend services
- `tests/integration/` - Full system tests requiring actual API keys and external services