# Claude Codex

****  Monitor and intercept Claude Code API requests with multi-backend LLM support.

A hackable Claude API proxy for monitoring AI agent requests and connecting multiple LLM backends. No complex AI frameworks - just FastAPI, requests, and clean code you can easily modify.


## Architecture

```
┌─────────────┐    ANTHROPIC_BASE_URL=localhost:8082    ┌─────────────────┐
│ Claude Code │ ──────────────────────────────────────► │  Claude Codex   │
└─────────────┘            POST /v1/messages            │   (server.py)   │
                                                        └─────────────────┘
                                                                  │
                                                    Auto-Detection │
                                                   (env vars/keys) │
                                                                  │
                           ┌──────────────────────────────────────┼──────────────────────────────────────┐
                           │                                      │                                      │
                           ▼                                      ▼                                      ▼
                 ┌─────────────────┐                   ┌─────────────────┐                   ┌─────────────────┐
                 │  AWS Bedrock    │                   │OpenAI Compatible│                   │  Future Backend │
                 │  (bedrock.py)   │                   │  (openai_compat │                   │                 │
                 │                 │                   │      .py)       │                   │                 │
                 └─────────────────┘                   └─────────────────┘                   └─────────────────┘
                           │                                      │                                      │
                           ▼                                      ▼                                      ▼
                 ┌─────────────────┐                   ┌─────────────────┐                   ┌─────────────────┐
                 │  AWS Bedrock    │                   │   OpenAI API    │                   │    Any LLM      │
                 │ Converse API    │                   │ Chat Completions│                   │   Provider      │
                 │                 │                   │                 │                   │                 │
                 │ • Claude Sonnet │                   │ • GPT-4/3.5     │                   │ • Local Models  │
                 │ • Claude Haiku  │                   │ • Gemini 2.0    │                   │ • Custom APIs   │
                 │ • Claude Opus   │                   │ • Ollama/LM Std │                   │ • Fine-tuned    │
                 └─────────────────┘                   └─────────────────┘                   └─────────────────┘
```

**How it works:**
1. Claude Code sends requests to Claude Codex (localhost:8082)
2. Claude Codex auto-detects backend (Bedrock vs OpenAI-compatible)
3. Claude Codex translates Claude API ↔ Provider API formats
4. Response flows back to Claude Code

**Backend Support:**
- **AWS Bedrock**: Claude Sonnet/Haiku/Opus (native format)
- **OpenAI**: GPT-4, GPT-3.5 (via OpenAI API)
- **Google Gemini**: 2.0-flash, 1.5-pro (via OpenAI-compatible API)
- **Local**: Ollama, LM Studio (via OpenAI-compatible API)


## Project Goals

### Primary: Hackable LLM Request Monitoring
- **See what Claude Code is actually doing** - intercept and log all requests/responses  
- **Monitor tool calling patterns** - watch which tools get called and when
- **Debug agentic workflows** - understand multi-step task execution in real-time
- **No AI frameworks required** - pure FastAPI + requests, easy to modify and extend

### Secondary: LLM Backend Flexibility
- **Connect any LLM** - swap between Claude, GPT-4, Gemini, local models instantly
- **Extend time limits** - run longer workflows by routing through your own infrastructure  
- **Add new backends** - simple translator pattern makes adding new LLMs easy
- **Custom model access** - use fine-tuned models or experimental endpoints

## Why Build This?

As developers working with AI agents, we often wonder:
- What requests is Claude Code actually making?
- How does it decide which tools to call?
- Can I run the same workflow on a different/cheaper model?
- How can I extend session time limits?

This proxy gives you **full visibility and control** without complex AI frameworks.

## Installation

Install the package directly from the source code:

```bash
pip install .
```

For development, install in editable mode:

```bash
pip install -e .
```

After installation, the command will be available as `claudecodex`.

## Development Setup

This project uses `uv` for package management and `ruff` for linting and formatting.

1.  **Install `uv`:**

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Create a virtual environment and install dependencies:**

    ```bash
    uv venv
    uv pip sync requirements.txt
    ```

3.  **Activate the virtual environment:**

    **macOS/Linux:**
    ```bash
    source .venv/bin/activate
    ```

    **Windows:**
    ```bash
    .venv\Scripts\activate
    ```

4.  **Run the linter:**

    ```bash
    ruff check src
    ```

## Quick Start

```bash
# Run with Bedrock (default)
claudecodex

# Run with Gemini
export LLM_BACKEND=openai_compatible
export OPENAI_API_KEY=your-gemini-key
export OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
claudecodex

# Connect Claude Code
export ANTHROPIC_BASE_URL=http://localhost:8082
```

## Backends

- **AWS Bedrock**: Claude Sonnet/Haiku/Opus (`LLM_BACKEND=bedrock`)
- **Google Gemini**: gemini-2.0-flash (`LLM_BACKEND=openai_compatible` + Gemini config)
- **OpenAI**: GPT-4/3.5 (`LLM_BACKEND=openai_compatible` + OpenAI config)  
- **Local**: Ollama, LM Studio (`LLM_BACKEND=openai_compatible` + local config)

## Configuration

**Important:** Do not commit the `.env` file to version control. It should be added to your `.gitignore` file.

### Environment Variables

```bash
# Backend selection
LLM_BACKEND=bedrock|openai_compatible  # default: bedrock
SERVER_PORT=8082                       # default: 8082

# Bedrock backend
AWS_PROFILE=your-profile               # default: saml
AWS_DEFAULT_REGION=us-east-1           # default: us-east-1
BEDROCK_MODEL_ID=model-id              # default: us.anthropic.claude-sonnet-4-*

# OpenAI-compatible backend
OPENAI_API_KEY=your-key                # required for openai_compatible
OPENAI_BASE_URL=endpoint-url           # provider-specific
OPENAI_MODEL=model-name                # default: gemini-2.0-flash

# Claude Code integration
ANTHROPIC_BASE_URL=http://localhost:8082
```

## Testing

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Integration tests (requires API keys)
pytest tests/integration/ -v
```

## File Structure

```
├── pyproject.toml                    # Modern packaging configuration
├── src/claudecodex/                  # Core package
│   ├── __init__.py                   # Package info
│   ├── main.py                       # Clean entry point (62 lines)
│   ├── server.py                     # Complete server logic (200 lines)
│   ├── bedrock.py                    # All Bedrock functionality (327 lines)
│   ├── openai_compatible.py          # All OpenAI-compatible functionality (447 lines)
│   ├── models.py                     # Pydantic models
│   └── logging_config.py             # Logging configuration
├── tests/                            # Test package
│   ├── unit/                         # Fast tests with mocks
│   │   ├── test_api_endpoints.py     # API endpoint tests
│   │   └── test_openai_compatible.py # OpenAI service tests
│   └── integration/                  # Full system tests
│       ├── test_bedrock_integration.py # Bedrock end-to-end tests
│       └── test_gemini_integration.py  # Gemini end-to-end tests
├── logs/                             # Generated log files
├── requirements.txt                  # Dependencies
└── README.md                         # This file
```

**Simple Architecture**: Core functionality organized in focused files
- `server.py` - FastAPI server with backend routing and API endpoints
- `bedrock.py` - AWS Bedrock integration with Claude API translation
- `openai_compatible.py` - OpenAI-compatible API integration (OpenAI, Gemini, local models)

## API Functions

### Server Functions (`server.py`)
- `get_backend_type()` - Determine backend from environment  
- `call_llm_service(request)` - Route requests to backends
- `count_llm_tokens(request)` - Token counting
- `get_backend_info()` - Runtime configuration info
- `create_message(request)` - `/v1/messages` endpoint
- `count_tokens(request)` - `/v1/messages/count_tokens` endpoint
- `health()` - `/health` endpoint

### Bedrock Backend (`bedrock.py`)
- `call_bedrock_converse(request)` - AWS Bedrock API calls
- `count_request_tokens(request)` - Bedrock token counting
- `get_bedrock_client()` - AWS client setup
- `convert_to_bedrock_messages()` - Claude → Bedrock format
- `create_claude_response()` - Bedrock → Claude format

### OpenAI-Compatible Backend (`openai_compatible.py`)
- `call_openai_compatible_chat(request)` - OpenAI API calls
- `count_openai_tokens(request)` - Token estimation
- `get_openai_compatible_client()` - HTTP session setup
- `convert_to_openai_messages()` - Claude → OpenAI format
- `create_claude_response_from_openai()` - OpenAI → Claude format

