# Claude Code Multi-Backend Proxy

Claude Code API-compatible server that routes to multiple LLM backends: AWS Bedrock, Google Gemini, OpenAI, and local models.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run with Bedrock (default)
python main.py

# Run with Gemini
export LLM_BACKEND=openai_compatible
export OPENAI_API_KEY=your-gemini-key
export OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
python main.py

# Connect Claude Code
export ANTHROPIC_BASE_URL=http://localhost:8082
```

## Backends

- **AWS Bedrock**: Claude Sonnet/Haiku/Opus (`LLM_BACKEND=bedrock`)
- **Google Gemini**: gemini-2.0-flash (`LLM_BACKEND=openai_compatible` + Gemini config)
- **OpenAI**: GPT-4/3.5 (`LLM_BACKEND=openai_compatible` + OpenAI config)  
- **Local**: Ollama, LM Studio (`LLM_BACKEND=openai_compatible` + local config)

## Configuration

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
python run_tests.py

# Unit tests only  
pytest tests/unit/ -v

# Integration tests (requires API keys)
pytest tests/integration/ -v
```

## File Structure

```
├── main.py          # Main entry point
├── run_tests.py                      # Comprehensive test runner
├── src/claude_code_relayx/         # Core package
│   ├── __init__.py                   # Package info
│   ├── proxy_server.py              # FastAPI server
│   ├── service_router.py             # Backend routing logic
│   ├── models.py                     # Pydantic models
│   ├── logging_config.py             # Logging configuration
│   └── backends/                     # Backend implementations
│       ├── bedrock/                  # AWS Bedrock backend
│       │   ├── client.py             # AWS client setup
│       │   ├── service.py            # Bedrock API integration
│       │   └── translator.py         # Claude ↔ Bedrock translation
│       └── openai_compatible/        # OpenAI-compatible backend
│           ├── client.py             # HTTP client for OpenAI APIs
│           ├── service.py            # OpenAI API integration
│           └── translator.py         # Claude ↔ OpenAI translation
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

## API Functions

### Core Functions (`service_router.py`)
- `get_backend_type()` - Determine backend from environment  
- `call_llm_service(request)` - Route requests to backends
- `count_llm_tokens(request)` - Token counting
- `get_backend_info()` - Runtime configuration info

### Bedrock Backend (`backends/bedrock/`)
- `service.call_bedrock_converse(request)` - AWS Bedrock API calls
- `service.count_request_tokens(request)` - Bedrock token counting
- `client.get_bedrock_client()` - AWS client setup
- `translator.convert_to_bedrock_messages()` - Claude → Bedrock format
- `translator.create_claude_response()` - Bedrock → Claude format

### OpenAI-Compatible Backend (`backends/openai_compatible/`)
- `service.call_openai_compatible_chat(request)` - OpenAI API calls
- `service.count_openai_tokens(request)` - Token estimation
- `client.get_openai_compatible_client()` - HTTP session setup
- `translator.convert_to_openai_messages()` - Claude → OpenAI format
- `translator.create_claude_response_from_openai()` - OpenAI → Claude format

### Proxy Server (`proxy_server.py`)
- `create_message(request)` - `/v1/messages` endpoint
- `count_tokens(request)` - `/v1/messages/count_tokens` endpoint
- `health()` - `/health` endpoint

## Features

- ✅ Multi-turn conversations
- ✅ System messages  
- ✅ Tool calling
- ✅ Token counting
- ✅ Temperature/top_p/top_k
- ❌ Streaming (detected but not implemented)
- ❌ Image content