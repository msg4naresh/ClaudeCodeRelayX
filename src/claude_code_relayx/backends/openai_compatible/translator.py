"""
Message translation between Claude API and OpenAI-compatible API formats.

This module handles the core translation logic for any provider that implements
the OpenAI Chat Completions API standard, including:
- OpenAI (api.openai.com)
- Google Gemini (generativelanguage.googleapis.com) 
- Azure OpenAI
- Local models (Ollama, LM Studio, etc.)
- Other OpenAI-compatible providers

Features:
- Converting Claude API message format to OpenAI chat completions format
- Extracting and processing system messages for OpenAI-compatible APIs
- Handling different content block types (text, tool_use, tool_result, etc.)
- Token counting and estimation utilities for various models
"""

import uuid
from typing import List, Dict, Any, Optional, Union
from ...models import (
    MessagesRequest, MessagesResponse, Message, SystemContent, 
    ContentBlockText, ContentBlockToolUse, Usage
)


def extract_system_message_openai(request: MessagesRequest) -> Optional[str]:
    """Extract system message for OpenAI format."""
    # First check system field
    if request.system:
        if isinstance(request.system, str):
            return request.system
        elif isinstance(request.system, list):
            system_text = ""
            for block in request.system:
                if hasattr(block, 'text'):
                    system_text += block.text + "\n"
            return system_text.strip()
    
    # Also check for system messages in messages list
    for msg in request.messages:
        if msg.role == "system" and isinstance(msg.content, str):
            return msg.content
    
    return None


def convert_to_openai_messages(request: MessagesRequest) -> List[Dict[str, Any]]:
    """Convert Claude messages to OpenAI format."""
    openai_messages = []
    
    # Add system message first if present
    system_message = extract_system_message_openai(request)
    if system_message:
        openai_messages.append({
            "role": "system",
            "content": system_message
        })
    
    for msg in request.messages:
        if msg.role == "system":
            continue  # Already handled above
            
        # Handle content conversion
        if isinstance(msg.content, str):
            # Simple string content
            openai_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        else:
            # Handle content blocks
            if len(msg.content) == 1 and hasattr(msg.content[0], 'type') and msg.content[0].type == "text":
                # Single text block - simplify to string
                openai_messages.append({
                    "role": msg.role,
                    "content": msg.content[0].text
                })
            else:
                # Multiple blocks or special content types
                content_parts = []
                tool_calls = []
                
                for block in msg.content:
                    if hasattr(block, 'type'):
                        if block.type == "text":
                            content_parts.append(block.text)
                        elif block.type == "tool_use":
                            # Convert tool use to OpenAI format
                            tool_calls.append({
                                "id": block.id,
                                "type": "function",
                                "function": {
                                    "name": block.name,
                                    "arguments": str(block.input) if isinstance(block.input, dict) else block.input
                                }
                            })
                        elif block.type == "tool_result":
                            # Tool results become separate user messages in OpenAI
                            if isinstance(block.content, str):
                                content_text = block.content
                            else:
                                content_text = str(block.content)
                            
                            openai_messages.append({
                                "role": "tool",
                                "content": content_text,
                                "tool_call_id": block.tool_use_id
                            })
                
                # Create message with text content and/or tool calls
                message = {"role": msg.role}
                
                if content_parts:
                    message["content"] = " ".join(content_parts)
                elif not tool_calls:
                    message["content"] = ""
                
                if tool_calls:
                    message["tool_calls"] = tool_calls
                
                if "content" in message or "tool_calls" in message:
                    openai_messages.append(message)
    
    return openai_messages


def convert_tools_to_openai(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Claude tools to OpenAI format."""
    openai_tools = []
    
    for tool in tools:
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema
            }
        }
        openai_tools.append(openai_tool)
    
    return openai_tools


def estimate_token_count_openai(text: str) -> int:
    """Estimate token count using character heuristics."""
    return max(1, len(text) // 4)


def count_tokens_from_messages_openai(messages: List[Message], system: Optional[Union[str, List[SystemContent]]] = None) -> int:
    """Count tokens from messages for OpenAI models."""
    total_tokens = 0
    
    # Count system tokens
    if system:
        if isinstance(system, str):
            total_tokens += estimate_token_count_openai(system)
        elif isinstance(system, list):
            for block in system:
                if hasattr(block, 'text'):
                    total_tokens += estimate_token_count_openai(block.text)
    
    # Count message tokens
    for msg in messages:
        if isinstance(msg.content, str):
            total_tokens += estimate_token_count_openai(msg.content)
        else:
            for block in msg.content:
                if hasattr(block, 'type') and block.type == "text":
                    total_tokens += estimate_token_count_openai(block.text)
                elif hasattr(block, 'type') and block.type == "tool_result":
                    content_str = str(block.content) if not isinstance(block.content, str) else block.content
                    total_tokens += estimate_token_count_openai(content_str)
    
    return total_tokens


def create_claude_response_from_openai(openai_response: Dict[str, Any], model_id: str) -> MessagesResponse:
    """Convert OpenAI response to Claude format."""
    # Extract first choice (OpenAI can return multiple choices)
    if not openai_response.get('choices'):
        raise ValueError("No choices in OpenAI response")
    
    choice = openai_response['choices'][0]
    message = choice['message']
    
    # Extract content blocks
    content_blocks = []
    
    # Handle text content
    if message.get('content'):
        content_blocks.append(ContentBlockText(
            type="text", 
            text=message['content']
        ))
    
    # Handle tool calls
    if message.get('tool_calls'):
        for tool_call in message['tool_calls']:
            if tool_call['type'] == 'function':
                function = tool_call['function']
                # Parse arguments back to dict if it's a string
                try:
                    import json
                    arguments = json.loads(function['arguments']) if isinstance(function['arguments'], str) else function['arguments']
                except (json.JSONDecodeError, ValueError):
                    arguments = function['arguments']
                
                content_blocks.append(ContentBlockToolUse(
                    type="tool_use",
                    id=tool_call['id'],
                    name=function['name'],
                    input=arguments
                ))
    
    # Ensure we have at least one content block
    if not content_blocks:
        content_blocks.append(ContentBlockText(type="text", text=""))
    
    # Map stop reason
    finish_reason = choice.get('finish_reason', 'stop')
    if finish_reason == 'tool_calls':
        stop_reason = 'tool_use'
    elif finish_reason == 'length':
        stop_reason = 'max_tokens'
    elif finish_reason == 'content_filter':
        stop_reason = 'stop_sequence'
    else:  # 'stop' or other
        stop_reason = 'end_turn'
    
    # Extract usage information
    usage_info = openai_response.get('usage', {})
    
    return MessagesResponse(
        id=f"msg_{uuid.uuid4().hex[:24]}",
        model=model_id,
        content=content_blocks,
        stop_reason=stop_reason,
        usage=Usage(
            input_tokens=usage_info.get('prompt_tokens', 0),
            output_tokens=usage_info.get('completion_tokens', 0)
        )
    )