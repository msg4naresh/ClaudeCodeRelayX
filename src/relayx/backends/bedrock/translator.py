"""
Message translation between Claude API and AWS Bedrock Converse API formats.

This module handles the core translation logic:
- Converting Claude API message format to Bedrock Converse format
- Extracting and processing system messages  
- Handling different content block types (text, tool_result, etc.)
- Token counting and estimation utilities
"""

import uuid
from typing import List, Dict, Any, Optional, Union
from ...models import (
    MessagesRequest, MessagesResponse, Message, SystemContent, 
    ContentBlockText, ContentBlockToolUse, Usage
)


def extract_system_message(request: MessagesRequest) -> Optional[str]:
    """Extract system message from request."""
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


def convert_to_bedrock_messages(request: MessagesRequest) -> List[Dict[str, Any]]:
    """Convert Claude messages to Bedrock format."""
    bedrock_messages = []
    
    for msg in request.messages:
        if msg.role == "system":
            continue  # Handle separately in system parameter
            
        # Handle content conversion
        if isinstance(msg.content, str):
            # Simple string content
            bedrock_messages.append({
                "role": msg.role,
                "content": [{"text": msg.content}]
            })
        else:
            # Handle content blocks
            content_parts = []
            for block in msg.content:
                if hasattr(block, 'type'):
                    if block.type == "text":
                        content_parts.append({"text": block.text})
                    elif block.type == "tool_use":
                        # Convert tool use to Bedrock format
                        content_parts.append({
                            "toolUse": {
                                "toolUseId": block.id,
                                "name": block.name,
                                "input": block.input
                            }
                        })
                    elif block.type == "tool_result":
                        # Convert tool result to Bedrock format
                        tool_result_content = []
                        
                        # Handle different content types
                        if isinstance(block.content, str):
                            # Simple text content
                            tool_result_content.append({"text": block.content})
                        elif isinstance(block.content, (dict, list)):
                            # Structured data as JSON
                            tool_result_content.append({"json": block.content})
                        else:
                            # Fallback to text representation
                            tool_result_content.append({"text": str(block.content)})
                        
                        content_parts.append({
                            "toolResult": {
                                "toolUseId": block.tool_use_id,
                                "content": tool_result_content
                            }
                        })
                    # Note: Bedrock Converse doesn't support images yet, would need different handling
            
            if content_parts:
                bedrock_messages.append({
                    "role": msg.role,
                    "content": content_parts
                })
    
    return bedrock_messages


def estimate_token_count(text: str) -> int:
    """Estimate token count using character heuristics."""
    return max(1, len(text) // 4)


def count_tokens_from_messages(messages: List[Message], system: Optional[Union[str, List[SystemContent]]] = None) -> int:
    """Count tokens from messages for Bedrock models."""
    total_tokens = 0
    
    # Count system tokens
    if system:
        if isinstance(system, str):
            total_tokens += estimate_token_count(system)
        elif isinstance(system, list):
            for block in system:
                if hasattr(block, 'text'):
                    total_tokens += estimate_token_count(block.text)
    
    # Count message tokens
    for msg in messages:
        if isinstance(msg.content, str):
            total_tokens += estimate_token_count(msg.content)
        else:
            for block in msg.content:
                if hasattr(block, 'type') and block.type == "text":
                    total_tokens += estimate_token_count(block.text)
                elif hasattr(block, 'type') and block.type == "tool_result":
                    content_str = str(block.content) if not isinstance(block.content, str) else block.content
                    total_tokens += estimate_token_count(content_str)
    
    return total_tokens


def create_claude_response(bedrock_response: Dict[str, Any], model_id: str) -> MessagesResponse:
    """Convert Bedrock response to Claude format."""
    # Extract response content blocks
    content_blocks = []
    bedrock_content = bedrock_response['output']['message']['content']
    
    for block in bedrock_content:
        if 'text' in block:
            # Text content block
            content_blocks.append(ContentBlockText(
                type="text", 
                text=block['text']
            ))
        elif 'toolUse' in block:
            # Tool use content block
            tool_use = block['toolUse']
            content_blocks.append(ContentBlockToolUse(
                type="tool_use",
                id=tool_use['toolUseId'],
                name=tool_use['name'],
                input=tool_use['input']
            ))
    
    # Ensure we have at least one content block
    if not content_blocks:
        content_blocks.append(ContentBlockText(type="text", text=""))
    
    usage_info = bedrock_response.get('usage', {})
    
    # Determine stop reason
    stop_reason = bedrock_response['output'].get('stopReason', 'end_turn')
    if stop_reason == 'tool_use':
        stop_reason = 'tool_use'
    elif stop_reason == 'stop_sequence':
        stop_reason = 'stop_sequence'
    elif stop_reason == 'max_tokens':
        stop_reason = 'max_tokens'
    else:
        stop_reason = 'end_turn'
    
    return MessagesResponse(
        id=f"msg_{uuid.uuid4().hex[:24]}",
        model=model_id,
        content=content_blocks,
        stop_reason=stop_reason,
        usage=Usage(
            input_tokens=usage_info.get('inputTokens', 0),
            output_tokens=usage_info.get('outputTokens', 0)
        )
    )