"""
AWS Bedrock client configuration and management.

This module handles:
- AWS Bedrock Runtime client initialization
- AWS credential management (profiles, environment variables, IAM roles)
- Bedrock client configuration with retry policies
- Region and profile configuration
"""

import os
import boto3
from botocore.exceptions import ClientError, BotoCoreError
from botocore.config import Config
from fastapi import HTTPException


def get_bedrock_client():
    """Get configured AWS Bedrock Runtime client."""
    try:
        config = Config(
            region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            retries={"max_attempts": 3, "mode": "adaptive"}
        )
        
        # Use specified AWS profile or default to "saml"
        profile_name = os.environ.get("AWS_PROFILE", "saml")
        session = boto3.Session(profile_name=profile_name)
        
        # Get region from environment or use default
        region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        
        return session.client("bedrock-runtime", region_name=region, config=config)
        
    except (BotoCoreError, ClientError) as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to create bedrock client: {str(e)}"
        )


def get_model_id() -> str:
    """Get Bedrock model ID from environment variables."""
    return os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0")