# mypy: disable-error-code=untyped-decorator
"""handlers for authentication and payment endpoints."""

import logging
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from ..mcp_registry import get_openrouter_client, mcp

logger = logging.getLogger(__name__)


class ExchangeAuthCodeRequest(BaseModel):
    """request for exchanging an auth code for an API key."""

    code: str = Field(..., description="Authorization code to exchange")
    code_verifier: Optional[str] = Field(None, description="PKCE code verifier")
    code_challenge_method: Optional[str] = Field(
        None, description="Code challenge method (e.g. 'S256')"
    )


class CreateAuthCodeRequest(BaseModel):
    """request for creating an authorization code."""

    callback_url: str = Field(..., description="OAuth callback URL")
    code_challenge: Optional[str] = Field(None, description="PKCE code challenge")
    code_challenge_method: Optional[str] = Field(
        None, description="Code challenge method (e.g. 'S256')"
    )
    limit: Optional[float] = Field(
        None, description="Spending limit for the resulting key"
    )
    expires_at: Optional[str] = Field(
        None, description="Expiration for the resulting key (ISO 8601)"
    )
    key_label: Optional[str] = Field(None, description="Label for the resulting key")
    usage_limit_type: Optional[str] = Field(None, description="Type of usage limit")


class CreateCoinbaseChargeRequest(BaseModel):
    """request for creating a Coinbase Commerce charge."""

    amount: float = Field(..., description="Amount in USD to charge")
    sender: Optional[str] = Field(None, description="Sender identifier")


@mcp.tool()
async def exchange_auth_code(request: ExchangeAuthCodeRequest) -> Dict[str, Any]:
    """Exchange an authorization code for an API key.

    Used in the OAuth PKCE flow to convert a temporary auth code
    into a permanent API key.

    Args:
        request: Request with authorization code and optional PKCE params

    Returns:
        Dictionary with the generated API key
    """
    logger.info("Exchanging auth code for API key")
    client = await get_openrouter_client()

    try:
        result = await client.exchange_auth_code(
            code=request.code,
            code_verifier=request.code_verifier,
            code_challenge_method=request.code_challenge_method,
        )
        logger.info("Exchanged auth code successfully")
        return result
    except Exception as e:
        logger.error(f"Failed to exchange auth code: {str(e)}")
        raise


@mcp.tool()
async def create_auth_code(request: CreateAuthCodeRequest) -> Dict[str, Any]:
    """Create an authorization code for the OAuth PKCE flow.

    Generates a temporary code that can be exchanged for an API key
    via the callback URL.

    Args:
        request: Request with callback URL and optional PKCE params

    Returns:
        Dictionary with the authorization code and redirect URL
    """
    logger.info(f"Creating auth code with callback: {request.callback_url}")
    client = await get_openrouter_client()

    try:
        result = await client.create_auth_code(
            callback_url=request.callback_url,
            code_challenge=request.code_challenge,
            code_challenge_method=request.code_challenge_method,
            limit=request.limit,
            expires_at=request.expires_at,
            key_label=request.key_label,
            usage_limit_type=request.usage_limit_type,
        )
        logger.info("Created auth code successfully")
        return result
    except Exception as e:
        logger.error(f"Failed to create auth code: {str(e)}")
        raise


@mcp.tool()
async def create_coinbase_charge(
    request: CreateCoinbaseChargeRequest,
) -> Dict[str, Any]:
    """Create a Coinbase Commerce charge for adding credits.

    Generates a crypto payment charge that, when completed, adds
    credits to the OpenRouter account.

    Args:
        request: Request with amount in USD

    Returns:
        Dictionary with Coinbase charge details and payment URL
    """
    logger.info(f"Creating Coinbase charge for ${request.amount}")
    client = await get_openrouter_client()

    try:
        result = await client.create_coinbase_charge(
            amount=request.amount,
            sender=request.sender,
        )
        logger.info(f"Created Coinbase charge for ${request.amount}")
        return result
    except Exception as e:
        logger.error(f"Failed to create Coinbase charge: {str(e)}")
        raise
