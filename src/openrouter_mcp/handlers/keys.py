# mypy: disable-error-code=untyped-decorator
"""handlers for API key management (list, create, get, update, delete)."""

import logging
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from ..mcp_registry import get_openrouter_client, mcp

logger = logging.getLogger(__name__)


class ListApiKeysRequest(BaseModel):
    """request for listing API keys."""

    include_disabled: Optional[bool] = Field(None, description="Include disabled keys")
    offset: Optional[int] = Field(None, description="Pagination offset")


class CreateApiKeyRequest(BaseModel):
    """request for creating an API key."""

    name: str = Field(..., description="Name for the new API key")
    limit: Optional[float] = Field(None, description="Spending limit in USD")
    limit_reset: Optional[str] = Field(None, description="Limit reset interval")
    include_byok_in_limit: Optional[bool] = Field(
        None, description="Whether BYOK usage counts toward the limit"
    )
    expires_at: Optional[str] = Field(None, description="Expiration date (ISO 8601)")


class GetApiKeyRequest(BaseModel):
    """request for getting a single API key."""

    hash: str = Field(..., description="API key hash identifier")


class UpdateApiKeyRequest(BaseModel):
    """request for updating an API key."""

    hash: str = Field(..., description="API key hash identifier")
    name: Optional[str] = Field(None, description="New name")
    disabled: Optional[bool] = Field(None, description="Whether to disable the key")
    limit: Optional[float] = Field(None, description="New spending limit in USD")
    limit_reset: Optional[str] = Field(None, description="Limit reset interval")
    include_byok_in_limit: Optional[bool] = Field(
        None, description="Whether BYOK usage counts toward the limit"
    )


class DeleteApiKeyRequest(BaseModel):
    """request for deleting an API key."""

    hash: str = Field(..., description="API key hash identifier")


@mcp.tool()
async def list_api_keys(request: ListApiKeysRequest) -> Dict[str, Any]:
    """List API keys for the current account.

    Returns all API keys associated with the account, optionally
    including disabled keys.

    Args:
        request: Request with optional filters

    Returns:
        Dictionary with list of API keys and their metadata
    """
    logger.info("Listing API keys")
    client = await get_openrouter_client()

    try:
        result = await client.list_api_keys(
            include_disabled=request.include_disabled,
            offset=request.offset,
        )
        logger.info("Retrieved API keys")
        return result
    except Exception as e:
        logger.error(f"Failed to list API keys: {str(e)}")
        raise


@mcp.tool()
async def create_api_key(request: CreateApiKeyRequest) -> Dict[str, Any]:
    """Create a new API key.

    Creates a new API key with the specified name and optional limits.

    Args:
        request: Request with key name and optional limits

    Returns:
        Dictionary with the newly created API key details
    """
    logger.info(f"Creating API key: {request.name}")
    client = await get_openrouter_client()

    try:
        result = await client.create_api_key(
            name=request.name,
            limit=request.limit,
            limit_reset=request.limit_reset,
            include_byok_in_limit=request.include_byok_in_limit,
            expires_at=request.expires_at,
        )
        logger.info(f"Created API key: {request.name}")
        return result
    except Exception as e:
        logger.error(f"Failed to create API key: {str(e)}")
        raise


@mcp.tool()
async def get_api_key(request: GetApiKeyRequest) -> Dict[str, Any]:
    """Get a single API key by its hash.

    Args:
        request: Request with the key hash

    Returns:
        Dictionary with the API key details
    """
    logger.info(f"Getting API key: {request.hash}")
    client = await get_openrouter_client()

    try:
        result = await client.get_api_key(request.hash)
        logger.info(f"Retrieved API key: {request.hash}")
        return result
    except Exception as e:
        logger.error(f"Failed to get API key: {str(e)}")
        raise


@mcp.tool()
async def update_api_key(request: UpdateApiKeyRequest) -> Dict[str, Any]:
    """Update an existing API key.

    Modify the name, limits, or disabled status of an API key.

    Args:
        request: Request with key hash and fields to update

    Returns:
        Dictionary with the updated API key details
    """
    logger.info(f"Updating API key: {request.hash}")
    client = await get_openrouter_client()

    try:
        result = await client.update_api_key(
            key_hash=request.hash,
            name=request.name,
            disabled=request.disabled,
            limit=request.limit,
            limit_reset=request.limit_reset,
            include_byok_in_limit=request.include_byok_in_limit,
        )
        logger.info(f"Updated API key: {request.hash}")
        return result
    except Exception as e:
        logger.error(f"Failed to update API key: {str(e)}")
        raise


@mcp.tool()
async def delete_api_key(request: DeleteApiKeyRequest) -> Dict[str, Any]:
    """Delete an API key.

    Permanently deletes the specified API key.

    Args:
        request: Request with the key hash

    Returns:
        Confirmation of deletion
    """
    logger.info(f"Deleting API key: {request.hash}")
    client = await get_openrouter_client()

    try:
        result = await client.delete_api_key(request.hash)
        logger.info(f"Deleted API key: {request.hash}")
        return result
    except Exception as e:
        logger.error(f"Failed to delete API key: {str(e)}")
        raise
