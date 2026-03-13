# mypy: disable-error-code=untyped-decorator
"""handlers for account info, credits, and activity."""

import logging
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from ..mcp_registry import get_openrouter_client, mcp

logger = logging.getLogger(__name__)


class GetActivityRequest(BaseModel):
    """request for getting user activity."""

    date: Optional[str] = Field(
        None,
        description="Filter by a single UTC date in the last 30 days (YYYY-MM-DD format)",
    )


@mcp.tool()
async def get_credits() -> Dict[str, Any]:
    """Get remaining credits for the current API key.

    Returns the current credit balance and usage limits for
    the authenticated API key.

    Returns:
        Dictionary with credit balance information
    """
    logger.info("Getting credits")
    client = await get_openrouter_client()

    try:
        result = await client.get_credits()
        logger.info(f"Retrieved credits info")
        return result
    except Exception as e:
        logger.error(f"Failed to get credits: {str(e)}")
        raise


@mcp.tool()
async def get_current_key() -> Dict[str, Any]:
    """Get information about the current API key.

    Returns metadata about the authenticated API key including
    its label, limits, creation date, and usage.

    Returns:
        Dictionary with API key metadata
    """
    logger.info("Getting current key info")
    client = await get_openrouter_client()

    try:
        result = await client.get_current_key()
        logger.info("Retrieved current key info")
        return result
    except Exception as e:
        logger.error(f"Failed to get current key: {str(e)}")
        raise


@mcp.tool()
async def get_activity(request: GetActivityRequest) -> Dict[str, Any]:
    """Get user activity grouped by endpoint.

    Returns API usage activity for the authenticated user, optionally
    filtered to a specific date within the last 30 days.

    Args:
        request: Request with optional date filter

    Returns:
        Dictionary with activity data grouped by endpoint
    """
    logger.info(f"Getting activity for date: {request.date or 'all'}")
    client = await get_openrouter_client()

    try:
        result = await client.get_activity(date=request.date)
        logger.info("Retrieved activity data")
        return result
    except Exception as e:
        logger.error(f"Failed to get activity: {str(e)}")
        raise
