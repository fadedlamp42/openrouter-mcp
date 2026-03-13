# mypy: disable-error-code=untyped-decorator
"""handlers for model discovery, providers, and endpoints."""

import logging
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from ..mcp_registry import get_openrouter_client, mcp

logger = logging.getLogger(__name__)


class ListModelEndpointsRequest(BaseModel):
    """request for listing endpoints for a specific model."""

    author: str = Field(..., description="Model author (e.g. 'openai', 'anthropic')")
    slug: str = Field(..., description="Model slug (e.g. 'gpt-4', 'claude-3-haiku')")


class ListUserModelsRequest(BaseModel):
    """request for listing user-filtered models."""

    pass


@mcp.tool()
async def list_model_endpoints(
    request: ListModelEndpointsRequest,
) -> Union[Dict[str, Any], List[Any]]:
    """List all provider endpoints serving a specific model.

    Returns which providers serve the model, their pricing, quantization
    levels, context lengths, and availability status.

    Args:
        request: Request with model author and slug

    Returns:
        Endpoint data including provider names, pricing, and capabilities

    Example:
        request = ListModelEndpointsRequest(author="openai", slug="gpt-4")
        endpoints = await list_model_endpoints(request)
    """
    logger.info(f"Listing endpoints for {request.author}/{request.slug}")
    client = await get_openrouter_client()

    try:
        result = await client.list_model_endpoints(request.author, request.slug)
        logger.info(f"Retrieved endpoints for {request.author}/{request.slug}")
        return result
    except Exception as e:
        logger.error(f"Failed to list model endpoints: {str(e)}")
        raise


@mcp.tool()
async def list_providers() -> Union[Dict[str, Any], List[Any]]:
    """List all available providers on OpenRouter.

    Returns information about all providers including their names,
    supported models, and capabilities.

    Returns:
        List or dict of provider information
    """
    logger.info("Listing all providers")
    client = await get_openrouter_client()

    try:
        result = await client.list_providers()
        logger.info("Retrieved providers list")
        return result
    except Exception as e:
        logger.error(f"Failed to list providers: {str(e)}")
        raise


@mcp.tool()
async def get_models_count() -> Dict[str, Any]:
    """Get the total count of available models on OpenRouter.

    Returns:
        Dictionary with the total model count
    """
    logger.info("Getting models count")
    client = await get_openrouter_client()

    try:
        result = await client.get_models_count()
        logger.info(f"Models count: {result}")
        return result
    except Exception as e:
        logger.error(f"Failed to get models count: {str(e)}")
        raise


@mcp.tool()
async def list_user_models() -> Dict[str, Any]:
    """List models filtered by user provider preferences, privacy settings, and guardrails.

    Returns models available to the current user after applying their
    configured provider preferences and organization guardrails.

    Returns:
        Dictionary with filtered model list
    """
    logger.info("Listing user-filtered models")
    client = await get_openrouter_client()

    try:
        result = await client.list_user_models()
        logger.info("Retrieved user-filtered models")
        return result
    except Exception as e:
        logger.error(f"Failed to list user models: {str(e)}")
        raise


@mcp.tool()
async def list_zdr_endpoints() -> Union[Dict[str, Any], List[Any]]:
    """Preview the impact of Zero Data Retention (ZDR) on available endpoints.

    Shows which endpoints support ZDR and how enabling it would affect
    model availability and routing.

    Returns:
        ZDR endpoint preview data
    """
    logger.info("Listing ZDR endpoints")
    client = await get_openrouter_client()

    try:
        result = await client.list_zdr_endpoints()
        logger.info("Retrieved ZDR endpoints")
        return result
    except Exception as e:
        logger.error(f"Failed to list ZDR endpoints: {str(e)}")
        raise
