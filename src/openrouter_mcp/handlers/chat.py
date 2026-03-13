# mypy: disable-error-code=untyped-decorator

import logging
from typing import Any, Dict, List, Optional, Union, cast

from pydantic import BaseModel, Field

# Import shared MCP instance and client manager from registry
from ..mcp_registry import get_openrouter_client, mcp

# Import centralized configuration constants
from ..models.requests import BaseChatRequest
from ..utils.async_utils import collect_async_iterable
from ..utils.message_utils import serialize_messages

logger = logging.getLogger(__name__)


class ChatCompletionRequest(BaseChatRequest):
    """Request for chat completion."""

    pass


class ModelListRequest(BaseModel):
    """Request for listing available models."""

    filter_by: Optional[str] = Field(
        None, description="Filter models by name substring"
    )


class GetGenerationRequest(BaseModel):
    """Request for getting generation metadata."""

    id: str = Field(
        ..., description="Generation ID returned in chat completion responses"
    )


@mcp.tool()
async def get_generation(request: GetGenerationRequest) -> Dict[str, Any]:
    """Get request and usage metadata for a generation.

    Look up cost, token counts, model, latency, and other metadata for a
    past chat completion by its generation ID.

    Args:
        request: Request containing the generation ID

    Returns:
        Dictionary with generation metadata including tokens_prompt,
        tokens_completion, total_cost, model, provider, latency, etc.

    Raises:
        OpenRouterError: If the API request fails

    Example:
        request = GetGenerationRequest(id="gen-abc123...")
        stats = await get_generation(request)
    """
    logger.info(f"Getting generation metadata for {request.id}")
    client = await get_openrouter_client()

    try:
        result = await client.get_generation(request.id)
        logger.info(f"Retrieved generation metadata for {request.id}")
        return result
    except Exception as e:
        logger.error(f"Failed to get generation: {str(e)}")
        raise


@mcp.tool()
async def list_available_models(request: ModelListRequest) -> List[Dict[str, Any]]:
    """
    List all available models from OpenRouter.

    This tool retrieves information about all AI models available through OpenRouter,
    including their pricing, capabilities, and context limits. You can optionally
    filter the results by model name.

    Args:
        request: Model list request with optional filter

    Returns:
        List of dictionaries containing model information:
        - id: Model identifier (e.g., "openai/gpt-4")
        - name: Human-readable model name
        - description: Model description
        - pricing: Cost per token for prompts and completions
        - context_length: Maximum context window size
        - architecture: Model architecture details

    Raises:
        OpenRouterError: If the API request fails

    Example:
        request = ModelListRequest(filter_by="gpt")
        models = await list_available_models(request)
    """
    logger.info(f"Listing models with filter: {request.filter_by or 'none'}")

    # Get shared client (already in async context, no need for 'async with')
    client = await get_openrouter_client()

    try:
        models = await client.list_models(filter_by=request.filter_by)
        models = [model for model in models if isinstance(model, dict)]
        logger.info(f"Retrieved {len(models)} models")
        return models

    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}")
        raise


@mcp.tool()
async def get_usage_stats(request: GetGenerationRequest) -> Dict[str, Any]:
    """Get request and usage metadata for a generation.

    Look up cost, token counts, model, latency, and other metadata for a
    past chat completion by its generation ID.

    Args:
        request: Request containing the generation ID

    Returns:
        Dictionary with generation metadata including tokens_prompt,
        tokens_completion, total_cost, model, provider, latency, etc.

    Example:
        request = GetGenerationRequest(id="gen-abc123...")
        stats = await get_usage_stats(request)
    """
    logger.info(f"Getting generation metadata for {request.id}")
    client = await get_openrouter_client()

    try:
        result = await client.get_generation(request.id)
        logger.info(f"Retrieved generation metadata for {request.id}")
        return result
    except Exception as e:
        logger.error(f"Failed to get generation: {str(e)}")
        raise
