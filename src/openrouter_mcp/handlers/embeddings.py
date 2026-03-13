# mypy: disable-error-code=untyped-decorator
"""handlers for embeddings API."""

import logging
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from ..mcp_registry import get_openrouter_client, mcp

logger = logging.getLogger(__name__)


class CreateEmbeddingRequest(BaseModel):
    """request for creating embeddings."""

    model: str = Field(
        ..., description="Embedding model to use (e.g. 'openai/text-embedding-3-small')"
    )
    input: Union[str, List[str]] = Field(
        ..., description="Text or list of texts to embed"
    )
    encoding_format: Optional[str] = Field(
        None, description="Output format: 'float' or 'base64'"
    )
    dimensions: Optional[int] = Field(
        None, description="Number of dimensions for the output embedding"
    )
    user: Optional[str] = Field(
        None, description="Unique end-user identifier for abuse detection"
    )


@mcp.tool()
async def create_embedding(request: CreateEmbeddingRequest) -> Dict[str, Any]:
    """Create embeddings for the given input text.

    Generate vector embeddings from text using the specified embedding model.
    Supports single strings or batches of strings.

    Args:
        request: Request with model, input text, and optional parameters

    Returns:
        Dictionary with embedding vectors, model, and usage info

    Example:
        request = CreateEmbeddingRequest(
            model="openai/text-embedding-3-small",
            input="Hello world"
        )
        result = await create_embedding(request)
    """
    logger.info(f"Creating embedding with model {request.model}")
    client = await get_openrouter_client()

    try:
        result = await client.create_embedding(
            model=request.model,
            input_text=request.input,
            encoding_format=request.encoding_format,
            dimensions=request.dimensions,
            user=request.user,
        )
        logger.info(f"Created embedding with model {request.model}")
        return result
    except Exception as e:
        logger.error(f"Failed to create embedding: {str(e)}")
        raise


@mcp.tool()
async def list_embedding_models() -> Union[Dict[str, Any], List[Any]]:
    """List all available embedding models on OpenRouter.

    Returns information about embedding models including their
    dimensions, pricing, and supported features.

    Returns:
        List or dict of embedding model information
    """
    logger.info("Listing embedding models")
    client = await get_openrouter_client()

    try:
        result = await client.list_embedding_models()
        logger.info("Retrieved embedding models")
        return result
    except Exception as e:
        logger.error(f"Failed to list embedding models: {str(e)}")
        raise
