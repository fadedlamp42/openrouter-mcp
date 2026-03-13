# mypy: disable-error-code=untyped-decorator
"""handler for OpenRouter's Responses API (OpenAI-compatible) and Messages API (Anthropic-compatible)."""

import logging
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from ..mcp_registry import get_openrouter_client, mcp

logger = logging.getLogger(__name__)


class CreateResponseRequest(BaseModel):
    """request for the Responses API."""

    model: str = Field(..., description="Model to use (e.g. 'openai/gpt-4o')")
    input: Union[str, List[Any]] = Field(
        ...,
        description="Input string or array of input items (messages, function call outputs, etc.)",
    )
    temperature: Optional[float] = Field(
        None, description="Sampling temperature (0.0 to 2.0)"
    )
    max_output_tokens: Optional[int] = Field(None, description="Maximum output tokens")
    instructions: Optional[str] = Field(
        None, description="System instructions for the model"
    )


class CreateMessageRequest(BaseModel):
    """request for the Anthropic Messages API."""

    model: str = Field(
        ..., description="Model to use (e.g. 'anthropic/claude-sonnet-4')"
    )
    messages: List[Dict[str, Any]] = Field(..., description="List of message objects")
    max_tokens: int = Field(1024, description="Maximum tokens to generate")
    system: Optional[str] = Field(None, description="System prompt")
    temperature: Optional[float] = Field(None, description="Sampling temperature")


@mcp.tool()
async def create_response(request: CreateResponseRequest) -> Dict[str, Any]:
    """Create a response using OpenRouter's Responses API.

    This is OpenRouter's implementation of the OpenAI Responses API,
    supporting structured input/output, tool use, web search, and
    reasoning output items.

    Args:
        request: Request with model, input, and optional parameters

    Returns:
        Response object with output items, usage, and metadata

    Example:
        request = CreateResponseRequest(
            model="openai/gpt-4o",
            input="What is the capital of France?"
        )
        result = await create_response(request)
    """
    logger.info(f"Creating response with model {request.model}")
    client = await get_openrouter_client()

    kwargs: Dict[str, Any] = {}
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.max_output_tokens is not None:
        kwargs["max_output_tokens"] = request.max_output_tokens
    if request.instructions is not None:
        kwargs["instructions"] = request.instructions

    try:
        result = await client.create_response(
            model=request.model,
            input_data=request.input,
            **kwargs,
        )
        logger.info(f"Created response with model {request.model}")
        return result
    except Exception as e:
        logger.error(f"Failed to create response: {str(e)}")
        raise


@mcp.tool()
async def create_message(request: CreateMessageRequest) -> Dict[str, Any]:
    """Create a message using the Anthropic Messages API format.

    This endpoint accepts Anthropic-style message requests and routes
    them through OpenRouter. Useful for applications built against
    the Anthropic API format.

    Args:
        request: Request with model, messages, and optional parameters

    Returns:
        Anthropic-format message response

    Example:
        request = CreateMessageRequest(
            model="anthropic/claude-sonnet-4",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100
        )
        result = await create_message(request)
    """
    logger.info(f"Creating message with model {request.model}")
    client = await get_openrouter_client()

    kwargs: Dict[str, Any] = {}
    if request.system is not None:
        kwargs["system"] = request.system
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature

    try:
        result = await client.create_message(
            model=request.model,
            messages=request.messages,
            max_tokens=request.max_tokens,
            **kwargs,
        )
        logger.info(f"Created message with model {request.model}")
        return result
    except Exception as e:
        logger.error(f"Failed to create message: {str(e)}")
        raise
