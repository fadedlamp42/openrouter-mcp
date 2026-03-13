# mypy: disable-error-code=untyped-decorator
"""handlers for guardrails CRUD and assignment management."""

import logging
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from ..mcp_registry import get_openrouter_client, mcp

logger = logging.getLogger(__name__)


# ── request models ──


class ListGuardrailsRequest(BaseModel):
    offset: Optional[int] = Field(None, description="Pagination offset")
    limit: Optional[int] = Field(None, description="Maximum results to return")


class CreateGuardrailRequest(BaseModel):
    name: str = Field(..., description="Guardrail name")
    description: Optional[str] = Field(None, description="Description of the guardrail")
    limit_usd: Optional[float] = Field(None, description="Spending limit in USD")
    reset_interval: Optional[str] = Field(None, description="Limit reset interval")
    allowed_providers: Optional[List[str]] = Field(
        None, description="Allowed provider IDs"
    )
    allowed_models: Optional[List[str]] = Field(None, description="Allowed model IDs")
    enforce_zdr: Optional[bool] = Field(None, description="Enforce Zero Data Retention")


class GuardrailIdRequest(BaseModel):
    id: str = Field(..., description="Guardrail UUID")


class UpdateGuardrailRequest(BaseModel):
    id: str = Field(..., description="Guardrail UUID")
    name: Optional[str] = Field(None, description="New name")
    description: Optional[str] = Field(None, description="New description")
    limit_usd: Optional[float] = Field(None, description="New spending limit in USD")
    reset_interval: Optional[str] = Field(None, description="Limit reset interval")
    allowed_providers: Optional[List[str]] = Field(
        None, description="Allowed provider IDs"
    )
    allowed_models: Optional[List[str]] = Field(None, description="Allowed model IDs")
    enforce_zdr: Optional[bool] = Field(None, description="Enforce Zero Data Retention")


class GuardrailAssignKeysRequest(BaseModel):
    id: str = Field(..., description="Guardrail UUID")
    key_hashes: List[str] = Field(..., description="List of API key hashes to assign")


class GuardrailAssignMembersRequest(BaseModel):
    id: str = Field(..., description="Guardrail UUID")
    member_user_ids: List[str] = Field(
        ..., description="List of member user IDs to assign"
    )


class GuardrailListAssignmentsRequest(BaseModel):
    id: str = Field(..., description="Guardrail UUID")
    offset: Optional[int] = Field(None, description="Pagination offset")
    limit: Optional[int] = Field(None, description="Maximum results to return")


class ListAllAssignmentsRequest(BaseModel):
    offset: Optional[int] = Field(None, description="Pagination offset")
    limit: Optional[int] = Field(None, description="Maximum results to return")


# ── guardrail CRUD ──


@mcp.tool()
async def list_guardrails(request: ListGuardrailsRequest) -> Dict[str, Any]:
    """List guardrails for the current organization.

    Returns:
        Dictionary with guardrail list and pagination info
    """
    client = await get_openrouter_client()
    return await client.list_guardrails(offset=request.offset, limit=request.limit)


@mcp.tool()
async def create_guardrail(request: CreateGuardrailRequest) -> Dict[str, Any]:
    """Create a new guardrail.

    Guardrails let you set spending limits, restrict providers/models,
    and enforce Zero Data Retention for API keys and organization members.

    Args:
        request: Guardrail configuration

    Returns:
        The created guardrail
    """
    kwargs: Dict[str, Any] = {}
    if request.description is not None:
        kwargs["description"] = request.description
    if request.limit_usd is not None:
        kwargs["limit_usd"] = request.limit_usd
    if request.reset_interval is not None:
        kwargs["reset_interval"] = request.reset_interval
    if request.allowed_providers is not None:
        kwargs["allowed_providers"] = request.allowed_providers
    if request.allowed_models is not None:
        kwargs["allowed_models"] = request.allowed_models
    if request.enforce_zdr is not None:
        kwargs["enforce_zdr"] = request.enforce_zdr

    client = await get_openrouter_client()
    return await client.create_guardrail(name=request.name, **kwargs)


@mcp.tool()
async def get_guardrail(request: GuardrailIdRequest) -> Dict[str, Any]:
    """Get a guardrail by ID.

    Args:
        request: Request with guardrail UUID

    Returns:
        Guardrail details
    """
    client = await get_openrouter_client()
    return await client.get_guardrail(request.id)


@mcp.tool()
async def update_guardrail(request: UpdateGuardrailRequest) -> Dict[str, Any]:
    """Update an existing guardrail.

    Args:
        request: Request with guardrail UUID and fields to update

    Returns:
        The updated guardrail
    """
    kwargs: Dict[str, Any] = {}
    for field_name in (
        "name",
        "description",
        "limit_usd",
        "reset_interval",
        "allowed_providers",
        "allowed_models",
        "enforce_zdr",
    ):
        value = getattr(request, field_name)
        if value is not None:
            kwargs[field_name] = value

    client = await get_openrouter_client()
    return await client.update_guardrail(request.id, **kwargs)


@mcp.tool()
async def delete_guardrail(request: GuardrailIdRequest) -> Dict[str, Any]:
    """Delete a guardrail.

    Args:
        request: Request with guardrail UUID

    Returns:
        Confirmation of deletion
    """
    client = await get_openrouter_client()
    return await client.delete_guardrail(request.id)


# ── key assignments ──


@mcp.tool()
async def list_guardrail_key_assignments(
    request: GuardrailListAssignmentsRequest,
) -> Dict[str, Any]:
    """List API key assignments for a specific guardrail.

    Args:
        request: Request with guardrail UUID and optional pagination

    Returns:
        List of assigned API key hashes
    """
    client = await get_openrouter_client()
    return await client.list_guardrail_key_assignments(
        request.id, offset=request.offset, limit=request.limit
    )


@mcp.tool()
async def assign_keys_to_guardrail(
    request: GuardrailAssignKeysRequest,
) -> Dict[str, Any]:
    """Bulk assign API keys to a guardrail.

    Args:
        request: Request with guardrail UUID and list of key hashes

    Returns:
        Assignment result
    """
    client = await get_openrouter_client()
    return await client.assign_keys_to_guardrail(request.id, request.key_hashes)


@mcp.tool()
async def unassign_keys_from_guardrail(
    request: GuardrailAssignKeysRequest,
) -> Dict[str, Any]:
    """Bulk unassign API keys from a guardrail.

    Args:
        request: Request with guardrail UUID and list of key hashes

    Returns:
        Unassignment result
    """
    client = await get_openrouter_client()
    return await client.unassign_keys_from_guardrail(request.id, request.key_hashes)


# ── member assignments ──


@mcp.tool()
async def list_guardrail_member_assignments(
    request: GuardrailListAssignmentsRequest,
) -> Dict[str, Any]:
    """List member assignments for a specific guardrail.

    Args:
        request: Request with guardrail UUID and optional pagination

    Returns:
        List of assigned member user IDs
    """
    client = await get_openrouter_client()
    return await client.list_guardrail_member_assignments(
        request.id, offset=request.offset, limit=request.limit
    )


@mcp.tool()
async def assign_members_to_guardrail(
    request: GuardrailAssignMembersRequest,
) -> Dict[str, Any]:
    """Bulk assign members to a guardrail.

    Args:
        request: Request with guardrail UUID and list of member user IDs

    Returns:
        Assignment result
    """
    client = await get_openrouter_client()
    return await client.assign_members_to_guardrail(request.id, request.member_user_ids)


@mcp.tool()
async def unassign_members_from_guardrail(
    request: GuardrailAssignMembersRequest,
) -> Dict[str, Any]:
    """Bulk unassign members from a guardrail.

    Args:
        request: Request with guardrail UUID and list of member user IDs

    Returns:
        Unassignment result
    """
    client = await get_openrouter_client()
    return await client.unassign_members_from_guardrail(
        request.id, request.member_user_ids
    )


# ── cross-guardrail assignment listings ──


@mcp.tool()
async def list_all_key_assignments(
    request: ListAllAssignmentsRequest,
) -> Dict[str, Any]:
    """List all API key assignments across all guardrails.

    Args:
        request: Request with optional pagination

    Returns:
        All key-to-guardrail assignments
    """
    client = await get_openrouter_client()
    return await client.list_all_key_assignments(
        offset=request.offset, limit=request.limit
    )


@mcp.tool()
async def list_all_member_assignments(
    request: ListAllAssignmentsRequest,
) -> Dict[str, Any]:
    """List all member assignments across all guardrails.

    Args:
        request: Request with optional pagination

    Returns:
        All member-to-guardrail assignments
    """
    client = await get_openrouter_client()
    return await client.list_all_member_assignments(
        offset=request.offset, limit=request.limit
    )
