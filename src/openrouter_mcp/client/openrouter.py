import json as json_lib
import logging
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from types import TracebackType
from typing import Any, AsyncGenerator, Dict, List, NoReturn, Optional, Union

import httpx

# Import centralized configuration constants
from ..config.constants import APIConfig, CacheConfig, EnvVars, ModelDefaults

# Import ModelCache for intelligent caching
from ..models.cache import ModelCache
from ..utils.async_utils import maybe_await
from ..utils.env import get_env_value, get_required_env
from ..utils.http import build_openrouter_headers
from ..utils.pricing import normalize_pricing

# Import sanitizer from utils (extracted for SRP compliance)
from ..utils.sanitizer import SensitiveDataSanitizer


class OpenRouterError(Exception):
    """Base exception for OpenRouter API errors."""

    pass


class AuthenticationError(OpenRouterError):
    """Raised when API key is invalid or missing."""

    pass


class RateLimitError(OpenRouterError):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class InvalidRequestError(OpenRouterError):
    """Raised when request is invalid."""

    pass


_MAX_RETRY_AFTER = 3600.0


def _parse_retry_after(header_value: Optional[str]) -> Optional[float]:
    """Parse Retry-After header into seconds (clamped to [0, 3600]).

    Supports integer/float seconds and HTTP-date formats.
    Returns None on missing/unparseable values.
    """
    if not header_value:
        return None

    # Try numeric seconds first
    try:
        value = float(header_value)
        if value != value:  # NaN check
            return None
        return max(0.0, min(value, _MAX_RETRY_AFTER))
    except (ValueError, TypeError):
        pass

    # Try HTTP-date format
    try:
        target = parsedate_to_datetime(header_value)
        delta = (target - datetime.now(timezone.utc)).total_seconds()
        return max(0.0, min(delta, _MAX_RETRY_AFTER))
    except (ValueError, TypeError):
        return None


# Note: SensitiveDataSanitizer has been moved to openrouter_mcp.utils.sanitizer
# for SRP compliance. Import it from there for new code.
# The import at module level provides backward compatibility.


class OpenRouterClient:
    """Client for OpenRouter API.

    This client provides async methods to interact with the OpenRouter API,
    including model listing, chat completions, and usage tracking.

    Example:
        >>> async with OpenRouterClient(api_key="your-key") as client:
        ...     models = await client.list_models()
        ...     response = await client.chat_completion(
        ...         model="openai/gpt-4",
        ...         messages=[{"role": "user", "content": "Hello!"}]
        ...     )
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = APIConfig.BASE_URL,
        app_name: Optional[str] = None,
        http_referer: Optional[str] = None,
        timeout: float = APIConfig.DEFAULT_TIMEOUT,
        logger: Optional[logging.Logger] = None,
        enable_cache: bool = True,
        cache_ttl: int = CacheConfig.DEFAULT_TTL_SECONDS,
        enable_verbose_logging: bool = False,
    ) -> None:
        """Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key
            base_url: Base URL for OpenRouter API
            app_name: Application name for tracking
            http_referer: HTTP referer for tracking
            timeout: Request timeout in seconds
            logger: Custom logger instance
            enable_cache: Whether to enable model caching
            cache_ttl: Cache time-to-live in seconds
            enable_verbose_logging: If True, log truncated request/response content.
                                   If False (default), only log metadata.
                                   WARNING: Even with this enabled, sensitive data is sanitized,
                                   but truncated prompts/responses may still contain PII.

        Raises:
            ValueError: If API key is empty or None
        """
        if not api_key or api_key.strip() == "":
            raise ValueError("API key is required")

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.app_name = app_name
        self.http_referer = http_referer
        self.timeout = timeout
        self.logger = logger or logging.getLogger(__name__)
        self.enable_cache = enable_cache
        self.enable_verbose_logging = enable_verbose_logging

        # Log warning if verbose logging is enabled
        if self.enable_verbose_logging:
            self.logger.warning(
                "Verbose logging is enabled. Truncated request/response content will be logged. "
                "This may include sensitive information. Use only for debugging."
            )

        self._client = httpx.AsyncClient(timeout=timeout)
        self._model_cache: Optional[ModelCache] = None

        # Initialize model cache with client credentials
        if enable_cache:
            # Convert seconds to hours for ModelCache (using float for sub-hour precision)
            # Minimum TTL to prevent too-frequent refreshes
            ttl_hours = max(CacheConfig.MIN_TTL_HOURS, cache_ttl / 3600.0)
            self._model_cache = ModelCache(
                ttl_hours=ttl_hours, api_key=self.api_key, base_url=self.base_url
            )
        else:
            self._model_cache = None

    @property
    def model_cache(self) -> "ModelCache":
        """Public accessor for the model cache."""
        if self._model_cache is None:
            raise RuntimeError("Model cache is not available (caching is disabled)")
        return self._model_cache

    @classmethod
    def from_env(cls) -> "OpenRouterClient":
        """Create client from environment variables."""
        api_key = get_required_env(EnvVars.API_KEY)
        return cls(
            api_key=api_key,
            base_url=get_env_value(EnvVars.BASE_URL, APIConfig.BASE_URL)
            or APIConfig.BASE_URL,
            app_name=get_env_value(EnvVars.APP_NAME),
            http_referer=get_env_value(EnvVars.HTTP_REFERER),
        )

    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for requests."""
        raw_headers = build_openrouter_headers(
            self.api_key,
            app_name=self.app_name,
            http_referer=self.http_referer,
            fallback_to_env=False,
        )
        return {
            str(key): str(value)
            for key, value in raw_headers.items()
            if isinstance(key, str) and isinstance(value, str)
        }

    def _validate_model(self, model: str) -> None:
        """Validate model parameter."""
        if not model or model.strip() == "":
            raise ValueError("Model cannot be empty")

    def _validate_messages(self, messages: List[Dict[str, str]]) -> None:
        """Validate messages parameter."""
        if not messages:
            raise ValueError("Messages cannot be empty")

        valid_roles = {"system", "user", "assistant"}

        for message in messages:
            if "role" not in message or "content" not in message:
                raise ValueError("Message must have 'role' and 'content' fields")

            if message["role"] not in valid_roles:
                raise ValueError(
                    f"Invalid role: {message['role']}. Must be one of {valid_roles}"
                )

    def _validate_messages_if_text(self, messages: List[Dict[str, Any]]) -> None:
        """Validate messages when they are simple text-only payloads."""
        if messages and all(isinstance(msg.get("content"), str) for msg in messages):
            text_messages: List[Dict[str, str]] = []
            for message in messages:
                role = message.get("role")
                content = message.get("content")
                if not isinstance(role, str) or not isinstance(content, str):
                    return
                text_messages.append({"role": role, "content": content})
            self._validate_messages(text_messages)

    def _build_chat_payload(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: Optional[int],
        stream: bool,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build a chat completion payload with shared validation."""
        self._validate_model(model)
        self._validate_messages_if_text(messages)

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
            **kwargs,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        return payload

    def _log_request(
        self,
        method_label: str,
        url: str,
        headers: Dict[str, str],
        payload: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log sanitized request details."""
        sanitized_headers = SensitiveDataSanitizer.sanitize_headers(headers)
        self.logger.debug(f"Making {method_label} request to {url}")
        self.logger.debug(f"Request headers: {sanitized_headers}")
        if payload:
            sanitized_payload = SensitiveDataSanitizer.sanitize_payload(
                payload, enable_verbose=self.enable_verbose_logging
            )
            self.logger.debug(f"Request payload: {sanitized_payload}")
        if params:
            self.logger.debug(f"Request params: {params}")

    def _handle_request_error(self, e: Exception, context: str, url: str) -> NoReturn:
        """Handle non-HTTP request errors (connect, timeout, generic)."""
        if isinstance(e, httpx.ConnectError):
            self.logger.error(f"Connection error for {context} {url}: {str(e)}")
            raise OpenRouterError(
                "Network error: Failed to connect to OpenRouter API"
            ) from e
        if isinstance(e, httpx.TimeoutException):
            self.logger.error(f"Timeout error for {context} {url}: {str(e)}")
            raise OpenRouterError(
                f"Request timeout after {self.timeout} seconds"
            ) from e
        self.logger.error(f"Unexpected error for {context} {url}: {str(e)}")
        raise OpenRouterError(f"Unexpected error: {str(e)}") from e

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make HTTP request to OpenRouter API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            json: JSON payload for POST/PUT requests
            params: URL parameters

        Returns:
            Response data as dictionary

        Raises:
            OpenRouterError: For API errors, network issues, or unexpected errors
        """
        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers()
        self._log_request(method, url, headers, payload=json, params=params)

        try:
            response = await self._client.request(
                method=method, url=url, headers=headers, json=json, params=params
            )

            self.logger.debug(f"Response status: {response.status_code}")
            await maybe_await(response.raise_for_status())

            response_data = await maybe_await(response.json())
            if not isinstance(response_data, dict):
                raise OpenRouterError(
                    f"Unexpected response type: {type(response_data).__name__}"
                )

            # Sanitize response for logging
            if "choices" in response_data or "data" in response_data:
                # This looks like a completion or model list response
                if "choices" in response_data:
                    sanitized_response = SensitiveDataSanitizer.sanitize_response(
                        response_data, enable_verbose=self.enable_verbose_logging
                    )
                    self.logger.debug(f"Response data: {sanitized_response}")
                else:
                    # For non-completion responses (like model lists), log keys only
                    self.logger.debug(
                        f"Response data keys: {list(response_data.keys())}"
                    )
            else:
                self.logger.debug(f"Response data keys: {list(response_data.keys())}")

            return response_data

        except httpx.HTTPStatusError as e:
            self.logger.warning(
                f"HTTP error {e.response.status_code} for {method} {url}"
            )
            await self._handle_http_error(e.response)
        except Exception as e:
            self._handle_request_error(e, method, url)

    async def _stream_request(
        self, endpoint: str, json_data: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Make streaming request to OpenRouter API.

        Args:
            endpoint: API endpoint path
            json_data: JSON payload for the request

        Yields:
            Streaming response chunks as dictionaries

        Raises:
            OpenRouterError: For API errors, network issues, or unexpected errors
        """
        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers()
        self._log_request("streaming POST", url, headers, payload=json_data)

        try:
            async with self._client.stream(
                "POST", url, headers=headers, json=json_data
            ) as response:
                self.logger.debug(f"Stream response status: {response.status_code}")
                await maybe_await(response.raise_for_status())

                chunk_count = 0
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix
                        if data.strip() == "[DONE]":
                            self.logger.debug(
                                f"Stream completed after {chunk_count} chunks"
                            )
                            break
                        try:
                            chunk = json_lib.loads(data)
                            chunk_count += 1

                            # Log chunk metadata only (don't log content even in verbose mode for streaming)
                            if (
                                chunk_count % 10 == 1
                            ):  # Log every 10th chunk to reduce noise
                                self.logger.debug(
                                    f"Streaming chunk {chunk_count} "
                                    f"(keys: {list(chunk.keys()) if isinstance(chunk, dict) else 'non-dict'})"
                                )

                            yield chunk
                        except json_lib.JSONDecodeError as e:
                            # Don't log the actual data content - could contain sensitive info
                            self.logger.warning(
                                f"Failed to parse stream chunk (length: {len(data)}): {str(e)}"
                            )
                            continue

        except httpx.HTTPStatusError as e:
            self.logger.warning(
                f"HTTP error {e.response.status_code} for streaming POST {url}"
            )
            await self._handle_http_error(e.response)
        except Exception as e:
            self._handle_request_error(e, "streaming POST", url)

    async def _handle_http_error(self, response: httpx.Response) -> NoReturn:
        """Handle HTTP errors from OpenRouter API.

        SECURITY: Response bodies are sanitized to prevent leaking sensitive data in error messages.
        """
        try:
            error_data = await maybe_await(response.json())
            error_message = error_data.get("error", {}).get("message", "Unknown error")
        except (json_lib.JSONDecodeError, KeyError):
            # SECURITY: Don't include raw response.text - it may contain sensitive data
            # Truncate and sanitize the response body
            response_preview = (
                SensitiveDataSanitizer.truncate_content(response.text, max_length=100)
                if response.text
                else "No response body"
            )
            error_message = f"HTTP {response.status_code}: {response_preview}"

        if response.status_code == 401:
            raise AuthenticationError(error_message)
        elif response.status_code == 429:
            retry_after = _parse_retry_after(response.headers.get("Retry-After"))
            raise RateLimitError(error_message, retry_after=retry_after)
        elif response.status_code == 400:
            raise InvalidRequestError(error_message)
        else:
            raise OpenRouterError(f"API error: {error_message}")

    async def list_models(
        self,
        filter_by: Optional[str] = None,
        use_cache: bool = True,
        _bypass_cache: bool = False,
    ) -> List[Dict[str, Any]]:
        """List available models from OpenRouter.

        Retrieves a list of all available AI models, optionally filtered by name.
        Each model includes information about pricing, context length, and capabilities.

        Args:
            filter_by: Optional string to filter model names (case-insensitive)
            use_cache: Whether to use cached models if available
            _bypass_cache: Internal flag to bypass cache (prevents recursion)

        Returns:
            List of dictionaries containing model information with keys:
            - id: Model identifier (e.g., "openai/gpt-4")
            - name: Human-readable model name
            - description: Model description
            - pricing: Dictionary with prompt/completion pricing
            - context_length: Maximum context window size
            - architecture: Model architecture details

        Raises:
            OpenRouterError: If the API request fails

        Example:
            >>> models = await client.list_models()
            >>> gpt_models = await client.list_models(filter_by="gpt")
        """
        # Use cache system if enabled and not explicitly bypassed
        if use_cache and self._model_cache and not _bypass_cache:
            try:
                cached_models_raw = await self._model_cache.get_models()
                all_models: List[Dict[str, Any]] = [
                    model for model in cached_models_raw if isinstance(model, dict)
                ]
                if all_models:
                    self.logger.info(f"Retrieved {len(all_models)} models from cache")

                    # Apply filter if specified
                    if filter_by:
                        filter_lower = filter_by.lower()
                        filtered_models = [
                            model
                            for model in all_models
                            if filter_lower in model.get("name", "").lower()
                            or filter_lower in model.get("id", "").lower()
                        ]
                        self.logger.info(f"Filtered to {len(filtered_models)} models")
                        return filtered_models
                    else:
                        return all_models
            except Exception as e:
                self.logger.warning(f"Failed to get cached models: {e}")
                # Continue to API fetch

        # Fallback: Fetch directly from API if cache is disabled or failed
        self.logger.info(
            f"Fetching models directly from API with filter: {filter_by or 'none'}"
        )

        params: Dict[str, Any] = {}
        if filter_by:
            params["filter"] = filter_by

        response = await self._make_request("GET", "/models", params=params)
        models_raw = response.get("data", [])
        if not isinstance(models_raw, list):
            self.logger.warning("Unexpected model list format from API")
            return []

        models: List[Dict[str, Any]] = [
            model for model in models_raw if isinstance(model, dict)
        ]

        self.logger.info(f"Retrieved {len(models)} models from API")
        return models

    async def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get information about a specific model.

        Args:
            model: Model identifier (e.g., "openai/gpt-4")

        Returns:
            Model information dictionary
        """
        self._validate_model(model)
        return await self._make_request("GET", f"/models/{model}")

    async def get_model_pricing(self, model: str) -> Dict[str, Any]:
        """Get normalized pricing for a specific model.

        Pricing values are normalized to per-token dollars to ensure consistent
        cost calculations across the codebase. The return value preserves the
        numeric ``prompt``/``completion`` fields and adds ``_meta`` to expose
        whether pricing data was available or a fallback was used.
        """
        self._validate_model(model)
        pricing: Dict[str, Any] = {}
        pricing_available = False
        fallback_used = False
        source = "cache" if self._model_cache else "api"

        try:
            if self._model_cache:
                model_info = await self._model_cache.get_model_info(model)
            else:
                model_info = await self.get_model_info(model)

            pricing_data = model_info.get("pricing") if model_info else None
            if isinstance(pricing_data, dict) and (
                "prompt" in pricing_data or "completion" in pricing_data
            ):
                pricing = pricing_data
                pricing_available = True
        except Exception as e:
            self.logger.warning(f"Failed to fetch pricing for model {model}: {e}")
            fallback_used = True

        if pricing_available:
            normalized = normalize_pricing(pricing, fill_missing=False)
            if "prompt" not in pricing and "completion" in pricing:
                normalized["prompt"] = normalized["completion"]
            if "completion" not in pricing and "prompt" in pricing:
                normalized["completion"] = normalized["prompt"]
        else:
            fallback_used = True
            source = "fallback"
            normalized = normalize_pricing(pricing)

        return {
            "prompt": float(normalized.get("prompt", 0.0)),
            "completion": float(normalized.get("completion", 0.0)),
            "_meta": {
                "pricing_available": pricing_available,
                "fallback_used": fallback_used,
                "source": source,
            },
        }

    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = ModelDefaults.TEMPERATURE,
        max_tokens: Optional[int] = ModelDefaults.MAX_TOKENS,
        stream: bool = ModelDefaults.STREAM,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create a chat completion.

        Args:
            model: Model to use
            messages: List of message dictionaries (can include image content)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional parameters

        Returns:
            Chat completion response
        """
        payload = self._build_chat_payload(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            **kwargs,
        )

        return await self._make_request("POST", "/chat/completions", json=payload)

    async def stream_chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = ModelDefaults.TEMPERATURE,
        max_tokens: Optional[int] = ModelDefaults.MAX_TOKENS,
        **kwargs: Any,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Create a streaming chat completion.

        Args:
            model: Model to use
            messages: List of message dictionaries (can include image content)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Yields:
            Chat completion chunks
        """
        payload = self._build_chat_payload(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )

        async for chunk in self._stream_request("/chat/completions", payload):
            yield chunk

    async def track_usage(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Deprecated: use get_generation(id) or get_activity() instead.

        Kept for backward compatibility but now calls get_activity.
        """
        return await self.get_activity(date=start_date)

    async def _make_request_flexible(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Union[Dict[str, Any], List[Any]]:
        """like _make_request but allows array responses too."""
        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers()
        self._log_request(method, url, headers, payload=json, params=params)

        try:
            response = await self._client.request(
                method=method, url=url, headers=headers, json=json, params=params
            )
            self.logger.debug(f"Response status: {response.status_code}")
            await maybe_await(response.raise_for_status())
            return await maybe_await(response.json())

        except httpx.HTTPStatusError as e:
            self.logger.warning(
                f"HTTP error {e.response.status_code} for {method} {url}"
            )
            await self._handle_http_error(e.response)
        except Exception as e:
            self._handle_request_error(e, method, url)

    # ── generation stats ──

    async def get_generation(self, generation_id: str) -> Dict[str, Any]:
        """get request and usage metadata for a single generation.

        Args:
            generation_id: the generation ID (returned in chat completion responses)

        Returns:
            generation metadata including tokens, cost, model, and latency
        """
        return await self._make_request(
            "GET", "/generation", params={"id": generation_id}
        )

    # ── model endpoints / providers ──

    async def list_model_endpoints(
        self, author: str, slug: str
    ) -> Union[Dict[str, Any], List[Any]]:
        """list all provider endpoints serving a specific model.

        Args:
            author: model author (e.g. "openai")
            slug: model slug (e.g. "gpt-4")
        """
        return await self._make_request_flexible(
            "GET", f"/models/{author}/{slug}/endpoints"
        )

    async def list_providers(self) -> Union[Dict[str, Any], List[Any]]:
        """list all available providers."""
        return await self._make_request_flexible("GET", "/providers")

    async def get_models_count(self) -> Dict[str, Any]:
        """get total count of available models."""
        return await self._make_request("GET", "/models/count")

    async def list_user_models(self) -> Dict[str, Any]:
        """list models filtered by user provider preferences, privacy settings, and guardrails."""
        return await self._make_request("GET", "/models/user")

    async def list_zdr_endpoints(self) -> Union[Dict[str, Any], List[Any]]:
        """preview the impact of Zero Data Retention on available endpoints."""
        return await self._make_request_flexible("GET", "/endpoints/zdr")

    # ── account / credits ──

    async def get_credits(self) -> Dict[str, Any]:
        """get remaining credits for the current API key."""
        return await self._make_request("GET", "/credits")

    async def get_current_key(self) -> Dict[str, Any]:
        """get info about the current API key."""
        return await self._make_request("GET", "/key")

    async def get_activity(self, date: Optional[str] = None) -> Dict[str, Any]:
        """get user activity grouped by endpoint.

        Args:
            date: filter by UTC date (YYYY-MM-DD), must be within last 30 days
        """
        params = {}
        if date:
            params["date"] = date
        return await self._make_request("GET", "/activity", params=params)

    # ── API key management ──

    async def list_api_keys(
        self,
        include_disabled: Optional[bool] = None,
        offset: Optional[int] = None,
    ) -> Dict[str, Any]:
        """list API keys for the current account."""
        params: Dict[str, Any] = {}
        if include_disabled is not None:
            params["include_disabled"] = str(include_disabled).lower()
        if offset is not None:
            params["offset"] = offset
        return await self._make_request("GET", "/keys", params=params)

    async def create_api_key(
        self,
        name: str,
        limit: Optional[float] = None,
        limit_reset: Optional[str] = None,
        include_byok_in_limit: Optional[bool] = None,
        expires_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """create a new API key."""
        payload: Dict[str, Any] = {"name": name}
        if limit is not None:
            payload["limit"] = limit
        if limit_reset is not None:
            payload["limit_reset"] = limit_reset
        if include_byok_in_limit is not None:
            payload["include_byok_in_limit"] = include_byok_in_limit
        if expires_at is not None:
            payload["expires_at"] = expires_at
        return await self._make_request("POST", "/keys", json=payload)

    async def get_api_key(self, key_hash: str) -> Dict[str, Any]:
        """get a single API key by hash."""
        return await self._make_request("GET", f"/keys/{key_hash}")

    async def update_api_key(
        self,
        key_hash: str,
        name: Optional[str] = None,
        disabled: Optional[bool] = None,
        limit: Optional[float] = None,
        limit_reset: Optional[str] = None,
        include_byok_in_limit: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """update an API key."""
        payload: Dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if disabled is not None:
            payload["disabled"] = disabled
        if limit is not None:
            payload["limit"] = limit
        if limit_reset is not None:
            payload["limit_reset"] = limit_reset
        if include_byok_in_limit is not None:
            payload["include_byok_in_limit"] = include_byok_in_limit
        return await self._make_request("PATCH", f"/keys/{key_hash}", json=payload)

    async def delete_api_key(self, key_hash: str) -> Dict[str, Any]:
        """delete an API key."""
        return await self._make_request("DELETE", f"/keys/{key_hash}")

    # ── embeddings ──

    async def create_embedding(
        self,
        model: str,
        input_text: Union[str, List[str]],
        encoding_format: Optional[str] = None,
        dimensions: Optional[int] = None,
        user: Optional[str] = None,
    ) -> Dict[str, Any]:
        """create embeddings for the given input text."""
        payload: Dict[str, Any] = {"model": model, "input": input_text}
        if encoding_format is not None:
            payload["encoding_format"] = encoding_format
        if dimensions is not None:
            payload["dimensions"] = dimensions
        if user is not None:
            payload["user"] = user
        return await self._make_request("POST", "/embeddings", json=payload)

    async def list_embedding_models(self) -> Union[Dict[str, Any], List[Any]]:
        """list all available embedding models."""
        return await self._make_request_flexible("GET", "/embeddings/models")

    # ── responses API (OpenAI-compatible) ──

    async def create_response(
        self,
        model: str,
        input_data: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """create a response using OpenRouter's Responses API.

        Args:
            model: model identifier
            input_data: string or array of input items
            **kwargs: additional params (tools, temperature, max_output_tokens, etc.)
        """
        payload: Dict[str, Any] = {"model": model, "input": input_data, **kwargs}
        return await self._make_request("POST", "/responses", json=payload)

    # ── messages API (Anthropic-compatible) ──

    async def create_message(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """create a message using the Anthropic Messages API format.

        Args:
            model: model identifier
            messages: list of message objects
            max_tokens: maximum tokens to generate
            **kwargs: additional params (system, temperature, tools, etc.)
        """
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            **kwargs,
        }
        return await self._make_request("POST", "/messages", json=payload)

    # ── auth ──

    async def exchange_auth_code(
        self,
        code: str,
        code_verifier: Optional[str] = None,
        code_challenge_method: Optional[str] = None,
    ) -> Dict[str, Any]:
        """exchange an authorization code for an API key."""
        payload: Dict[str, Any] = {"code": code}
        if code_verifier is not None:
            payload["code_verifier"] = code_verifier
        if code_challenge_method is not None:
            payload["code_challenge_method"] = code_challenge_method
        return await self._make_request("POST", "/auth/keys", json=payload)

    async def create_auth_code(
        self,
        callback_url: str,
        code_challenge: Optional[str] = None,
        code_challenge_method: Optional[str] = None,
        limit: Optional[float] = None,
        expires_at: Optional[str] = None,
        key_label: Optional[str] = None,
        usage_limit_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """create an authorization code for OAuth PKCE flow."""
        payload: Dict[str, Any] = {"callback_url": callback_url}
        for key, value in {
            "code_challenge": code_challenge,
            "code_challenge_method": code_challenge_method,
            "limit": limit,
            "expires_at": expires_at,
            "key_label": key_label,
            "usage_limit_type": usage_limit_type,
        }.items():
            if value is not None:
                payload[key] = value
        return await self._make_request("POST", "/auth/keys/code", json=payload)

    # ── credits / coinbase ──

    async def create_coinbase_charge(
        self,
        amount: float,
        sender: Optional[str] = None,
    ) -> Dict[str, Any]:
        """create a Coinbase Commerce charge for adding credits."""
        payload: Dict[str, Any] = {"amount": amount}
        if sender is not None:
            payload["sender"] = sender
        return await self._make_request("POST", "/credits/coinbase", json=payload)

    # ── guardrails ──

    async def list_guardrails(
        self,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """list guardrails for the current organization."""
        params: Dict[str, Any] = {}
        if offset is not None:
            params["offset"] = offset
        if limit is not None:
            params["limit"] = limit
        return await self._make_request("GET", "/guardrails", params=params)

    async def create_guardrail(self, name: str, **kwargs: Any) -> Dict[str, Any]:
        """create a guardrail."""
        payload: Dict[str, Any] = {"name": name, **kwargs}
        return await self._make_request("POST", "/guardrails", json=payload)

    async def get_guardrail(self, guardrail_id: str) -> Dict[str, Any]:
        """get a guardrail by ID."""
        return await self._make_request("GET", f"/guardrails/{guardrail_id}")

    async def update_guardrail(
        self, guardrail_id: str, **kwargs: Any
    ) -> Dict[str, Any]:
        """update a guardrail."""
        return await self._make_request(
            "PATCH", f"/guardrails/{guardrail_id}", json=kwargs
        )

    async def delete_guardrail(self, guardrail_id: str) -> Dict[str, Any]:
        """delete a guardrail."""
        return await self._make_request("DELETE", f"/guardrails/{guardrail_id}")

    async def list_guardrail_key_assignments(
        self,
        guardrail_id: str,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """list key assignments for a guardrail."""
        params: Dict[str, Any] = {}
        if offset is not None:
            params["offset"] = offset
        if limit is not None:
            params["limit"] = limit
        return await self._make_request(
            "GET", f"/guardrails/{guardrail_id}/assignments/keys", params=params
        )

    async def assign_keys_to_guardrail(
        self, guardrail_id: str, key_hashes: List[str]
    ) -> Dict[str, Any]:
        """bulk assign keys to a guardrail."""
        return await self._make_request(
            "POST",
            f"/guardrails/{guardrail_id}/assignments/keys",
            json={"key_hashes": key_hashes},
        )

    async def unassign_keys_from_guardrail(
        self, guardrail_id: str, key_hashes: List[str]
    ) -> Dict[str, Any]:
        """bulk unassign keys from a guardrail."""
        return await self._make_request(
            "POST",
            f"/guardrails/{guardrail_id}/assignments/keys/remove",
            json={"key_hashes": key_hashes},
        )

    async def list_guardrail_member_assignments(
        self,
        guardrail_id: str,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """list member assignments for a guardrail."""
        params: Dict[str, Any] = {}
        if offset is not None:
            params["offset"] = offset
        if limit is not None:
            params["limit"] = limit
        return await self._make_request(
            "GET", f"/guardrails/{guardrail_id}/assignments/members", params=params
        )

    async def assign_members_to_guardrail(
        self, guardrail_id: str, member_user_ids: List[str]
    ) -> Dict[str, Any]:
        """bulk assign members to a guardrail."""
        return await self._make_request(
            "POST",
            f"/guardrails/{guardrail_id}/assignments/members",
            json={"member_user_ids": member_user_ids},
        )

    async def unassign_members_from_guardrail(
        self, guardrail_id: str, member_user_ids: List[str]
    ) -> Dict[str, Any]:
        """bulk unassign members from a guardrail."""
        return await self._make_request(
            "POST",
            f"/guardrails/{guardrail_id}/assignments/members/remove",
            json={"member_user_ids": member_user_ids},
        )

    async def list_all_key_assignments(
        self, offset: Optional[int] = None, limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """list all key assignments across guardrails."""
        params: Dict[str, Any] = {}
        if offset is not None:
            params["offset"] = offset
        if limit is not None:
            params["limit"] = limit
        return await self._make_request(
            "GET", "/guardrails/assignments/keys", params=params
        )

    async def list_all_member_assignments(
        self, offset: Optional[int] = None, limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """list all member assignments across guardrails."""
        params: Dict[str, Any] = {}
        if offset is not None:
            params["offset"] = offset
        if limit is not None:
            params["limit"] = limit
        return await self._make_request(
            "GET", "/guardrails/assignments/members", params=params
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    def get_cache_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the model cache.

        Returns:
            Cache information dictionary or None if cache is disabled.
        """
        if self._model_cache:
            stats = self._model_cache.get_cache_stats()
            return {str(key): value for key, value in stats.items()}
        return None

    async def clear_cache(self) -> None:
        """Clear the model cache."""
        if self._model_cache:
            self._model_cache.clear()
            self.logger.info("Model cache cleared")

    async def __aenter__(self) -> "OpenRouterClient":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Async context manager exit."""
        await self.close()
