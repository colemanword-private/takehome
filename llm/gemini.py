from __future__ import annotations

import asyncio
import inspect
import math
import os
import random
from dataclasses import dataclass, replace
from typing import Any, Awaitable, Callable

import httpx
import google.auth
from google import genai
from google.auth import exceptions as google_auth_exceptions
from google.genai import errors, types

from .llm import LLM
from .retry import (
    BackoffEvent,
    RetryBudget,
    RetryEvent,
    RetryExhaustedEvent,
    RetryPolicy,
    retry_after_seconds,
    retry_with_backoff,
)

# Retrying other 4xx responses would amplify permanent configuration or prompt errors.
_RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
_TRANSPORT_ERRORS = (
    TimeoutError,
    ConnectionError,
    httpx.TransportError,
    google_auth_exceptions.TransportError,
)
_HANDLED_ERRORS = (errors.APIError, *_TRANSPORT_ERRORS)


class GeminiResponseError(RuntimeError):
    """Raised when Vertex returns a successful response without usable text."""

    def __init__(
        self, message: str, *, input_tokens: int, output_tokens: int
    ) -> None:
        super().__init__(message)
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


@dataclass(frozen=True)
class GeminiConfig:
    """Runtime controls that materially affect capacity, latency, and cost."""

    project: str
    location: str = "global"
    model: str = "gemini-2.5-flash"
    parallelism: int = 32
    timeout_seconds: float = 20.0
    max_retries: int = 4
    retry_base_delay_seconds: float = 0.5
    retry_max_delay_seconds: float = 8.0
    max_output_tokens: int | None = None
    thinking_budget: int | None = None
    request_deadline_seconds: float = 60.0
    retry_budget_capacity: int = 32
    retry_budget_refill_per_second: float = 2.0

    @classmethod
    def from_env(cls) -> GeminiConfig:
        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise ValueError("GOOGLE_CLOUD_PROJECT must be set for Vertex AI")

        return cls(
            project=project,
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            parallelism=_env_int("GEMINI_PARALLELISM", 32, minimum=1),
            timeout_seconds=_env_float(
                "GEMINI_TIMEOUT_SECONDS", 20.0, minimum=0.001
            ),
            request_deadline_seconds=_env_float(
                "GEMINI_REQUEST_DEADLINE_SECONDS", 60.0, minimum=0.001
            ),
            max_retries=_env_int("GEMINI_MAX_RETRIES", 4, minimum=0),
            retry_base_delay_seconds=_env_float(
                "GEMINI_RETRY_BASE_DELAY_SECONDS", 0.5, minimum=0.0
            ),
            retry_max_delay_seconds=_env_float(
                "GEMINI_RETRY_MAX_DELAY_SECONDS", 8.0, minimum=0.0
            ),
            retry_budget_capacity=_env_int(
                "GEMINI_RETRY_BUDGET_CAPACITY", 32, minimum=1
            ),
            retry_budget_refill_per_second=_env_float(
                "GEMINI_RETRY_BUDGET_REFILL_PER_SECOND", 2.0, minimum=0.0
            ),
            max_output_tokens=_env_optional_int(
                "GEMINI_MAX_OUTPUT_TOKENS", minimum=1
            ),
            thinking_budget=_env_optional_int(
                "GEMINI_THINKING_BUDGET", minimum=0
            ),
        )

    def __post_init__(self) -> None:
        if not self.project:
            raise ValueError("project must not be empty")
        if not self.location:
            raise ValueError("location must not be empty")
        if not self.model:
            raise ValueError("model must not be empty")
        if self.parallelism < 1:
            raise ValueError("parallelism must be at least 1")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        if (
            not math.isfinite(self.request_deadline_seconds)
            or self.request_deadline_seconds <= 0
        ):
            raise ValueError("request_deadline_seconds must be finite and positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if (
            not math.isfinite(self.retry_base_delay_seconds)
            or self.retry_base_delay_seconds < 0
        ):
            raise ValueError(
                "retry_base_delay_seconds must be finite and non-negative"
            )
        if not math.isfinite(self.retry_max_delay_seconds):
            raise ValueError("retry_max_delay_seconds must be finite")
        if self.retry_max_delay_seconds < self.retry_base_delay_seconds:
            raise ValueError(
                "retry_max_delay_seconds must be at least retry_base_delay_seconds"
            )
        if self.retry_budget_capacity < 1:
            raise ValueError("retry_budget_capacity must be at least 1")
        if (
            not math.isfinite(self.retry_budget_refill_per_second)
            or self.retry_budget_refill_per_second < 0
        ):
            raise ValueError(
                "retry_budget_refill_per_second must be finite and non-negative"
            )
        if self.max_output_tokens is not None and self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if self.thinking_budget is not None and self.thinking_budget < 0:
            raise ValueError("thinking_budget must not be negative")


class Gemini(LLM):
    """Gemini 2.5 Flash provider backed by Google Vertex AI."""

    def __init__(
        self,
        config: GeminiConfig | None = None,
        *,
        client: Any | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        jitter: Callable[[float, float], float] = random.uniform,
        on_backoff: Callable[[BackoffEvent], None] | None = None,
        on_retry: Callable[[RetryEvent], None] | None = None,
        on_exhausted: Callable[[RetryExhaustedEvent], None] | None = None,
    ) -> None:
        self._config = config or GeminiConfig.from_env()
        self._sleep = sleep
        self._jitter = jitter
        self._on_backoff = on_backoff
        self._on_retry = on_retry
        self._on_exhausted = on_exhausted
        self._closed = False
        self._retry_policy = RetryPolicy(
            max_retries=self._config.max_retries,
            base_delay_seconds=self._config.retry_base_delay_seconds,
            max_delay_seconds=self._config.retry_max_delay_seconds,
        )
        self._retry_budget = RetryBudget(
            capacity=self._config.retry_budget_capacity,
            refill_rate_per_second=self._config.retry_budget_refill_per_second,
        )

        # Construct one SDK client per provider so concurrent calls share its
        # authentication state and connection pool instead of reconnecting per request.
        self._client = client or genai.Client(
            vertexai=True,
            project=self._config.project,
            location=self._config.location,
            http_options=types.HttpOptions(
                # Pin Vertex's stable API rather than inheriting an SDK preview default.
                api_version="v1",
                # The SDK expects milliseconds, while our public setting uses seconds.
                timeout=int(self._config.timeout_seconds * 1_000),
                # The SDK otherwise selects aiohttp whenever it is installed, making
                # the transport depend on unrelated packages in the environment.
                async_client_args={"transport": httpx.AsyncHTTPTransport()},
            ),
        )
        self._async_client = self._client.aio

    def parallelism(self) -> int:
        return self._config.parallelism

    def metadata(self) -> dict[str, object]:
        return {
            "provider": "Gemini",
            "platform": "Vertex AI",
            "project": self._config.project,
            "location": self._config.location,
            "model": self._config.model,
            "parallelism": self._config.parallelism,
            "timeout_seconds": self._config.timeout_seconds,
            "timeout_scope": "per_attempt",
            "request_deadline_seconds": self._config.request_deadline_seconds,
            "max_retries": self._config.max_retries,
            "retry_base_delay_seconds": self._config.retry_base_delay_seconds,
            "retry_max_delay_seconds": self._config.retry_max_delay_seconds,
            "retry_budget_capacity": self._config.retry_budget_capacity,
            "retry_budget_refill_per_second": (
                self._config.retry_budget_refill_per_second
            ),
            "max_output_tokens": self._config.max_output_tokens,
            "thinking_budget": self._config.thinking_budget,
        }

    async def ask_generic_question(
        self, system_prompt: str, question: str, temperature: float
    ) -> LLM.SimpleResponse:
        if self._closed:
            raise RuntimeError("Gemini client is closed")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")

        # Keep the system instruction out of user content so Gemini applies the
        # two inputs with their intended roles. None preserves Vertex model defaults.
        generation_config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=self._config.max_output_tokens,
            thinking_config=self._thinking_config(),
        )

        response = await self._generate_with_retries(
            question=question,
            generation_config=generation_config,
        )
        # Capture usage before inspecting text: safety-blocked and otherwise empty
        # responses may still consume billable tokens that callers need to observe.
        input_tokens, output_tokens = _token_counts(response)
        answer = response.text
        if not answer:
            raise GeminiResponseError(
                _empty_response_message(response),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        return LLM.SimpleResponse(
            answer=answer,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        try:
            # The SDK exposes a separate async transport beneath the owning client.
            await self._async_client.aclose()
        finally:
            # Close the owner even if async cleanup fails. Supporting an awaitable
            # result also keeps cleanup compatible across SDK client implementations.
            close_result = self._client.close()
            if inspect.isawaitable(close_result):
                await close_result

    async def __aenter__(self) -> Gemini:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    def _thinking_config(self) -> types.ThinkingConfig | None:
        if self._config.thinking_budget is None:
            # Omitting the config lets the model choose its default; a budget of zero
            # is intentionally different because it explicitly disables thinking.
            return None
        return types.ThinkingConfig(thinking_budget=self._config.thinking_budget)

    async def _generate_with_retries(
        self,
        *,
        question: str,
        generation_config: types.GenerateContentConfig,
    ) -> Any:
        async def generate_once() -> Any:
            # The outer deadline also bounds authentication and SDK transport work.
            return await asyncio.wait_for(
                self._async_client.models.generate_content(
                    model=self._config.model,
                    contents=question,
                    config=generation_config,
                ),
                timeout=self._config.timeout_seconds,
            )

        return await retry_with_backoff(
            generate_once,
            policy=self._retry_policy,
            handled_errors=_HANDLED_ERRORS,
            is_retryable=_is_retryable,
            status_code=_status_code,
            retry_after=retry_after_seconds,
            retry_budget=self._retry_budget,
            total_timeout_seconds=self._config.request_deadline_seconds,
            sleep=self._sleep,
            jitter=self._jitter,
            on_backoff=self._on_backoff,
            on_retry=self._on_retry,
            on_exhausted=self._on_exhausted,
        )


def check_gemini_readiness(model: str | None = None) -> dict[str, object]:
    """Validate local Vertex configuration and ADC without calling Vertex AI."""
    config = GeminiConfig.from_env()
    if model is not None:
        config = replace(config, model=model)
    try:
        google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
    except google_auth_exceptions.DefaultCredentialsError as error:
        raise ValueError(
            "Application Default Credentials are missing; run "
            "`gcloud auth application-default login`"
        ) from error
    return {
        "provider": "gemini",
        "platform": "Vertex AI",
        "project": config.project,
        "location": config.location,
        "model": config.model,
        "credentials": "found",
    }


def _token_counts(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return 0, 0

    input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
    candidate_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
    thought_tokens = int(getattr(usage, "thoughts_token_count", 0) or 0)
    total_tokens = getattr(usage, "total_token_count", None)

    # Gemini thinking tokens are hidden from response.text but still consume output
    # capacity. Prefer total minus prompt so they are not silently discarded.
    if total_tokens is None:
        output_tokens = candidate_tokens + thought_tokens
    else:
        output_tokens = max(candidate_tokens, int(total_tokens) - input_tokens)
    return input_tokens, output_tokens


def _is_retryable(error: BaseException) -> bool:
    if isinstance(error, _TRANSPORT_ERRORS):
        return True
    return _status_code(error) in _RETRYABLE_STATUS_CODES


def _status_code(error: BaseException) -> int | None:
    code = getattr(error, "code", None) or getattr(error, "status", None)
    try:
        return int(code) if code is not None else None
    except (TypeError, ValueError):
        return None


def _empty_response_message(response: Any) -> str:
    # Vertex can return HTTP success with no text after a safety block or a terminal
    # candidate state. Surface those reasons without logging prompt/response content.
    prompt_feedback = getattr(response, "prompt_feedback", None)
    block_reason = getattr(prompt_feedback, "block_reason", None)

    candidates = getattr(response, "candidates", None) or []
    finish_reason = (
        getattr(candidates[0], "finish_reason", None) if candidates else None
    )

    details = []
    if block_reason:
        details.append(f"block_reason={block_reason}")
    if finish_reason:
        details.append(f"finish_reason={finish_reason}")
    suffix = f" ({', '.join(details)})" if details else ""
    return f"Gemini returned no text{suffix}"


def _env_int(name: str, default: int, *, minimum: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _env_optional_int(name: str, *, minimum: int) -> int | None:
    if os.getenv(name) is None:
        return None
    return _env_int(name, minimum, minimum=minimum)


def _env_float(name: str, default: float, *, minimum: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be a number") from error
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value
