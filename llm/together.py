from __future__ import annotations

import asyncio
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import httpx
from together import (
    APIConnectionError,
    APIError,
    APIStatusError,
    AsyncTogether,
)
from together.types.chat.completion_create_params import (
    MessageChatCompletionSystemMessageParam,
    MessageChatCompletionUserMessageParam,
)

from .llm import LLM
from .retry import RetryEvent, RetryPolicy, retry_with_backoff

_RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
_TRANSPORT_ERRORS = (TimeoutError, ConnectionError, httpx.TransportError)
_HANDLED_ERRORS = (APIError, *_TRANSPORT_ERRORS)


class TogetherResponseError(RuntimeError):
    """Raised when Together returns a response without usable answer text."""

    def __init__(
        self, message: str, *, input_tokens: int, output_tokens: int
    ) -> None:
        super().__init__(message)
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


@dataclass(frozen=True)
class TogetherConfig:
    """Runtime controls for the Together provider."""

    model: str
    parallelism: int = 100
    max_retries: int = 2
    retry_base_delay_seconds: float = 0.5
    retry_max_delay_seconds: float = 8.0
    max_output_tokens: int | None = None

    @classmethod
    def from_env(cls, *, model: str | None = None) -> TogetherConfig:
        resolved_model = model or os.getenv("TOGETHER_MODEL")
        if not resolved_model:
            raise ValueError("--model or TOGETHER_MODEL is required for Together")
        return cls(
            model=resolved_model,
            parallelism=_env_int("TOGETHER_PARALLELISM", 100, minimum=1),
            max_retries=_env_int("TOGETHER_MAX_RETRIES", 2, minimum=0),
            retry_base_delay_seconds=_env_float(
                "TOGETHER_RETRY_BASE_DELAY_SECONDS", 0.5, minimum=0.0
            ),
            retry_max_delay_seconds=_env_float(
                "TOGETHER_RETRY_MAX_DELAY_SECONDS", 8.0, minimum=0.0
            ),
            max_output_tokens=_env_optional_int(
                "TOGETHER_MAX_OUTPUT_TOKENS", minimum=1
            ),
        )

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must not be empty")
        if self.parallelism < 1:
            raise ValueError("parallelism must be at least 1")
        RetryPolicy(
            max_retries=self.max_retries,
            base_delay_seconds=self.retry_base_delay_seconds,
            max_delay_seconds=self.retry_max_delay_seconds,
        )
        if self.max_output_tokens is not None and self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")


class Together(LLM):
    def __init__(
        self,
        model: str | None = None,
        *,
        config: TogetherConfig | None = None,
        client: AsyncTogether | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        jitter: Callable[[float, float], float] = random.uniform,
        on_retry: Callable[[RetryEvent], None] | None = None,
    ) -> None:
        if config is not None and model is not None:
            raise ValueError("pass either model or config, not both")
        self._config = config or TogetherConfig.from_env(model=model)
        api_key = os.getenv("TOGETHER_API_KEY")
        if client is None and not api_key:
            raise ValueError("TOGETHER_API_KEY must be set")

        # Disable SDK retries so all providers use the same observable retry layer.
        self._client = client or AsyncTogether(api_key=api_key, max_retries=0)
        self._retry_policy = RetryPolicy(
            max_retries=self._config.max_retries,
            base_delay_seconds=self._config.retry_base_delay_seconds,
            max_delay_seconds=self._config.retry_max_delay_seconds,
        )
        self._sleep = sleep
        self._jitter = jitter
        self._on_retry = on_retry
        self._closed = False

    def parallelism(self) -> int:
        return self._config.parallelism

    def metadata(self) -> dict[str, object]:
        return {
            "provider": "Together",
            "model": self._config.model,
            "parallelism": self._config.parallelism,
            "max_retries": self._config.max_retries,
            "retry_base_delay_seconds": self._config.retry_base_delay_seconds,
            "retry_max_delay_seconds": self._config.retry_max_delay_seconds,
            "max_output_tokens": self._config.max_output_tokens,
        }

    async def ask_generic_question(
        self, system_prompt: str, question: str, temperature: float
    ) -> LLM.SimpleResponse:
        if self._closed:
            raise RuntimeError("Together client is closed")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")

        async def create_completion() -> Any:
            request: dict[str, Any] = {
                "model": self._config.model,
                "messages": [
                    MessageChatCompletionSystemMessageParam(
                        role="system", content=system_prompt
                    ),
                    MessageChatCompletionUserMessageParam(
                        role="user", content=question
                    ),
                ],
                "temperature": temperature,
            }
            if self._config.max_output_tokens is not None:
                request["max_tokens"] = self._config.max_output_tokens
            return await self._client.chat.completions.create(**request)

        response = await retry_with_backoff(
            create_completion,
            policy=self._retry_policy,
            handled_errors=_HANDLED_ERRORS,
            is_retryable=_is_retryable,
            status_code=_status_code,
            sleep=self._sleep,
            jitter=self._jitter,
            on_retry=self._on_retry,
        )
        input_tokens, output_tokens = _token_counts(response)
        choices = getattr(response, "choices", None) or []
        message = getattr(choices[0], "message", None) if choices else None
        answer = getattr(message, "content", None)
        if not isinstance(answer, str) or not answer:
            raise TogetherResponseError(
                "Together returned no text",
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
        await self._client.close()


def check_together_readiness(model: str | None = None) -> dict[str, object]:
    """Validate local Together configuration without sending a provider request."""
    config = TogetherConfig.from_env(model=model)
    if not os.getenv("TOGETHER_API_KEY"):
        raise ValueError("TOGETHER_API_KEY must be set")
    return {"provider": "together", "model": config.model, "credentials": "found"}


def _token_counts(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    return (
        int(getattr(usage, "prompt_tokens", 0) or 0),
        int(getattr(usage, "completion_tokens", 0) or 0),
    )


def _is_retryable(error: BaseException) -> bool:
    if isinstance(error, (APIConnectionError, *_TRANSPORT_ERRORS)):
        return True
    if isinstance(error, APIStatusError):
        return _status_code(error) in _RETRYABLE_STATUS_CODES
    return False


def _status_code(error: BaseException) -> int | None:
    code = getattr(error, "status_code", None) or getattr(error, "status", None)
    try:
        return int(code) if code is not None else None
    except (TypeError, ValueError):
        return None


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
