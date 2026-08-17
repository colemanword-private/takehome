"""Shared Gemini test fakes: an outcome-scripted client and response builder.

An outcome may be a response object (returned), an exception (raised), or an
async callable (awaited), letting tests script success, fault, and stall
sequences through the real provider code.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from llm import GeminiConfig


class FakeModels:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes
        self.calls: list[dict[str, Any]] = []

    async def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        if callable(outcome):
            return await outcome()
        return outcome


class FakeAsyncClient:
    def __init__(self, outcomes: list[Any]) -> None:
        self.models = FakeModels(outcomes)
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class FakeClient:
    def __init__(self, outcomes: list[Any]) -> None:
        self.aio = FakeAsyncClient(outcomes)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def response(
    text: str | None = "answer",
    *,
    prompt_tokens: int = 5,
    candidate_tokens: int = 6,
    thought_tokens: int = 2,
    total_tokens: int | None = 13,
) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt_tokens,
            candidates_token_count=candidate_tokens,
            thoughts_token_count=thought_tokens,
            total_token_count=total_tokens,
        ),
        candidates=[],
        prompt_feedback=None,
    )


def config(**overrides: Any) -> GeminiConfig:
    values: dict[str, Any] = {
        "project": "project-id",
        "max_retries": 2,
        "retry_base_delay_seconds": 0.5,
        "retry_max_delay_seconds": 2.0,
    }
    values.update(overrides)
    return GeminiConfig(**values)
