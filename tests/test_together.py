from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from together import RateLimitError

from llm import (
    RetryDeadlineExceeded,
    RetryEvent,
    RetryExhaustedEvent,
    Together,
    TogetherConfig,
    TogetherResponseError,
)


class FakeCompletions:
    def __init__(self) -> None:
        self.request: dict[str, Any] | None = None

    async def create(self, **request: Any) -> SimpleNamespace:
        self.request = request
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=4),
        )


class FakeTogetherClient:
    def __init__(self) -> None:
        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_together_uses_explicit_model_and_reusable_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    client = FakeTogetherClient()
    provider = Together(model="organization/model", client=client)  # type: ignore[arg-type]

    response = await provider.ask_generic_question("system", "question", 0.3)
    await provider.close()

    assert response.answer == "answer"
    assert response.input_tokens == 11
    assert response.output_tokens == 4
    assert client.completions.request == {
        "model": "organization/model",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
        ],
        "temperature": 0.3,
    }
    assert provider.metadata()["model"] == "organization/model"
    assert provider.metadata()["request_deadline_seconds"] == 60.0
    assert client.closed is True


def test_together_validates_missing_model_and_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TOGETHER_MODEL", raising=False)
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)

    with pytest.raises(ValueError, match="TOGETHER_MODEL"):
        Together(client=FakeTogetherClient())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="TOGETHER_API_KEY"):
        Together(model="organization/model")


def test_together_reads_request_hardening_controls_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TOGETHER_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("TOGETHER_REQUEST_DEADLINE_SECONDS", "30")
    monkeypatch.setenv("TOGETHER_RETRY_BUDGET_CAPACITY", "7")
    monkeypatch.setenv("TOGETHER_RETRY_BUDGET_REFILL_PER_SECOND", "1.5")

    configured = TogetherConfig.from_env(model="organization/model")

    assert configured.timeout_seconds == 12.5
    assert configured.request_deadline_seconds == 30
    assert configured.retry_budget_capacity == 7
    assert configured.retry_budget_refill_per_second == 1.5


@pytest.mark.asyncio
async def test_together_uses_shared_retry_layer_and_output_limit() -> None:
    request = httpx.Request("POST", "https://api.together.xyz/v1/chat/completions")
    rate_limited = RateLimitError(
        "busy",
        response=httpx.Response(
            429,
            request=request,
            headers={"Retry-After": "1.25"},
        ),
        body={"error": "busy"},
    )

    class RetryingCompletions(FakeCompletions):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        async def create(self, **request: Any) -> SimpleNamespace:
            self.request = request
            self.calls += 1
            if self.calls == 1:
                raise rate_limited
            return await super().create(**request)

    client = FakeTogetherClient()
    client.completions = RetryingCompletions()
    client.chat = SimpleNamespace(completions=client.completions)
    retries: list[RetryEvent] = []
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    provider = Together(
        config=TogetherConfig(
            model="organization/model",
            max_retries=1,
            retry_base_delay_seconds=0,
            max_output_tokens=128,
        ),
        client=client,  # type: ignore[arg-type]
        sleep=record_sleep,
        on_retry=retries.append,
    )

    response = await provider.ask_generic_question("system", "question", 0.3)

    assert response.answer == "answer"
    assert client.completions.calls == 2
    assert client.completions.request["max_tokens"] == 128
    assert delays == [1.25]
    assert retries == [
        RetryEvent(
            attempt=1,
            delay_seconds=1.25,
            status_code=429,
            error_type="RateLimitError",
            retry_after_seconds=1.25,
            cumulative_backoff_seconds=1.25,
        )
    ]


@pytest.mark.asyncio
async def test_together_enforces_total_request_deadline() -> None:
    class HangingCompletions:
        def __init__(self) -> None:
            self.calls = 0

        async def create(self, **_: Any) -> SimpleNamespace:
            self.calls += 1
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    client = FakeTogetherClient()
    client.completions = HangingCompletions()
    client.chat = SimpleNamespace(completions=client.completions)
    provider = Together(
        config=TogetherConfig(
            model="organization/model",
            request_deadline_seconds=0.01,
        ),
        client=client,  # type: ignore[arg-type]
    )

    with pytest.raises(RetryDeadlineExceeded) as raised:
        await provider.ask_generic_question("system", "question", 0.3)

    assert client.completions.calls == 1
    assert raised.value.event.reason == "deadline"


@pytest.mark.asyncio
async def test_together_retries_strict_per_attempt_timeouts() -> None:
    class HangingCompletions:
        def __init__(self) -> None:
            self.calls = 0

        async def create(self, **_: Any) -> SimpleNamespace:
            self.calls += 1
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    client = FakeTogetherClient()
    client.completions = HangingCompletions()
    client.chat = SimpleNamespace(completions=client.completions)
    provider = Together(
        config=TogetherConfig(
            model="organization/model",
            timeout_seconds=0.01,
            request_deadline_seconds=0.1,
            max_retries=1,
            retry_base_delay_seconds=0,
        ),
        client=client,  # type: ignore[arg-type]
        sleep=lambda _: asyncio.sleep(0),
    )

    with pytest.raises(TimeoutError):
        await provider.ask_generic_question("system", "question", 0.3)

    assert client.completions.calls == 2


@pytest.mark.asyncio
async def test_together_shared_retry_budget_stops_retries() -> None:
    request = httpx.Request("POST", "https://api.together.xyz/v1/chat/completions")
    rate_limited = RateLimitError(
        "busy",
        response=httpx.Response(429, request=request),
        body={"error": "busy"},
    )

    class FailingCompletions:
        def __init__(self) -> None:
            self.calls = 0

        async def create(self, **_: Any) -> SimpleNamespace:
            self.calls += 1
            raise rate_limited

    client = FakeTogetherClient()
    client.completions = FailingCompletions()
    client.chat = SimpleNamespace(completions=client.completions)
    exhaustions: list[RetryExhaustedEvent] = []
    provider = Together(
        config=TogetherConfig(
            model="organization/model",
            max_retries=3,
            retry_base_delay_seconds=0,
            retry_budget_capacity=1,
            retry_budget_refill_per_second=0,
        ),
        client=client,  # type: ignore[arg-type]
        sleep=lambda _: asyncio.sleep(0),
        on_exhausted=exhaustions.append,
    )

    with pytest.raises(RateLimitError):
        await provider.ask_generic_question("system", "question", 0.3)

    assert client.completions.calls == 2
    assert exhaustions[0].reason == "retry_budget"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "choices",
    (
        [],
        [SimpleNamespace(message=None)],
        [SimpleNamespace(message=SimpleNamespace(content=None))],
    ),
)
async def test_together_rejects_responses_without_text(choices: list[Any]) -> None:
    class EmptyCompletions:
        async def create(self, **_: Any) -> SimpleNamespace:
            return SimpleNamespace(
                choices=choices,
                usage=SimpleNamespace(prompt_tokens=7, completion_tokens=2),
            )

    client = FakeTogetherClient()
    client.chat = SimpleNamespace(completions=EmptyCompletions())
    provider = Together(
        config=TogetherConfig(model="organization/model"),
        client=client,  # type: ignore[arg-type]
    )

    with pytest.raises(TogetherResponseError, match="no text") as raised:
        await provider.ask_generic_question("system", "question", 0.3)

    assert raised.value.input_tokens == 7
    assert raised.value.output_tokens == 2


@pytest.mark.asyncio
async def test_together_missing_usage_defaults_to_zero() -> None:
    class NoUsageCompletions:
        async def create(self, **_: Any) -> SimpleNamespace:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content="answer"))
                ],
                usage=None,
            )

    client = FakeTogetherClient()
    client.chat = SimpleNamespace(completions=NoUsageCompletions())
    provider = Together(
        config=TogetherConfig(model="organization/model"),
        client=client,  # type: ignore[arg-type]
    )

    response = await provider.ask_generic_question("system", "question", 0.3)

    assert response.input_tokens == 0
    assert response.output_tokens == 0
