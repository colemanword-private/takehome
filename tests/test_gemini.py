from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from google.genai import errors

from llm import (
    Gemini,
    GeminiConfig,
    GeminiResponseError,
    RetryDeadlineExceeded,
    RetryEvent,
    RetryExhaustedEvent,
)

pytestmark = pytest.mark.asyncio


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
    values = {
        "project": "project-id",
        "max_retries": 2,
        "retry_base_delay_seconds": 0.5,
        "retry_max_delay_seconds": 2.0,
    }
    values.update(overrides)
    return GeminiConfig(**values)


async def test_maps_request_and_includes_thinking_in_output_usage() -> None:
    client = FakeClient([response()])
    provider = Gemini(
        config(max_output_tokens=256, thinking_budget=0),
        client=client,
    )

    result = await provider.ask_generic_question("be concise", "hello", 0.25)

    assert result.answer == "answer"
    assert result.input_tokens == 5
    assert result.output_tokens == 8
    assert result.thought_tokens == 2
    assert provider.parallelism() == 32

    call = client.aio.models.calls[0]
    assert call["model"] == "gemini-2.5-flash"
    assert call["contents"] == "hello"
    assert call["config"].system_instruction == "be concise"
    assert call["config"].temperature == 0.25
    assert call["config"].max_output_tokens == 256
    assert call["config"].thinking_config.thinking_budget == 0


async def test_constructs_vertex_client_with_stable_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient([response()])
    captured: dict[str, Any] = {}

    def make_client(**kwargs: Any) -> FakeClient:
        captured.update(kwargs)
        return client

    monkeypatch.setattr("llm.gemini.genai.Client", make_client)
    provider = Gemini(config(location="us-central1", timeout_seconds=12.5))

    assert captured["vertexai"] is True
    assert captured["project"] == "project-id"
    assert captured["location"] == "us-central1"
    assert captured["http_options"].api_version == "v1"
    assert captured["http_options"].timeout == 12_500
    assert isinstance(
        captured["http_options"].async_client_args["transport"],
        httpx.AsyncHTTPTransport,
    )
    await provider.close()


async def test_transport_pool_is_sized_to_parallelism(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # httpx defaults to 20 keepalive connections; below the configured
    # parallelism that forces TLS re-handshakes under load.
    client = FakeClient([response()])
    captured: dict[str, Any] = {}

    class RecordingTransport(httpx.AsyncHTTPTransport):
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr("llm.gemini.httpx.AsyncHTTPTransport", RecordingTransport)
    monkeypatch.setattr("llm.gemini.genai.Client", lambda **kwargs: client)
    provider = Gemini(config(parallelism=48))

    limits = captured["limits"]
    assert limits.max_connections == 48
    assert limits.max_keepalive_connections == 48
    await provider.close()


async def test_retries_resource_exhausted_with_full_jitter() -> None:
    request = httpx.Request("POST", "https://aiplatform.googleapis.com")
    rate_limited = errors.ClientError(
        429,
        {"error": {"message": "busy"}},
        httpx.Response(429, request=request, headers={"Retry-After": "1.5"}),
    )
    client = FakeClient(
        [
            rate_limited,
            response(),
        ]
    )
    delays: list[float] = []
    retries: list[RetryEvent] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    provider = Gemini(
        config(),
        client=client,
        sleep=record_sleep,
        jitter=lambda low, high: high / 2,
        on_retry=retries.append,
    )

    result = await provider.ask_generic_question("system", "question", 0.0)

    assert result.answer == "answer"
    assert len(client.aio.models.calls) == 2
    assert delays == [1.5]
    assert retries == [
        RetryEvent(
            attempt=1,
            delay_seconds=1.5,
            status_code=429,
            error_type="ClientError",
            retry_after_seconds=1.5,
            cumulative_backoff_seconds=1.5,
        )
    ]


async def test_retries_transient_provider_and_transport_errors() -> None:
    transient_errors = (
        errors.ServerError(503, {"error": {"message": "unavailable"}}),
        httpx.ConnectError("disconnected"),
    )
    for transient_error in transient_errors:
        client = FakeClient([transient_error, response()])
        provider = Gemini(
            config(retry_base_delay_seconds=0, retry_max_delay_seconds=0),
            client=client,
            sleep=lambda _: asyncio.sleep(0),
        )

        result = await provider.ask_generic_question("system", "question", 0.0)

        assert result.answer == "answer"
        assert len(client.aio.models.calls) == 2


async def test_does_not_retry_non_transient_client_error() -> None:
    error = errors.ClientError(400, {"error": {"message": "bad request"}})
    client = FakeClient([error])
    provider = Gemini(config(), client=client)

    with pytest.raises(errors.ClientError) as raised:
        await provider.ask_generic_question("system", "question", 0.0)

    assert raised.value is error
    assert len(client.aio.models.calls) == 1


async def test_retries_provider_timeout_until_budget_is_exhausted() -> None:
    async def too_slow() -> SimpleNamespace:
        await asyncio.sleep(0.05)
        return response()

    client = FakeClient([too_slow, too_slow])
    provider = Gemini(
        config(timeout_seconds=0.001, max_retries=1),
        client=client,
        sleep=lambda _: asyncio.sleep(0),
        jitter=lambda _low, _high: 0.0,
    )

    with pytest.raises(TimeoutError):
        await provider.ask_generic_question("system", "question", 0.0)

    assert len(client.aio.models.calls) == 2


async def test_end_to_end_deadline_bounds_repeated_timeouts() -> None:
    async def too_slow() -> SimpleNamespace:
        await asyncio.Event().wait()
        return response()

    exhausted: list[RetryExhaustedEvent] = []
    client = FakeClient([too_slow])
    provider = Gemini(
        config(
            timeout_seconds=1,
            request_deadline_seconds=0.01,
            max_retries=4,
            retry_base_delay_seconds=0,
        ),
        client=client,
        on_exhausted=exhausted.append,
    )

    with pytest.raises(RetryDeadlineExceeded) as raised:
        await provider.ask_generic_question("system", "question", 0.0)

    assert len(client.aio.models.calls) == 1
    assert raised.value.event.reason == "deadline"
    assert exhausted == [raised.value.event]


async def test_shared_retry_budget_stops_repeated_transient_failures() -> None:
    unavailable = errors.ServerError(
        503, {"error": {"message": "unavailable"}}
    )
    exhausted: list[RetryExhaustedEvent] = []
    client = FakeClient([unavailable, unavailable])
    provider = Gemini(
        config(
            max_retries=4,
            retry_base_delay_seconds=0,
            retry_max_delay_seconds=0,
            retry_budget_capacity=1,
            retry_budget_refill_per_second=0,
        ),
        client=client,
        sleep=lambda _: asyncio.sleep(0),
        on_exhausted=exhausted.append,
    )

    with pytest.raises(errors.ServerError):
        await provider.ask_generic_question("system", "question", 0.0)

    assert len(client.aio.models.calls) == 2
    assert exhausted[0].reason == "retry_budget"


async def test_reports_blocked_or_empty_response() -> None:
    blocked = response(text=None)
    blocked.prompt_feedback = SimpleNamespace(block_reason="SAFETY")
    client = FakeClient([blocked])
    provider = Gemini(config(), client=client)

    with pytest.raises(GeminiResponseError, match="block_reason=SAFETY") as raised:
        await provider.ask_generic_question("system", "question", 0.0)

    assert raised.value.input_tokens == 5
    assert raised.value.output_tokens == 8


async def test_missing_usage_metadata_falls_back_to_zero() -> None:
    client = FakeClient([SimpleNamespace(text="answer", usage_metadata=None)])
    provider = Gemini(config(), client=client)

    result = await provider.ask_generic_question("system", "question", 0.0)

    assert result.input_tokens == 0
    assert result.output_tokens == 0


async def test_close_releases_both_clients_and_is_idempotent() -> None:
    client = FakeClient([response()])
    provider = Gemini(config(), client=client)

    await provider.close()
    await provider.close()

    assert client.aio.closed
    assert client.closed
    with pytest.raises(RuntimeError, match="closed"):
        await provider.ask_generic_question("system", "question", 0.0)


async def test_rejects_invalid_temperature_before_calling_vertex() -> None:
    client = FakeClient([response()])
    provider = Gemini(config(), client=client)

    with pytest.raises(ValueError, match="temperature"):
        await provider.ask_generic_question("system", "question", 2.1)

    assert client.aio.models.calls == []


async def test_environment_rejects_invalid_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid_values = (
        ("GEMINI_PARALLELISM", "not-an-int"),
        ("GEMINI_RETRY_MAX_DELAY_SECONDS", "nan"),
        ("GEMINI_REQUEST_DEADLINE_SECONDS", "0"),
        ("GEMINI_RETRY_BUDGET_CAPACITY", "0"),
        ("GEMINI_RETRY_BUDGET_REFILL_PER_SECOND", "nan"),
    )
    for variable, value in invalid_values:
        with monkeypatch.context() as patch:
            patch.setenv("GOOGLE_CLOUD_PROJECT", "project-id")
            patch.setenv(variable, value)
            with pytest.raises(ValueError, match=variable):
                GeminiConfig.from_env()


async def test_environment_configures_deadline_and_retry_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "project-id")
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "12")
    monkeypatch.setenv("GEMINI_REQUEST_DEADLINE_SECONDS", "45")
    monkeypatch.setenv("GEMINI_RETRY_BUDGET_CAPACITY", "20")
    monkeypatch.setenv("GEMINI_RETRY_BUDGET_REFILL_PER_SECOND", "1.5")

    configured = GeminiConfig.from_env()

    assert configured.timeout_seconds == 12
    assert configured.request_deadline_seconds == 45
    assert configured.retry_budget_capacity == 20
    assert configured.retry_budget_refill_per_second == 1.5
