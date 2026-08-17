from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from google.genai import errors

from conftest import FakeClient, config, response
from llm import (
    Gemini,
    GeminiConfig,
    GeminiResponseError,
)

pytestmark = pytest.mark.asyncio


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
    transport_kwargs: dict[str, Any] = {}

    class RecordingTransport(httpx.AsyncHTTPTransport):
        def __init__(self, **kwargs: Any) -> None:
            transport_kwargs.update(kwargs)
            super().__init__(**kwargs)

    def make_client(**kwargs: Any) -> FakeClient:
        captured.update(kwargs)
        return client

    monkeypatch.setattr("llm.gemini.httpx.AsyncHTTPTransport", RecordingTransport)
    monkeypatch.setattr("llm.gemini.genai.Client", make_client)
    provider = Gemini(
        config(location="us-central1", timeout_seconds=12.5, parallelism=48)
    )

    assert captured["vertexai"] is True
    assert captured["project"] == "project-id"
    assert captured["location"] == "us-central1"
    assert captured["http_options"].api_version == "v1"
    assert captured["http_options"].timeout == 12_500
    assert isinstance(
        captured["http_options"].async_client_args["transport"],
        httpx.AsyncHTTPTransport,
    )
    # The pool is sized to parallelism so httpx's default 20-keepalive limit
    # cannot force TLS re-handshakes under load.
    assert transport_kwargs["limits"].max_connections == 48
    assert transport_kwargs["limits"].max_keepalive_connections == 48
    await provider.close()


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


async def test_reports_blocked_or_empty_response() -> None:
    blocked = response(text=None)
    blocked.prompt_feedback = SimpleNamespace(block_reason="SAFETY")
    client = FakeClient([blocked])
    provider = Gemini(config(), client=client)

    with pytest.raises(GeminiResponseError, match="block_reason=SAFETY") as raised:
        await provider.ask_generic_question("system", "question", 0.0)

    assert raised.value.input_tokens == 5
    assert raised.value.output_tokens == 8


async def test_close_releases_both_clients_and_is_idempotent() -> None:
    client = FakeClient([response()])
    provider = Gemini(config(), client=client)

    await provider.close()
    await provider.close()

    assert client.aio.closed
    assert client.closed
    with pytest.raises(RuntimeError, match="closed"):
        await provider.ask_generic_question("system", "question", 0.0)


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
