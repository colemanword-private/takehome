from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from together import RateLimitError

from llm import RetryEvent, Together, TogetherConfig, TogetherResponseError


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


@pytest.mark.asyncio
async def test_together_uses_shared_retry_layer_and_output_limit() -> None:
    request = httpx.Request("POST", "https://api.together.xyz/v1/chat/completions")
    rate_limited = RateLimitError(
        "busy",
        response=httpx.Response(429, request=request),
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
    provider = Together(
        config=TogetherConfig(
            model="organization/model",
            max_retries=1,
            retry_base_delay_seconds=0,
            max_output_tokens=128,
        ),
        client=client,  # type: ignore[arg-type]
        on_retry=retries.append,
    )

    response = await provider.ask_generic_question("system", "question", 0.3)

    assert response.answer == "answer"
    assert client.completions.calls == 2
    assert client.completions.request["max_tokens"] == 128
    assert retries == [
        RetryEvent(
            attempt=1,
            delay_seconds=0,
            status_code=429,
            error_type="RateLimitError",
        )
    ]


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
