from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from llm import Gemini, GeminiConfig, LLM, Together, TogetherConfig, provider_names


class FakeGeminiModels:
    async def generate_content(self, **_: Any) -> SimpleNamespace:
        return SimpleNamespace(
            text="contract answer",
            usage_metadata=SimpleNamespace(
                prompt_token_count=3,
                candidates_token_count=2,
                thoughts_token_count=0,
                total_token_count=5,
            ),
        )


class FakeGeminiClient:
    def __init__(self) -> None:
        self.aio = SimpleNamespace(
            models=FakeGeminiModels(),
            aclose=self._close_async,
        )

    async def _close_async(self) -> None:
        return None

    def close(self) -> None:
        return None


class FakeTogetherCompletions:
    async def create(self, **_: Any) -> SimpleNamespace:
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="contract answer"))
            ],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
        )


class FakeTogetherClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=FakeTogetherCompletions())

    async def close(self) -> None:
        return None


@dataclass(frozen=True)
class ProviderCase:
    provider: LLM
    expected_name: str
    expected_model: str


@pytest.fixture(params=provider_names())
def provider_case(request: pytest.FixtureRequest) -> ProviderCase:
    if request.param == "gemini":
        return ProviderCase(
            Gemini(
                GeminiConfig(project="project-id", model="gemini-test"),
                client=FakeGeminiClient(),
            ),
            "Gemini",
            "gemini-test",
        )
    if request.param == "together":
        return ProviderCase(
            Together(
                config=TogetherConfig(model="together-test"),
                client=FakeTogetherClient(),
            ),
            "Together",
            "together-test",
        )
    raise AssertionError(f"add a contract fixture for provider {request.param!r}")


@pytest.mark.asyncio
async def test_registered_provider_implementation_satisfies_llm_contract(
    provider_case: ProviderCase,
) -> None:
    provider = provider_case.provider

    assert provider.parallelism() >= 1
    assert provider.metadata()["provider"] == provider_case.expected_name
    assert provider.metadata()["model"] == provider_case.expected_model

    response = await provider.ask_generic_question("system", "question", 0.2)
    assert response == LLM.SimpleResponse(
        answer="contract answer",
        input_tokens=3,
        output_tokens=2,
    )

    await provider.close()
    await provider.close()
    with pytest.raises(RuntimeError, match="closed"):
        await provider.ask_generic_question("system", "question", 0.2)


def test_provider_response_errors_share_the_declared_base() -> None:
    # Callers recover billable tokens from failures via one declared type, not
    # duck-typed attribute names that each provider happens to share.
    from llm import GeminiResponseError, LLMResponseError, TogetherResponseError

    for error_type in (GeminiResponseError, TogetherResponseError):
        error = error_type("no usable text", input_tokens=3, output_tokens=1)
        assert isinstance(error, LLMResponseError)
        assert (error.input_tokens, error.output_tokens) == (3, 1)


def test_provider_metadata_reports_own_sdk_versions(
    provider_case: ProviderCase,
) -> None:
    # Each provider owns its dependency report so the load harness never
    # hardcodes another provider's packages.
    import importlib.metadata

    dependencies = provider_case.provider.metadata()["dependencies"]
    expected_package = {"Gemini": "google-genai", "Together": "together"}[
        provider_case.expected_name
    ]

    assert expected_package in dependencies
    for name, version in dependencies.items():
        assert version == importlib.metadata.version(name)
