from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

import llm.providers as provider_registry
from llm import (
    ProviderOptions,
    add_provider_arguments,
    create_provider,
    provider_names,
)
from llm.gemini import check_gemini_readiness
from llm.together import check_together_readiness


def test_registry_exposes_stable_provider_names() -> None:
    assert provider_names() == ("gemini", "together")


def test_shared_cli_arguments_use_registry_choices() -> None:
    parser = argparse.ArgumentParser()
    add_provider_arguments(parser)

    args = parser.parse_args(
        [
            "--provider",
            "together",
            "--model",
            "organization/model",
            "--max-retries",
            "3",
            "--max-output-tokens",
            "512",
            "--thinking-budget",
            "0",
        ]
    )

    assert vars(args) == {
        "provider": "together",
        "model": "organization/model",
        "max_retries": 3,
        "max_output_tokens": 512,
        "thinking_budget": 0,
    }


def test_gemini_factory_maps_shared_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "project-id")
    captured: dict[str, Any] = {}

    def build(config: Any, *, on_retry: Any) -> object:
        captured.update(config=config, on_retry=on_retry)
        return object()

    observer = lambda _: None
    monkeypatch.setattr(provider_registry, "Gemini", build)

    created = create_provider(
        "gemini",
        ProviderOptions(
            model="gemini-test",
            max_retries=0,
            max_output_tokens=128,
            thinking_budget=0,
            on_retry=observer,
        ),
    )

    assert created is not None
    assert captured["config"].model == "gemini-test"
    assert captured["config"].max_retries == 0
    assert captured["config"].max_output_tokens == 128
    assert captured["config"].thinking_budget == 0
    assert captured["on_retry"] is observer


def test_together_factory_maps_shared_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def build(*, config: Any, on_retry: Any) -> object:
        captured.update(config=config, on_retry=on_retry)
        return object()

    observer = lambda _: None
    monkeypatch.setattr(provider_registry, "Together", build)

    created = create_provider(
        "together",
        ProviderOptions(
            model="organization/model",
            max_retries=4,
            max_output_tokens=256,
            on_retry=observer,
        ),
    )

    assert created is not None
    assert captured["config"].model == "organization/model"
    assert captured["config"].max_retries == 4
    assert captured["config"].max_output_tokens == 256
    assert captured["on_retry"] is observer


@pytest.mark.parametrize("thinking_budget", (0, 10))
def test_provider_factory_rejects_unknown_or_unsupported_options(
    thinking_budget: int,
) -> None:
    with pytest.raises(ValueError, match="unknown provider"):
        create_provider("missing")
    with pytest.raises(ValueError, match="thinking-budget controls"):
        create_provider(
            "together",
            ProviderOptions(
                model="organization/model",
                thinking_budget=thinking_budget,
            ),
        )


def test_gemini_preflight_checks_adc_without_constructing_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "project-id")
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "llm.gemini.google.auth.default",
        lambda **kwargs: calls.append(kwargs) or (object(), "adc-project"),
    )

    readiness = check_gemini_readiness("gemini-test")

    assert readiness == {
        "provider": "gemini",
        "platform": "Vertex AI",
        "project": "project-id",
        "location": "global",
        "model": "gemini-test",
        "credentials": "found",
    }
    assert calls == [
        {"scopes": ["https://www.googleapis.com/auth/cloud-platform"]}
    ]


def test_together_preflight_checks_environment_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TOGETHER_API_KEY", "secret")

    assert check_together_readiness("organization/model") == {
        "provider": "together",
        "model": "organization/model",
        "credentials": "found",
    }


def test_makefile_delegates_provider_specific_behavior() -> None:
    makefile = Path(__file__).parents[1].joinpath("Makefile").read_text()

    assert "PROVIDER ?= gemini" in makefile
    assert "--provider gemini" not in makefile
    assert "GEMINI_MAX_RETRIES" not in makefile
    assert "check-provider: check-python" in makefile
