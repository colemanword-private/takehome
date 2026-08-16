from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from typing import Callable

from .gemini import Gemini, GeminiConfig, check_gemini_readiness
from .llm import LLM
from .retry import RetryEvent
from .together import Together, TogetherConfig, check_together_readiness

RetryObserver = Callable[[RetryEvent], None]


@dataclass(frozen=True)
class ProviderOptions:
    """Provider-neutral controls accepted by experiment entry points."""

    model: str | None = None
    max_retries: int | None = None
    max_output_tokens: int | None = None
    thinking_budget: int | None = None
    on_retry: RetryObserver | None = None


@dataclass(frozen=True)
class _ProviderSpec:
    name: str
    build: Callable[[ProviderOptions], LLM]
    check_readiness: Callable[[str | None], dict[str, object]]


def _build_gemini(options: ProviderOptions) -> LLM:
    config = GeminiConfig.from_env()
    overrides = {
        name: value
        for name, value in (
            ("model", options.model),
            ("max_retries", options.max_retries),
            ("max_output_tokens", options.max_output_tokens),
            ("thinking_budget", options.thinking_budget),
        )
        if value is not None
    }
    return Gemini(replace(config, **overrides), on_retry=options.on_retry)


def _build_together(options: ProviderOptions) -> LLM:
    if options.thinking_budget is not None:
        raise ValueError("Together does not support thinking-budget controls")

    config = TogetherConfig.from_env(model=options.model)
    overrides = {
        name: value
        for name, value in (
            ("max_retries", options.max_retries),
            ("max_output_tokens", options.max_output_tokens),
        )
        if value is not None
    }
    return Together(
        config=replace(config, **overrides),
        on_retry=options.on_retry,
    )


_PROVIDERS = {
    "gemini": _ProviderSpec("gemini", _build_gemini, check_gemini_readiness),
    "together": _ProviderSpec(
        "together", _build_together, check_together_readiness
    ),
}


def provider_names() -> tuple[str, ...]:
    """Return stable CLI names for every registered provider."""
    return tuple(sorted(_PROVIDERS))


def create_provider(
    name: str,
    options: ProviderOptions | None = None,
) -> LLM:
    """Build a registered provider from shared experiment controls."""
    return _provider_spec(name).build(options or ProviderOptions())


def check_provider_readiness(
    name: str,
    *,
    model: str | None = None,
) -> dict[str, object]:
    """Run a provider-owned local preflight without sending an inference request."""
    return _provider_spec(name).check_readiness(model)


def add_provider_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the same provider selection and experiment controls to a CLI."""
    parser.add_argument(
        "--provider",
        choices=provider_names(),
        default="gemini",
        help="Registered provider to exercise (default: gemini).",
    )
    parser.add_argument(
        "--model",
        help="Model ID (defaults to the selected provider's environment setting).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        help="Override retries after the initial attempt.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        help="Override the provider's response-token limit.",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        help="Override thinking tokens when the provider supports them.",
    )


def _provider_spec(name: str) -> _ProviderSpec:
    try:
        return _PROVIDERS[name]
    except KeyError as error:
        choices = ", ".join(provider_names())
        raise ValueError(
            f"unknown provider {name!r}; choose one of: {choices}"
        ) from error
