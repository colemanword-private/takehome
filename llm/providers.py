from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from typing import Callable

from .gemini import Gemini, GeminiConfig, check_gemini_readiness
from .llm import LLM
from .retry import BackoffEvent, RetryEvent, RetryExhaustedEvent
from .together import Together, TogetherConfig, check_together_readiness

BackoffObserver = Callable[[BackoffEvent], None]
RetryObserver = Callable[[RetryEvent], None]
RetryExhaustedObserver = Callable[[RetryExhaustedEvent], None]


@dataclass(frozen=True)
class ProviderOptions:
    """Provider-neutral controls accepted by experiment entry points."""

    model: str | None = None
    max_retries: int | None = None
    max_output_tokens: int | None = None
    thinking_budget: int | None = None
    on_retry: RetryObserver | None = None
    on_backoff: BackoffObserver | None = None
    on_exhausted: RetryExhaustedObserver | None = None


# Optional tuning fields on ProviderOptions that providers may not all support.
_TUNING_CONTROLS = ("model", "max_retries", "max_output_tokens", "thinking_budget")


@dataclass(frozen=True)
class _ProviderSpec:
    name: str
    build: Callable[[ProviderOptions], LLM]
    check_readiness: Callable[[str | None], dict[str, object]]
    # Declared capabilities let the registry reject unsupported controls
    # uniformly instead of each builder raising its own ad hoc error.
    controls: frozenset[str]


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
    return Gemini(
        replace(config, **overrides),
        on_backoff=options.on_backoff,
        on_retry=options.on_retry,
        on_exhausted=options.on_exhausted,
    )


def _build_together(options: ProviderOptions) -> LLM:
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
        on_backoff=options.on_backoff,
        on_retry=options.on_retry,
        on_exhausted=options.on_exhausted,
    )


_PROVIDERS = {
    "gemini": _ProviderSpec(
        "gemini",
        _build_gemini,
        check_gemini_readiness,
        controls=frozenset(_TUNING_CONTROLS),
    ),
    "together": _ProviderSpec(
        "together",
        _build_together,
        check_together_readiness,
        controls=frozenset(_TUNING_CONTROLS) - {"thinking_budget"},
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
    spec = _provider_spec(name)
    resolved = options or ProviderOptions()
    _validate_controls(spec, resolved)
    return spec.build(resolved)


def _validate_controls(spec: _ProviderSpec, options: ProviderOptions) -> None:
    for control in _TUNING_CONTROLS:
        if getattr(options, control) is not None and control not in spec.controls:
            raise ValueError(
                f"{spec.name} does not support the "
                f"{control.replace('_', '-')} control"
            )


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
