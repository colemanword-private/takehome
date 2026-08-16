from __future__ import annotations

import argparse
import asyncio
import json
import platform
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals import GoldenCase, GoldenDataset, validate_output
from llm import (
    LLM,
    ProviderOptions,
    add_provider_arguments,
    create_provider,
)


@dataclass(frozen=True)
class CaseResult:
    id: str
    category: str
    passed: bool
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    output: str | None
    validations: tuple[dict[str, object], ...]
    error_type: str | None = None
    error_message: str | None = None


async def evaluate_dataset(
    provider: LLM,
    dataset: GoldenDataset,
    *,
    concurrency: int = 4,
) -> dict[str, Any]:
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")

    # Keep evaluation traffic bounded independently from the load-test settings.
    semaphore = asyncio.Semaphore(min(concurrency, len(dataset.cases)))

    async def evaluate(case: GoldenCase) -> CaseResult:
        async with semaphore:
            return await _evaluate_case(provider, case)

    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    results = await asyncio.gather(*(evaluate(case) for case in dataset.cases))
    elapsed_seconds = time.perf_counter() - started

    passed_count = sum(result.passed for result in results)
    category_counts: dict[str, Counter[str]] = {}
    for result in results:
        counts = category_counts.setdefault(result.category, Counter())
        counts["passed" if result.passed else "failed"] += 1

    return {
        "started_at": started_at.isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "dataset": dataset.metadata(),
        "provider": provider.metadata(),
        "runtime": {"python": platform.python_version()},
        "cases": len(results),
        "passed": passed_count,
        "failed": len(results) - passed_count,
        "pass_rate": passed_count / len(results),
        "observed_input_tokens": sum(result.input_tokens for result in results),
        "observed_output_tokens": sum(result.output_tokens for result in results),
        "by_category": {
            category: dict(counts)
            for category, counts in sorted(category_counts.items())
        },
        "results": [asdict(result) for result in results],
    }


async def _evaluate_case(provider: LLM, case: GoldenCase) -> CaseResult:
    started = time.perf_counter()
    try:
        response = await provider.ask_generic_question(
            case.system_prompt,
            case.input,
            case.temperature,
        )
    except Exception as error:
        # Provider failures are quality failures but remain distinct from rule failures.
        return CaseResult(
            id=case.id,
            category=case.category,
            passed=False,
            latency_seconds=time.perf_counter() - started,
            input_tokens=int(getattr(error, "input_tokens", 0) or 0),
            output_tokens=int(getattr(error, "output_tokens", 0) or 0),
            output=None,
            validations=(),
            error_type=type(error).__name__,
            error_message=str(error),
        )

    validations = validate_output(response.answer, case.validators)
    return CaseResult(
        id=case.id,
        category=case.category,
        passed=all(validation.passed for validation in validations),
        latency_seconds=time.perf_counter() - started,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        output=response.answer,
        validations=tuple(asdict(validation) for validation in validations),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an LLM provider against deterministic golden cases."
    )
    add_provider_arguments(parser)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).with_name("evals") / "golden_dataset.json",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


async def _main() -> int:
    args = _parse_args()
    dataset = GoldenDataset.load(args.dataset)
    provider = create_provider(
        args.provider,
        ProviderOptions(
            model=args.model,
            max_retries=args.max_retries,
            max_output_tokens=args.max_output_tokens,
            thinking_budget=args.thinking_budget,
        ),
    )
    try:
        report = await evaluate_dataset(
            provider,
            dataset,
            concurrency=args.concurrency,
        )
    finally:
        await provider.close()

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")
    return 0 if report["failed"] == 0 else 1

if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
