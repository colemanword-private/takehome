"""Small deterministic quality check used for the Gemini parameter experiment."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from llm import Gemini, GeminiConfig, LLM, LLMResponseError

DATASET = Path(__file__).with_name("evals") / "golden_dataset.json"


def load_dataset(path: Path = DATASET) -> dict[str, Any]:
    """Load the golden dataset, stamping the raw bytes' SHA-256 so every
    report records exactly which dataset revision it scored against."""
    raw = path.read_bytes()
    dataset = json.loads(raw)
    cases = dataset.get("cases") if isinstance(dataset, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError("dataset must contain a non-empty cases list")
    dataset["source"] = path.name
    dataset["sha256"] = hashlib.sha256(raw).hexdigest()
    return dataset


def validate_output(output: str, rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Evaluate the ten rule types used by the checked-in dataset; an
    unknown type raises rather than silently passing. Text rules compare
    case-insensitively on whitespace-normalized output at word boundaries;
    `percentage_facts` ignores signs and, unless `allow_additional`, rejects
    extra percentages so hallucinated figures fail the case."""
    results = []
    for rule in rules:
        kind = rule["type"]
        normalized = " ".join(output.split()).casefold()
        if kind == "non_empty":
            passed = bool(output.strip())
        elif kind == "exact_match":
            passed = normalized == " ".join(rule["expected"].split()).casefold()
        elif kind == "json_equals":
            try:
                passed = json.loads(output) == rule["expected"]
            except json.JSONDecodeError:
                passed = False
        elif kind in {"contains_all", "contains_any", "excludes_all"}:
            matches = [
                value
                for value in rule["values"]
                if re.search(rf"(?<!\w){re.escape(value.casefold())}(?!\w)", normalized)
            ]
            if kind == "contains_all":
                passed = len(matches) == len(rule["values"])
            elif kind == "contains_any":
                passed = bool(matches)
            else:
                passed = not matches
        elif kind == "regex":
            passed = re.search(rule["pattern"], output, re.IGNORECASE) is not None
        elif kind == "max_words":
            passed = len(re.findall(r"\b[\w'-]+\b", output)) <= rule["max"]
        elif kind == "max_sentences":
            boundaries = re.findall(r"[.!?]+(?=\s+[A-Z]|\s*$)", output.strip())
            passed = (len(boundaries) or bool(output.strip())) <= rule["max"]
        elif kind == "percentage_facts":
            actual = {
                value.lstrip("+-")
                for value in re.findall(r"(?<!\w)[+-]?\d+(?:\.\d+)?%", output)
            }
            expected = {value.lstrip("+-") for value in rule["expected"]}
            passed = expected <= actual and (
                rule.get("allow_additional", False) or actual == expected
            )
        else:
            raise ValueError(f"unsupported validator: {kind}")
        results.append({"type": kind, "passed": passed})
    return results


async def evaluate_dataset(
    provider: LLM, dataset: dict[str, Any], concurrency: int = 4
) -> dict[str, Any]:
    """Score every case concurrently and return the full report.

    Fails closed: a provider error marks its case failed (recording any
    billable usage the error carried) instead of aborting the run, so one
    transport fault cannot void a whole experiment cell.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    cases = dataset["cases"]
    semaphore = asyncio.Semaphore(concurrency)

    async def evaluate(case: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            async with semaphore:
                response = await provider.ask_generic_question(
                    case["system_prompt"], case["input"], case.get("temperature", 0)
                )
            validations = validate_output(response.answer, case["validators"])
            return {
                "id": case["id"],
                "category": case["category"],
                "passed": all(item["passed"] for item in validations),
                "latency_seconds": time.perf_counter() - started,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "thought_tokens": response.thought_tokens,
                "output": response.answer,
                "validations": validations,
            }
        except Exception as error:
            return {
                "id": case["id"],
                "category": case["category"],
                "passed": False,
                "latency_seconds": time.perf_counter() - started,
                "input_tokens": getattr(error, "input_tokens", 0)
                if isinstance(error, LLMResponseError)
                else 0,
                "output_tokens": getattr(error, "output_tokens", 0)
                if isinstance(error, LLMResponseError)
                else 0,
                "thought_tokens": 0,
                "output": None,
                "validations": [],
                "error_type": type(error).__name__,
                "error_message": str(error),
            }

    started = time.perf_counter()
    results = await asyncio.gather(*(evaluate(case) for case in cases))
    categories: dict[str, Counter[str]] = {}
    for result in results:
        counts = categories.setdefault(result["category"], Counter())
        counts["passed" if result["passed"] else "failed"] += 1
    passed = sum(result["passed"] for result in results)
    return {
        "elapsed_seconds": time.perf_counter() - started,
        "dataset": {
            key: dataset[key]
            for key in ("name", "version", "source", "sha256")
        },
        "provider": provider.metadata(),
        "cases": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": passed / len(results),
        "observed_input_tokens": sum(item["input_tokens"] for item in results),
        "observed_output_tokens": sum(item["output_tokens"] for item in results),
        "observed_thought_tokens": sum(item["thought_tokens"] for item in results),
        "by_category": {key: dict(value) for key, value in sorted(categories.items())},
        "results": results,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--model")
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--max-output-tokens")
    parser.add_argument("--thinking-budget")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


async def _main() -> int:
    args = _parse_args()
    config = GeminiConfig.from_env()
    overrides: dict[str, Any] = {"max_retries": args.max_retries}
    if args.model is not None:
        overrides["model"] = args.model
    # The factorial's Make loop passes "none"/"default" as explicit cell
    # labels; both map to None, which preserves the provider's defaults.
    if args.max_output_tokens is not None:
        overrides["max_output_tokens"] = (
            None if args.max_output_tokens == "none" else int(args.max_output_tokens)
        )
    if args.thinking_budget is not None:
        overrides["thinking_budget"] = (
            None if args.thinking_budget == "default" else int(args.thinking_budget)
        )
    provider = Gemini(replace(config, **overrides))
    try:
        report = await evaluate_dataset(
            provider, load_dataset(args.dataset), args.concurrency
        )
    finally:
        await provider.close()

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return int(report["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
