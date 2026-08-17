from __future__ import annotations

from typing import Any

import pytest

from llm import LLM
from quality_eval import evaluate_dataset, load_dataset, validate_output


def test_checked_in_dataset_rules_accept_known_good_answers() -> None:
    answers = {
        "sentiment-positive-001": "positive",
        "sentiment-neutral-001": "neutral",
        "sentiment-negative-001": "negative",
        "incident-json-001": '{"category":"data_incident","urgency":"high"}',
        "incident-json-002": '{"category":"feature_request","urgency":"low"}',
        "campaign-summary-001": "Spend rose 12%, conversions rose 18%, and CPA fell 5%.",
        "campaign-summary-002": "Search improved, display declined, and social was unchanged.",
        "headline-rewrite-001": "See Every Marketing Dollar Perform",
        "data-quality-checks-001": "Check for missing spend and duplicate campaign-date rows.",
        "no-invented-facts-001": "unavailable",
    }
    dataset = load_dataset()

    assert dataset["version"] == 2
    assert len(dataset["sha256"]) == 64
    for case in dataset["cases"]:
        results = validate_output(answers[case["id"]], case["validators"])
        assert all(result["passed"] for result in results), case["id"]


class FakeProvider(LLM):
    def __init__(self, answers: list[Any]) -> None:
        self.answers = answers

    def parallelism(self) -> int:
        return 2

    def metadata(self) -> dict[str, object]:
        return {"provider": "fake"}

    async def ask_generic_question(
        self, system_prompt: str, question: str, temperature: float
    ) -> LLM.SimpleResponse:
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return LLM.SimpleResponse(answer, 3, 2, 1)


@pytest.mark.asyncio
async def test_evaluation_separates_quality_and_provider_failures() -> None:
    dataset = {
        "name": "small",
        "version": 1,
        "source": "inline",
        "sha256": "abc",
        "cases": [
            {
                "id": "pass",
                "category": "test",
                "system_prompt": "system",
                "input": "one",
                "validators": [{"type": "exact_match", "expected": "yes"}],
            },
            {
                "id": "error",
                "category": "test",
                "system_prompt": "system",
                "input": "two",
                "validators": [{"type": "non_empty"}],
            },
        ],
    }
    report = await evaluate_dataset(FakeProvider(["yes", RuntimeError("boom")]), dataset)

    assert report["passed"] == 1
    assert report["failed"] == 1
    assert report["observed_output_tokens"] == 2
    assert report["observed_thought_tokens"] == 1
    assert report["results"][1]["error_type"] == "RuntimeError"
