from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from evals import GoldenDataset
from llm import LLM
from quality_eval import evaluate_dataset

pytestmark = pytest.mark.asyncio

GOLDEN_DATASET = Path(__file__).parents[1] / "evals" / "golden_dataset.json"


class FakeQualityProvider(LLM):
    def __init__(self, outputs: dict[str, str | BaseException]) -> None:
        self._outputs = outputs
        self.active = 0
        self.max_active = 0

    def parallelism(self) -> int:
        return 100

    def metadata(self) -> dict[str, object]:
        return {"provider": "fake-quality"}

    async def ask_generic_question(
        self, system_prompt: str, question: str, temperature: float
    ) -> LLM.SimpleResponse:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.001)
            output = self._outputs[question]
            if isinstance(output, BaseException):
                raise output
            return LLM.SimpleResponse(output, input_tokens=5, output_tokens=2)
        finally:
            self.active -= 1


async def test_quality_evaluation_reports_validator_and_provider_failures() -> None:
    dataset = GoldenDataset.load(GOLDEN_DATASET)
    outputs: dict[str, str | BaseException] = {
        case.input: _passing_output(case.id) for case in dataset.cases
    }
    outputs[dataset.cases[0].input] = "negative"
    outputs[dataset.cases[1].input] = RuntimeError("provider unavailable")
    provider = FakeQualityProvider(outputs)

    report = await evaluate_dataset(provider, dataset, concurrency=3)

    assert provider.max_active == 3
    assert report["cases"] == 10
    assert report["passed"] == 8
    assert report["failed"] == 2
    assert report["pass_rate"] == 0.8
    assert report["observed_input_tokens"] == 45
    assert report["observed_output_tokens"] == 18
    assert report["dataset"]["sha256"] == dataset.sha256
    assert report["provider"] == {"provider": "fake-quality"}

    validator_failure = report["results"][0]
    assert validator_failure["error_type"] is None
    assert validator_failure["validations"][0]["passed"] is False

    provider_failure = report["results"][1]
    assert provider_failure["error_type"] == "RuntimeError"
    assert provider_failure["validations"] == ()


async def test_quality_evaluation_rejects_invalid_concurrency() -> None:
    dataset = GoldenDataset.load(GOLDEN_DATASET)

    with pytest.raises(ValueError, match="concurrency"):
        await evaluate_dataset(FakeQualityProvider({}), dataset, concurrency=0)


def _passing_output(case_id: str) -> str:
    outputs = {
        "sentiment-positive-001": "positive",
        "sentiment-neutral-001": "neutral",
        "sentiment-negative-001": "negative",
        "incident-json-001": '{"category":"data_incident","urgency":"high"}',
        "incident-json-002": '{"category":"feature_request","urgency":"low"}',
        "campaign-summary-001": (
            "Spend rose 12%, conversions rose 18%, and cost per acquisition fell 5%."
        ),
        "campaign-summary-002": "Search improved, display declined, and social was unchanged.",
        "headline-rewrite-001": "See Every Marketing Dollar Perform",
        "data-quality-checks-001": "Reject missing spend and duplicate campaign-date rows.",
        "no-invented-facts-001": "unavailable",
    }
    return outputs[case_id]
