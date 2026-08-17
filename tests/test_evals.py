from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals import GoldenCase, GoldenDataset, validate_output
from evals.validators import validate_validator_config


GOLDEN_DATASET = Path(__file__).parents[1] / "evals" / "golden_dataset.json"


@pytest.mark.parametrize(
    ("output", "validator"),
    [
        ("answer", {"type": "non_empty"}),
        (
            " POSITIVE\n",
            {"type": "exact_match", "expected": "positive"},
        ),
        (
            "neutral",
            {"type": "one_of", "values": ["positive", "neutral", "negative"]},
        ),
        (
            '{"urgency":"high","category":"data_incident"}',
            {
                "type": "json_equals",
                "expected": {"category": "data_incident", "urgency": "high"},
            },
        ),
        (
            "Search rose while display fell.",
            {"type": "contains_all", "values": ["search", "display"]},
        ),
        (
            "Null spend should be rejected.",
            {"type": "contains_any", "values": ["missing", "null"]},
        ),
        (
            "Check missing and duplicate rows.",
            {"type": "excludes_all", "values": ["grammar", "color"]},
        ),
        (
            "No numbers were supplied.",
            {"type": "regex", "pattern": r"^(?!.*\d).+$"},
        ),
        (
            "Measure every marketing dollar.",
            {"type": "max_words", "max": 4},
        ),
        (
            "Spend rose while acquisition cost fell.",
            {"type": "max_sentences", "max": 1},
        ),
        (
            "CPA fell 5% vs. last week.",
            {"type": "max_sentences", "max": 1},
        ),
        (
            "Spend rose 12%, conversions rose 18%, and CPA fell 5%.",
            {
                "type": "percentage_facts",
                "expected": ["12%", "18%", "5%"],
            },
        ),
        (
            "Spend rose +12%, conversions rose +18%, and CPA fell -5%.",
            {
                "type": "percentage_facts",
                "expected": ["12%", "18%", "5%"],
            },
        ),
        (
            "Spend rose 12%, conversions rose 18%, and CPA fell 5%; "
            "the 12% rise drove the gain.",
            {
                "type": "percentage_facts",
                "expected": ["12%", "18%", "5%"],
            },
        ),
        (
            "Research improved this quarter.",
            {"type": "excludes_all", "values": ["search"]},
        ),
        (
            "The colorful banner performed well.",
            {"type": "excludes_all", "values": ["color"]},
        ),
    ],
)
def test_deterministic_validator_passes(
    output: str, validator: dict[str, object]
) -> None:
    validate_validator_config(validator)

    result = validate_output(output, (validator,))[0]

    assert result.passed, result.message


@pytest.mark.parametrize(
    ("output", "validator"),
    [
        (" ", {"type": "non_empty"}),
        ("negative", {"type": "exact_match", "expected": "positive"}),
        ("```json\n{}\n```", {"type": "json_equals", "expected": {}}),
        ("Search improved.", {"type": "contains_all", "values": ["display"]}),
        (
            "Research improved this quarter.",
            {"type": "contains_all", "values": ["search"]},
        ),
        ("Use a color check.", {"type": "excludes_all", "values": ["color"]}),
        (
            "Spend rose 12%, 18%, and 5%, and margin rose 7%.",
            {
                "type": "percentage_facts",
                "expected": ["12%", "18%", "5%"],
            },
        ),
        ("one two three", {"type": "max_words", "max": 2}),
        ("One. Two.", {"type": "max_sentences", "max": 1}),
        (
            "Spend rose 12%, conversions rose 19%, and CPA fell 5%.",
            {
                "type": "percentage_facts",
                "expected": ["12%", "18%", "5%"],
            },
        ),
    ],
)
def test_deterministic_validator_fails(
    output: str, validator: dict[str, object]
) -> None:
    validate_validator_config(validator)

    result = validate_output(output, (validator,))[0]

    assert not result.passed
    assert result.message


def test_checked_in_golden_dataset_is_valid_and_versioned() -> None:
    dataset = GoldenDataset.load(GOLDEN_DATASET)

    assert dataset.version == 2
    assert len(dataset.cases) == 10
    assert len({case.id for case in dataset.cases}) == len(dataset.cases)
    assert {case.category for case in dataset.cases} == {
        "classification",
        "groundedness",
        "instruction-following",
        "structured-output",
        "summarization",
    }
    assert len(dataset.sha256) == 64


@pytest.mark.parametrize(
    "output",
    [
        "Reject missing spend and duplicate campaign-date rows.",
        "Check spend completeness and duplicate campaign-date rows.",
        "Ensure that all expected spend is present and reject duplicate rows.",
        "Flag absent spend values and duplicate campaign-date rows.",
    ],
)
def test_data_quality_case_accepts_completeness_paraphrases(output: str) -> None:
    case = _golden_case("data-quality-checks-001")

    results = validate_output(output, case.validators)

    assert all(result.passed for result in results), results


@pytest.mark.parametrize(
    "output",
    [
        "Check that spend is positive and reject duplicate campaign-date rows.",
        "Validate spend formatting and duplicate campaign-date rows.",
    ],
)
def test_data_quality_case_rejects_outputs_without_completeness_check(
    output: str,
) -> None:
    case = _golden_case("data-quality-checks-001")

    results = validate_output(output, case.validators)

    assert not results[1].passed


def _golden_case(case_id: str) -> GoldenCase:
    dataset = GoldenDataset.load(GOLDEN_DATASET)
    return next(case for case in dataset.cases if case.id == case_id)


def test_golden_dataset_rejects_unknown_validator(tmp_path: Path) -> None:
    dataset_path = tmp_path / "invalid.json"
    dataset_path.write_text(
        json.dumps(
            {
                "name": "invalid",
                "version": 1,
                "description": "invalid test dataset",
                "cases": [
                    {
                        "id": "case",
                        "category": "test",
                        "system_prompt": "system",
                        "input": "input",
                        "validators": [{"type": "subjective_magic"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported validator"):
        GoldenDataset.load(dataset_path)
