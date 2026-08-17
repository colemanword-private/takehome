from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ValidationResult:
    type: str
    passed: bool
    message: str


Validator = Callable[[str, dict[str, Any]], tuple[bool, str]]


def validate_output(
    output: str, validators: tuple[dict[str, Any], ...]
) -> tuple[ValidationResult, ...]:
    results = []
    for config in validators:
        validator_type = config["type"]
        passed, message = _VALIDATORS[validator_type](output, config)
        results.append(
            ValidationResult(type=validator_type, passed=passed, message=message)
        )
    return tuple(results)


def validate_validator_config(config: object) -> None:
    if not isinstance(config, dict):
        raise ValueError("validator must be an object")
    validator_type = config.get("type")
    if validator_type not in _VALIDATORS:
        raise ValueError(f"unsupported validator type: {validator_type!r}")
    _CONFIG_VALIDATORS[validator_type](config)


def _non_empty(output: str, _: dict[str, Any]) -> tuple[bool, str]:
    passed = bool(output.strip())
    return passed, "output is non-empty" if passed else "output is empty"


def _exact_match(output: str, config: dict[str, Any]) -> tuple[bool, str]:
    actual = _normalize(output, config)
    expected = _normalize(config["expected"], config)
    passed = actual == expected
    return passed, f"expected {expected!r}; got {actual!r}"


def _one_of(output: str, config: dict[str, Any]) -> tuple[bool, str]:
    actual = _normalize(output, config)
    expected = [_normalize(value, config) for value in config["values"]]
    passed = actual in expected
    return passed, f"expected one of {expected!r}; got {actual!r}"


def _json_equals(output: str, config: dict[str, Any]) -> tuple[bool, str]:
    try:
        # Parsing the entire output intentionally rejects Markdown code fences.
        actual = json.loads(output)
    except json.JSONDecodeError as error:
        return False, f"output is not valid JSON: {error.msg}"
    expected = config["expected"]
    passed = actual == expected
    return passed, f"expected JSON {expected!r}; got {actual!r}"


def _contains_all(output: str, config: dict[str, Any]) -> tuple[bool, str]:
    expected = config["values"]
    matches = _matching_values(output, config)
    missing = [value for value in expected if value not in matches]
    passed = not missing
    return passed, "all required text is present" if passed else f"missing {missing!r}"


def _contains_any(output: str, config: dict[str, Any]) -> tuple[bool, str]:
    expected = config["values"]
    matches = _matching_values(output, config)
    passed = bool(matches)
    return passed, f"matched {matches!r}" if passed else f"none of {expected!r} found"


def _excludes_all(output: str, config: dict[str, Any]) -> tuple[bool, str]:
    matches = _matching_values(output, config)
    passed = not matches
    return passed, "no forbidden text is present" if passed else f"found {matches!r}"


def _regex(output: str, config: dict[str, Any]) -> tuple[bool, str]:
    flags = 0 if config.get("case_sensitive", False) else re.IGNORECASE
    passed = re.search(config["pattern"], output, flags) is not None
    return passed, f"output {'matches' if passed else 'does not match'} /{config['pattern']}/"


def _max_words(output: str, config: dict[str, Any]) -> tuple[bool, str]:
    count = len(re.findall(r"\b[\w'-]+\b", output, re.UNICODE))
    limit = config["max"]
    passed = count <= limit
    return passed, f"word count {count}; maximum {limit}"


def _max_sentences(output: str, config: dict[str, Any]) -> tuple[bool, str]:
    # A boundary requires a following capitalized word so abbreviations such as
    # "vs." do not fail a correct one-sentence answer. Undercounting is the safer
    # error direction for a maximum check.
    count = len(re.findall(r"[.!?]+(?=\s+[A-Z]|\s*$)", output.strip()))
    if output.strip() and count == 0:
        count = 1
    limit = config["max"]
    passed = count <= limit
    return passed, f"sentence count {count}; maximum {limit}"


def _percentage_facts(output: str, config: dict[str, Any]) -> tuple[bool, str]:
    # Signs are stripped and repeats collapsed so "+12%" or a restated figure
    # still counts as the same fact; distinct invented percentages still fail.
    actual = {
        match.lstrip("+-")
        for match in re.findall(r"(?<!\w)[+-]?\d+(?:\.\d+)?%", output)
    }
    expected = {value.lstrip("+-") for value in config["expected"]}
    allow_additional = config.get("allow_additional", False)
    if allow_additional:
        missing = sorted(expected - actual)
        passed = not missing
        message = "all percentage facts are present" if passed else f"missing {missing!r}"
        return passed, message
    passed = actual == expected
    return passed, f"expected percentages {sorted(expected)!r}; got {sorted(actual)!r}"


def _normalize(value: str, config: dict[str, Any]) -> str:
    if config.get("normalize_whitespace", True):
        normalized = " ".join(value.split())
    else:
        normalized = value
    return normalized if config.get("case_sensitive", False) else normalized.casefold()


def _matching_values(output: str, config: dict[str, Any]) -> list[str]:
    # Word-bounded matching prevents "search" from matching inside "Research"
    # or an excluded "color" from flagging "colorful".
    case_sensitive = config.get("case_sensitive", False)
    haystack = output if case_sensitive else output.casefold()
    return [
        value
        for value in config["values"]
        if re.search(
            rf"(?<!\w){re.escape(value if case_sensitive else value.casefold())}(?!\w)",
            haystack,
        )
    ]


def _require_string(config: dict[str, Any], key: str) -> None:
    if not isinstance(config.get(key), str):
        raise ValueError(f"{config['type']} validator {key} must be a string")


def _require_string_list(config: dict[str, Any], key: str) -> None:
    values = config.get(key)
    valid = (
        isinstance(values, list)
        and bool(values)
        and all(isinstance(value, str) for value in values)
    )
    if not valid:
        raise ValueError(
            f"{config['type']} validator {key} must be a non-empty string list"
        )


def _require_positive_int(config: dict[str, Any], key: str) -> None:
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{config['type']} validator {key} must be a positive integer")


def _no_config(_: dict[str, Any]) -> None:
    return None


def _exact_match_config(config: dict[str, Any]) -> None:
    _require_string(config, "expected")


def _values_config(config: dict[str, Any]) -> None:
    _require_string_list(config, "values")


def _json_equals_config(config: dict[str, Any]) -> None:
    if "expected" not in config:
        raise ValueError("json_equals validator must define expected")


def _regex_config(config: dict[str, Any]) -> None:
    _require_string(config, "pattern")
    try:
        re.compile(config["pattern"])
    except re.error as error:
        raise ValueError(f"regex validator has invalid pattern: {error}") from error


def _max_config(config: dict[str, Any]) -> None:
    _require_positive_int(config, "max")


def _percentage_config(config: dict[str, Any]) -> None:
    _require_string_list(config, "expected")


_VALIDATORS: dict[str, Validator] = {
    "non_empty": _non_empty,
    "exact_match": _exact_match,
    "one_of": _one_of,
    "json_equals": _json_equals,
    "contains_all": _contains_all,
    "contains_any": _contains_any,
    "excludes_all": _excludes_all,
    "regex": _regex,
    "max_words": _max_words,
    "max_sentences": _max_sentences,
    "percentage_facts": _percentage_facts,
}

_CONFIG_VALIDATORS: dict[str, Callable[[dict[str, Any]], None]] = {
    "non_empty": _no_config,
    "exact_match": _exact_match_config,
    "one_of": _values_config,
    "json_equals": _json_equals_config,
    "contains_all": _values_config,
    "contains_any": _values_config,
    "excludes_all": _values_config,
    "regex": _regex_config,
    "max_words": _max_config,
    "max_sentences": _max_config,
    "percentage_facts": _percentage_config,
}
