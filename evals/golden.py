from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .validators import validate_validator_config


@dataclass(frozen=True)
class GoldenCase:
    id: str
    category: str
    system_prompt: str
    input: str
    validators: tuple[dict[str, Any], ...]
    temperature: float = 0.0


@dataclass(frozen=True)
class GoldenDataset:
    name: str
    version: int
    description: str
    cases: tuple[GoldenCase, ...]
    source: str
    sha256: str

    @classmethod
    def load(cls, path: Path) -> GoldenDataset:
        raw = path.read_bytes()
        document = json.loads(raw)
        if not isinstance(document, dict):
            raise ValueError("golden dataset must be a JSON object")

        name = _required_string(document, "name")
        description = _required_string(document, "description")
        version = document.get("version")
        if not isinstance(version, int) or version < 1:
            raise ValueError("golden dataset version must be a positive integer")

        raw_cases = document.get("cases")
        if not isinstance(raw_cases, list) or not raw_cases:
            raise ValueError("golden dataset cases must be a non-empty list")

        cases = tuple(
            _parse_case(raw_case, index)
            for index, raw_case in enumerate(raw_cases)
        )
        ids = [case.id for case in cases]
        if len(ids) != len(set(ids)):
            raise ValueError("golden dataset case IDs must be unique")

        return cls(
            name=name,
            version=version,
            description=description,
            cases=cases,
            source=path.name,
            # Hash the exact file so every report identifies the evaluated revision.
            sha256=hashlib.sha256(raw).hexdigest(),
        )

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "source": self.source,
            "sha256": self.sha256,
            "case_count": len(self.cases),
        }


def _parse_case(raw_case: object, index: int) -> GoldenCase:
    if not isinstance(raw_case, dict):
        raise ValueError(f"golden case at index {index} must be an object")

    validators = raw_case.get("validators")
    if not isinstance(validators, list) or not validators:
        raise ValueError(f"golden case at index {index} must define validators")
    for validator in validators:
        validate_validator_config(validator)

    temperature = raw_case.get("temperature", 0.0)
    if not isinstance(temperature, (int, float)) or not 0.0 <= temperature <= 2.0:
        raise ValueError(f"golden case at index {index} has invalid temperature")

    return GoldenCase(
        id=_required_string(raw_case, "id", index),
        category=_required_string(raw_case, "category", index),
        system_prompt=_required_string(raw_case, "system_prompt", index),
        input=_required_string(raw_case, "input", index),
        validators=tuple(validators),
        temperature=float(temperature),
    )


def _required_string(
    document: dict[str, Any], key: str, case_index: int | None = None
) -> str:
    value = document.get(key)
    if isinstance(value, str) and value.strip():
        return value
    if case_index is None:
        location = "golden dataset"
    else:
        location = f"golden case at index {case_index}"
    raise ValueError(f"{location} {key} must be a non-empty string")
