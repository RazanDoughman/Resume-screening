"""The scoring dimensions, loaded from `dimensions.yaml`.

Adding or renaming a scoring dimension used to mean editing the prompt, the
Pydantic models, the CSV column list and the report in four separate places.
This module makes `dimensions.yaml` the single source of truth for that list.

Deliberately dependency-light: it holds plain frozen dataclasses and imports
nothing from the rest of `src`. `models.py` imports `DIMENSIONS` from here to
build its score schema, so anything this module imported back from `models.py`
would be a circular import.

The file is read once, at import time. A malformed `dimensions.yaml` raises
`DimensionConfigError` immediately rather than producing a half-built schema
that only fails later, mid-scoring-run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DIMENSIONS_PATH = Path(__file__).resolve().parent.parent / "dimensions.yaml"

_REQUIRED_FIELDS = ("key", "label", "prompt_description")


class DimensionConfigError(ValueError):
    """Raised when `dimensions.yaml` is missing, malformed, or inconsistent."""


@dataclass(frozen=True)
class Dimension:
    """One scoring dimension.

    `key` is the field name used everywhere downstream — in `ScoreReport`, in
    `results.json`, and as the `{key}_score` / `{key}_reasoning` CSV column
    prefix. `label` is the human-readable name for reports. `prompt_description`
    is the sentence describing this dimension to Claude in the scoring prompt.
    """

    key: str
    label: str
    prompt_description: str


def _require_text(entry: dict, field: str, where: str) -> str:
    """Return `entry[field]` as a non-blank string, or raise."""

    if field not in entry:
        raise DimensionConfigError(f"{where} is missing required field {field!r}.")

    value = entry[field]
    if not isinstance(value, str) or not value.strip():
        raise DimensionConfigError(
            f"{where} has an empty or non-text {field!r} "
            f"(got {value!r}); it must be a non-empty string."
        )

    return value.strip()


def load_dimensions(path: Path = DIMENSIONS_PATH) -> list[Dimension]:
    """Parse and validate a dimensions file into `Dimension` records.

    Order is preserved: it is the order dimensions appear in the prompt and in
    the report, so it is part of the contract, not an implementation detail.
    """

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DimensionConfigError(f"No dimensions file at {path}.") from exc
    except yaml.YAMLError as exc:
        raise DimensionConfigError(f"{path} is not valid YAML: {exc}") from exc

    if not isinstance(raw, list):
        raise DimensionConfigError(
            f"{path} must contain a list of dimensions, got {type(raw).__name__}."
        )

    if not raw:
        raise DimensionConfigError(
            f"{path} defines no dimensions; at least one is required."
        )

    dimensions: list[Dimension] = []
    seen: dict[str, int] = {}

    for index, entry in enumerate(raw):
        where = f"{path}: dimension #{index + 1}"

        if not isinstance(entry, dict):
            raise DimensionConfigError(
                f"{where} must be a mapping with "
                f"{', '.join(_REQUIRED_FIELDS)}, got {type(entry).__name__}."
            )

        key = _require_text(entry, "key", where)
        if key in seen:
            raise DimensionConfigError(
                f"{where} repeats the key {key!r}, already defined by "
                f"dimension #{seen[key] + 1}. Keys must be unique."
            )
        seen[key] = index

        dimensions.append(
            Dimension(
                key=key,
                label=_require_text(entry, "label", where),
                prompt_description=_require_text(entry, "prompt_description", where),
            )
        )

    return dimensions


DIMENSIONS: list[Dimension] = load_dimensions()

DIMENSION_KEYS: list[str] = [d.key for d in DIMENSIONS]
