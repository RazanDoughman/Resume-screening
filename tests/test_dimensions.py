"""Tests for dimensions.py — pure config loading, no API calls."""

from pathlib import Path

import pytest

from src.dimensions import (
    DIMENSIONS,
    DIMENSIONS_PATH,
    Dimension,
    DimensionConfigError,
    load_dimensions,
)

# The four dimensions the pipeline scores today, in prompt order. Changing this
# list is a scoring change, not a refactor — the evals in tests/evals/cases.py
# assert score ranges per dimension.
EXPECTED_KEYS = [
    "skills_match",
    "experience_match",
    "role_relevance",
    "overall_fit",
]


def _write_yaml(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "dimensions.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_default_file_loads_the_four_expected_keys_in_order() -> None:
    assert [d.key for d in DIMENSIONS] == EXPECTED_KEYS


def test_default_file_is_the_one_next_to_the_package() -> None:
    assert DIMENSIONS_PATH.name == "dimensions.yaml"
    assert DIMENSIONS_PATH.exists()
    assert [d.key for d in load_dimensions(DIMENSIONS_PATH)] == EXPECTED_KEYS


def test_every_dimension_has_a_non_empty_prompt_description() -> None:
    for dimension in DIMENSIONS:
        assert dimension.prompt_description.strip(), f"{dimension.key} has no description"


def test_every_dimension_has_a_non_empty_label() -> None:
    for dimension in DIMENSIONS:
        assert dimension.label.strip(), f"{dimension.key} has no label"


def test_records_are_immutable() -> None:
    # models.py will build its schema from these at import time; a mutable
    # record would let one caller reshape the schema for everyone else.
    with pytest.raises(Exception):
        DIMENSIONS[0].key = "something_else"  # type: ignore[misc]


def test_records_carry_only_the_three_config_fields() -> None:
    dimension = DIMENSIONS[0]
    assert isinstance(dimension, Dimension)
    assert (dimension.key, dimension.label, dimension.prompt_description) == (
        "skills_match",
        "Skills match",
        DIMENSIONS[0].prompt_description,
    )


def test_empty_list_is_rejected(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, "[]\n")

    with pytest.raises(DimensionConfigError, match="no dimensions"):
        load_dimensions(path)


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    # An all-comments file parses to None, not to a list.
    path = _write_yaml(tmp_path, "# nothing here\n")

    with pytest.raises(DimensionConfigError, match="must contain a list"):
        load_dimensions(path)


def test_duplicate_keys_are_rejected(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        """
- key: skills_match
  label: Skills match
  prompt_description: first description.
- key: skills_match
  label: Skills match again
  prompt_description: second description.
""",
    )

    with pytest.raises(DimensionConfigError, match="repeats the key 'skills_match'"):
        load_dimensions(path)


def test_missing_prompt_description_is_rejected(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        """
- key: skills_match
  label: Skills match
""",
    )

    with pytest.raises(DimensionConfigError, match="missing required field"):
        load_dimensions(path)


def test_empty_prompt_description_is_rejected(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        """
- key: skills_match
  label: Skills match
  prompt_description: "   "
""",
    )

    with pytest.raises(DimensionConfigError, match="prompt_description"):
        load_dimensions(path)


def test_missing_key_is_rejected(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        """
- label: Skills match
  prompt_description: overlap between skills.
""",
    )

    with pytest.raises(DimensionConfigError, match="missing required field 'key'"):
        load_dimensions(path)


def test_empty_key_is_rejected(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        """
- key: ""
  label: Skills match
  prompt_description: overlap between skills.
""",
    )

    with pytest.raises(DimensionConfigError, match="empty or non-text 'key'"):
        load_dimensions(path)


def test_non_mapping_entry_is_rejected(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, "- skills_match\n")

    with pytest.raises(DimensionConfigError, match="must be a mapping"):
        load_dimensions(path)


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DimensionConfigError, match="No dimensions file"):
        load_dimensions(tmp_path / "does_not_exist.yaml")


def test_error_messages_name_the_offending_dimension(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        """
- key: skills_match
  label: Skills match
  prompt_description: overlap between skills.
- key: experience_match
  label: Experience match
""",
    )

    with pytest.raises(DimensionConfigError, match="dimension #2"):
        load_dimensions(path)
