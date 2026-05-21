"""Path-leak detection regex tests for Stage 9 reproducibility verification."""

from __future__ import annotations

import pytest

from src.analysis.reproducibility_verification import ABSOLUTE_PATH_PATTERN


CASES_THAT_MUST_MATCH = (
    "[PRIVATE_PATH_REDACTED]",
    "[PRIVATE_PATH_REDACTED]",
    "/mnt/c[PRIVATE_PATH_REDACTED]",
    "/mnt/data/share/file.csv",
    r"[PRIVATE_PATH_REDACTED]",
    "[PRIVATE_PATH_REDACTED]",
    "tracebacks: file \"[PRIVATE_PATH_REDACTED]\"",
)

CASES_THAT_MUST_NOT_MATCH = (
    "data/processed/sample_index.csv",
    "outputs/analysis/stage9_audit.json",
    "src/training/load_final_model.py",
    "configs/modeling_global.yaml",
    "./tests/test_path_leak_detection.py",
    "../docs/modeling/validated_modeling_run_summary.md",
    "/var/log/system.log",
    "/etc/passwd",
    "no_paths_here_only_text",
)


@pytest.mark.parametrize("text", CASES_THAT_MUST_MATCH)
def test_pattern_matches_absolute_paths(text: str) -> None:
    """The regex catches the four absolute-path prefix families."""
    assert ABSOLUTE_PATH_PATTERN.search(text) is not None, (
        f"Expected pattern to match {text!r}, but it did not."
    )


@pytest.mark.parametrize("text", CASES_THAT_MUST_NOT_MATCH)
def test_pattern_skips_project_relative_paths(text: str) -> None:
    """Project-root-relative paths and other safe paths are not matched."""
    assert ABSOLUTE_PATH_PATTERN.search(text) is None, (
        f"Expected pattern NOT to match {text!r}, but it did."
    )


def test_pattern_extracts_prefix_only() -> None:
    """The matched prefix is the leak signature (e.g., `/home/<user>/`), not the full path."""
    match = ABSOLUTE_PATH_PATTERN.search("ls [PRIVATE_PATH_REDACTED]")
    assert match is not None
    assert match.group(1) == "[PRIVATE_PATH_REDACTED]"


def test_pattern_catches_multiple_leaks_in_one_line() -> None:
    """The regex returns one match per leak instance via findall."""
    line = "Linked [PRIVATE_PATH_REDACTED] and [PRIVATE_PATH_REDACTED] and [PRIVATE_PATH_REDACTED]"
    hits = ABSOLUTE_PATH_PATTERN.findall(line)
    assert len(hits) == 3
    assert hits[0].startswith("/home/")
    assert hits[1].startswith("/Users/")
    assert hits[2].startswith("C:\\Users\\")
