from __future__ import annotations

from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_SOURCE_FILES = [
    PROJECT_ROOT / "src/analysis/tumor_marker_model.py",
    PROJECT_ROOT / "src/analysis/vocs_marker_fusion.py",
    PROJECT_ROOT / "src/analysis/stage7_part_b_common.py",
]
OPTIONAL_SOURCE_FILES = [
    PROJECT_ROOT / "src/analysis/stage7_part_c_common.py",
]
FORBIDDEN_SUBSTRINGS = [
    "iterativeimputer",
    "knnimputer",
    "simpleimputer",
    "from sklearn.impute",
    "import sklearn.impute",
    "miceforest",
    "fancyimpute",
]
FORBIDDEN_FILLNA_PATTERN = re.compile(
    r"\.fillna\(\s*(?:0(?:\.0+)?|['\"](?:mean|median|ffill|bfill)['\"]|mean|median|ffill|bfill)",
    re.IGNORECASE,
)


def _source_paths_to_scan() -> list[Path]:
    """Return the tumor-marker-relevant source files, failing if required ones are absent."""
    missing_files = [path for path in REQUIRED_SOURCE_FILES if not path.exists()]
    assert not missing_files, f"Missing tumor-marker source files: {[str(path.relative_to(PROJECT_ROOT)) for path in missing_files]}"
    return REQUIRED_SOURCE_FILES + [path for path in OPTIONAL_SOURCE_FILES if path.exists()]


def test_no_imputation_library_or_fillna_pattern_exists_in_tumor_marker_sources() -> None:
    """Tumor-marker analysis must remain complete-case only with no imputation code path."""
    for path in _source_paths_to_scan():
        source_text = path.read_text(encoding="utf-8")
        lowered = source_text.lower()
        for forbidden in FORBIDDEN_SUBSTRINGS:
            assert forbidden not in lowered, f"Forbidden imputation pattern {forbidden!r} found in {path.relative_to(PROJECT_ROOT)}"
        assert FORBIDDEN_FILLNA_PATTERN.search(source_text) is None, (
            f"Forbidden fillna-style imputation pattern found in {path.relative_to(PROJECT_ROOT)}"
        )


def test_fit_and_predict_steps_assert_no_nan_matrix_in_tumor_marker_sources() -> None:
    """Any tumor-marker fit or predict_proba use should be guarded by a no-NaN matrix assertion helper."""
    for path in _source_paths_to_scan():
        source_text = path.read_text(encoding="utf-8")
        if ".fit(" in source_text or ".predict_proba(" in source_text:
            assert "_assert_no_nan_matrix(" in source_text, (
                f"Expected _assert_no_nan_matrix guard in {path.relative_to(PROJECT_ROOT)}"
            )
