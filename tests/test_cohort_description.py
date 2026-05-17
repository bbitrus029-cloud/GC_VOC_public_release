from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.analysis.data_split import _compute_continuous_smd
from src.analysis.cohort_description import (
    _compute_max_pairwise_continuous_smd,
    _format_count_percent,
    _format_median_iqr,
    _omnibus_p_categorical,
    _omnibus_p_continuous,
    _parse_truncated_numeric,
    generate_pos_vs_neg_table,
)
from src.config import METADATA_PATH, PROJECT_ROOT, PROCESSED_DATA_ROOT


REAL_DATA_AVAILABLE = METADATA_PATH.exists() and (PROCESSED_DATA_ROOT / "sample_index.csv").exists()
CONFIG_PATH = PROJECT_ROOT / "configs" / "modeling_global.yaml"
TABLE_ONE_MARKDOWN_PATH = PROJECT_ROOT / "outputs" / "analysis" / "table_one_pos_vs_neg.md"
TABLE_ONE_CSV_PATH = PROJECT_ROOT / "outputs" / "analysis" / "table_one_pos_vs_neg.csv"


def test_parse_truncated_numeric_handles_right_censored() -> None:
    """Right-censored tumor-marker values should map to the censoring bound."""
    assert _parse_truncated_numeric(">1000") == 1000.0


def test_parse_truncated_numeric_handles_left_censored() -> None:
    """Left-censored tumor-marker values should map to the censoring bound."""
    assert _parse_truncated_numeric("<0.1") == 0.1


def test_parse_truncated_numeric_handles_missing_sentinels() -> None:
    """Known missing sentinels should map to None."""
    for value in [np.nan, None, "", "-", "/", "?"]:
        assert _parse_truncated_numeric(value) is None


def test_parse_truncated_numeric_handles_numeric_string() -> None:
    """Plain numeric inputs should parse as floats."""
    assert _parse_truncated_numeric("42.5") == 42.5
    assert _parse_truncated_numeric(42.5) == 42.5


def test_parse_truncated_numeric_unparseable_returns_none(caplog: pytest.LogCaptureFixture) -> None:
    """Unparseable raw strings should log a warning and become missing."""
    with caplog.at_level("WARNING"):
        parsed = _parse_truncated_numeric("abc")

    assert parsed is None
    assert any("abc" in record.message for record in caplog.records)


def test_format_median_iqr_basic() -> None:
    """Median and IQR formatting should use two decimals and an en dash."""
    formatted = _format_median_iqr(pd.Series([1, 2, 3, 4, 5], dtype=float))
    assert "3.00" in formatted
    assert "–" in formatted


def test_format_median_iqr_all_nan_returns_dash() -> None:
    """All-NaN continuous summaries should render as an em dash."""
    formatted = _format_median_iqr(pd.Series([np.nan, np.nan], dtype=float))
    assert formatted == "—"


def test_format_count_percent_excludes_nan_by_default() -> None:
    """NaN values should not contribute to the default denominator."""
    formatted = _format_count_percent(pd.Series([1, 0, 1, np.nan]), target_value=1)
    assert "2/3" in formatted


def test_omnibus_p_continuous_dispatches_correctly() -> None:
    """Two groups use Mann-Whitney; three or more use Kruskal-Wallis."""
    with (
        patch("src.analysis.cohort_description.mannwhitneyu") as mannwhitneyu_mock,
        patch("src.analysis.cohort_description.kruskal") as kruskal_mock,
    ):
        mannwhitneyu_mock.return_value = Mock(pvalue=0.1234)
        kruskal_mock.return_value = Mock(pvalue=0.5678)

        assert _omnibus_p_continuous([np.array([1.0, 2.0]), np.array([3.0, 4.0])]) == pytest.approx(0.1234)
        mannwhitneyu_mock.assert_called_once()
        kruskal_mock.assert_not_called()

    with (
        patch("src.analysis.cohort_description.mannwhitneyu") as mannwhitneyu_mock,
        patch("src.analysis.cohort_description.kruskal") as kruskal_mock,
    ):
        mannwhitneyu_mock.return_value = Mock(pvalue=0.1234)
        kruskal_mock.return_value = Mock(pvalue=0.5678)

        assert _omnibus_p_continuous(
            [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])]
        ) == pytest.approx(0.5678)
        kruskal_mock.assert_called_once()
        mannwhitneyu_mock.assert_not_called()


def test_omnibus_p_categorical_returns_expected_below_5_count() -> None:
    """Chi-square helper should surface low expected-cell counts for transparency."""
    contingency_table = np.array(
        [
            [1, 200, 200, 200, 200, 200],
            [4, 200, 200, 200, 200, 200],
        ],
        dtype=int,
    )

    p_value, expected_below_5 = _omnibus_p_categorical(contingency_table)

    assert np.isfinite(p_value)
    assert expected_below_5 >= 1


def test_six_categories_smd_uses_max_pairwise_for_continuous() -> None:
    """Six-category continuous SMD should use the maximum pairwise absolute SMD."""
    group_values = {
        "HC": pd.Series([0.0, 0.0, 0.0, 0.0]),
        "BGL": pd.Series([0.0, 0.0, 0.0, 0.0]),
        "LGIN": pd.Series([0.0, 0.0, 0.0, 0.0]),
        "HGIN": pd.Series([0.0, 0.0, 0.0, 0.0]),
        "EGC": pd.Series([10.0, 10.0, 10.0, 10.0]),
        "AGC": pd.Series([0.0, 0.0, 0.0, 0.0]),
    }

    smd_value, max_pair = _compute_max_pairwise_continuous_smd(group_values)

    expected_smd = _compute_continuous_smd(group_values["HC"], group_values["EGC"])
    assert smd_value == pytest.approx(expected_smd)
    assert set(max_pair) == {"HC", "EGC"}


@pytest.mark.skipif(not REAL_DATA_AVAILABLE, reason="real data required; run in WSL")
def test_pos_vs_neg_row_count_and_total_consistency() -> None:
    """Real-data table generation should preserve the required row structure and cohort counts."""
    generate_pos_vs_neg_table(CONFIG_PATH)

    assert TABLE_ONE_MARKDOWN_PATH.exists()
    assert TABLE_ONE_CSV_PATH.exists()

    markdown = TABLE_ONE_MARKDOWN_PATH.read_text(encoding="utf-8")
    table_frame = pd.read_csv(TABLE_ONE_CSV_PATH)

    assert "Positive (n=413)" in markdown
    assert "Negative (n=559)" in markdown
    assert "Total (N=972)" in markdown

    section_headers = table_frame["variable"].fillna("").str.startswith("**")
    assert section_headers.sum() == 5
    assert (~section_headers).sum() == 20

    assert "**Demographics**" in markdown
    assert "**Lifestyle and family history**" in markdown
    assert "**Helicobacter pylori**" in markdown
    assert "**Clinical setting**" in markdown
    assert "**Serum tumor markers**" in markdown
