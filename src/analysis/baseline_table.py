from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.data_split import (
    DISEASE_CATEGORY_LABELS,
    _compute_binary_smd,
    _compute_continuous_smd,
    _encode_hp_infection,
    _format_path,
    _load_yaml_config,
    _prepare_cohort_dataframe,
    _resolve_path,
    _write_text,
)


LOGGER = logging.getLogger(__name__)


def _load_split_indices(path: Path, key: str) -> np.ndarray:
    """Load one saved split-index array from a compressed npz file."""
    try:
        with np.load(path, allow_pickle=False) as data:
            return data[key].astype(int)
    except OSError as exc:
        LOGGER.error("Failed to read split indices from %s.", path)
        raise OSError(f"Failed to read split indices from {path}.") from exc


def _format_continuous_summary(values: pd.Series) -> str:
    """Format a continuous variable as mean ± SD."""
    return f"{values.mean():.2f} ± {values.std(ddof=1):.2f}"


def _format_binary_summary(values: pd.Series) -> str:
    """Format a binary indicator as count and percentage for value 1."""
    count = int(values.astype(int).sum())
    percentage = 100.0 * count / len(values)
    return f"{count}/{len(values)} ({percentage:.1f}%)"


def _build_table_rows(train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> list[dict[str, str]]:
    """Build Table 1 rows for the prespecified baseline variables."""
    train_hp = _encode_hp_infection(train_frame["hp_infection"])
    test_hp = _encode_hp_infection(test_frame["hp_infection"])
    rows: list[dict[str, str]] = [
        {
            "variable": "age",
            "train_summary": _format_continuous_summary(train_frame["age"]),
            "test_summary": _format_continuous_summary(test_frame["age"]),
            "smd": f"{_compute_continuous_smd(train_frame['age'], test_frame['age']):.6f}",
        },
        {
            "variable": "sex",
            "train_summary": _format_binary_summary(train_frame["sex"]),
            "test_summary": _format_binary_summary(test_frame["sex"]),
            "smd": f"{_compute_binary_smd(train_frame['sex'], test_frame['sex']):.6f}",
        },
        {
            "variable": "BMI",
            "train_summary": _format_continuous_summary(train_frame["bmi"]),
            "test_summary": _format_continuous_summary(test_frame["bmi"]),
            "smd": f"{_compute_continuous_smd(train_frame['bmi'], test_frame['bmi']):.6f}",
        },
        {
            "variable": "smoking",
            "train_summary": _format_binary_summary(train_frame["smoking"]),
            "test_summary": _format_binary_summary(test_frame["smoking"]),
            "smd": f"{_compute_binary_smd(train_frame['smoking'], test_frame['smoking']):.6f}",
        },
        {
            "variable": "drinking",
            "train_summary": _format_binary_summary(train_frame["drinking"]),
            "test_summary": _format_binary_summary(test_frame["drinking"]),
            "smd": f"{_compute_binary_smd(train_frame['drinking'], test_frame['drinking']):.6f}",
        },
        {
            "variable": "first_degree_family_history_of_GC",
            "train_summary": _format_binary_summary(train_frame["family_history"]),
            "test_summary": _format_binary_summary(test_frame["family_history"]),
            "smd": f"{_compute_binary_smd(train_frame['family_history'], test_frame['family_history']):.6f}",
        },
        {
            "variable": "Hp_infection=1",
            "train_summary": _format_binary_summary(train_hp == "1"),
            "test_summary": _format_binary_summary(test_hp == "1"),
            "smd": f"{_compute_binary_smd(train_hp == '1', test_hp == '1'):.6f}",
        },
        {
            "variable": "Hp_infection=missing",
            "train_summary": _format_binary_summary(train_hp == "missing"),
            "test_summary": _format_binary_summary(test_hp == "missing"),
            "smd": f"{_compute_binary_smd(train_hp == 'missing', test_hp == 'missing'):.6f}",
        },
        {
            "variable": "collection_location",
            "train_summary": _format_binary_summary(train_frame["collection_location"]),
            "test_summary": _format_binary_summary(test_frame["collection_location"]),
            "smd": f"{_compute_binary_smd(train_frame['collection_location'], test_frame['collection_location']):.6f}",
        },
        {
            "variable": "label",
            "train_summary": _format_binary_summary(train_frame["label"]),
            "test_summary": _format_binary_summary(test_frame["label"]),
            "smd": f"{_compute_binary_smd(train_frame['label'], test_frame['label']):.6f}",
        },
    ]

    for level, label in DISEASE_CATEGORY_LABELS.items():
        train_indicator = train_frame["disease_category"] == level
        test_indicator = test_frame["disease_category"] == level
        rows.append(
            {
                "variable": f"disease_category={level} ({label})",
                "train_summary": _format_binary_summary(train_indicator),
                "test_summary": _format_binary_summary(test_indicator),
                "smd": f"{_compute_binary_smd(train_indicator, test_indicator):.6f}",
            }
        )
    return rows


def _table_rows_to_markdown(table_rows: list[dict[str, str]]) -> str:
    """Render Table 1 rows as markdown."""
    header = ["variable", "train_summary", "test_summary", "smd"]
    divider = ["---"] * len(header)
    body = [
        "| " + " | ".join([row["variable"], row["train_summary"], row["test_summary"], row["smd"]]) + " |"
        for row in table_rows
    ]
    return "\n".join(
        [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(divider) + " |",
            *body,
        ]
    )


def generate_table_one(config_path: Path) -> None:
    """Generate the train-vs-test Table 1 markdown and CSV outputs.

    Args:
        config_path: Path to `configs/modeling_global.yaml`.

    Raises:
        OSError: If any required input or output file cannot be accessed.
        ValueError: If the config schema is invalid.
    """
    config = _load_yaml_config(config_path=config_path)
    cohort = _prepare_cohort_dataframe(config=config)
    splits_dir = _resolve_path(config["paths"]["splits_dir"])
    train_idx = _load_split_indices(path=splits_dir / "train_idx.npz", key="train_idx")
    test_idx = _load_split_indices(path=splits_dir / "test_idx.npz", key="test_idx")

    train_frame = cohort.iloc[train_idx].reset_index(drop=True)
    test_frame = cohort.iloc[test_idx].reset_index(drop=True)
    table_rows = _build_table_rows(train_frame=train_frame, test_frame=test_frame)
    table_frame = pd.DataFrame(table_rows)

    outputs_analysis_dir = _resolve_path(config["paths"]["outputs_analysis_dir"])
    csv_path = outputs_analysis_dir / "table_one_train_test.csv"
    markdown_path = outputs_analysis_dir / "table_one_train_test.md"

    outputs_analysis_dir.mkdir(parents=True, exist_ok=True)
    try:
        table_frame.to_csv(csv_path, index=False)
    except OSError as exc:
        LOGGER.error("Failed to write Table 1 CSV to %s.", csv_path)
        raise OSError(f"Failed to write Table 1 CSV to {csv_path}.") from exc

    markdown = (
        "# Table 1 — Train vs Test Baseline Characteristics\n\n"
        f"- Source splits directory: `{_format_path(splits_dir)}`\n"
        f"- CSV companion: `{_format_path(csv_path)}`\n\n"
        f"{_table_rows_to_markdown(table_rows=table_rows)}\n"
    )
    _write_text(path=markdown_path, content=markdown)
    LOGGER.info(
        "Wrote Table 1 outputs to %s and %s.",
        _format_path(markdown_path),
        _format_path(csv_path),
    )
