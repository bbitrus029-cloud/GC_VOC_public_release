from __future__ import annotations

from datetime import datetime, timezone
from itertools import combinations
import logging
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu

from src.analysis.data_split import (
    DISEASE_CATEGORY_LABELS,
    METADATA_COLUMN_MAP,
    _compute_binary_smd,
    _compute_continuous_smd,
    _encode_hp_infection,
    _format_path,
    _load_yaml_config,
    _prepare_cohort_dataframe,
    _resolve_path,
    _write_text,
)
from src.config import METADATA_PATH, PROJECT_ROOT


LOGGER = logging.getLogger(__name__)

CEA_UPPER_NORMAL: float = 5.0
CA199_UPPER_NORMAL: float = 37.0
CA724_UPPER_NORMAL: float = 10.0

TUMOR_MARKER_COLUMNS: dict[str, str] = {
    "CEA(0-5μg/L)": "cea",
    "CA199(0.1-37μ/ml)": "ca199",
    "CA724(0.1-10μ/ml)": "ca724",
}


def _parse_truncated_numeric(value: object, *, decimal_comma: bool = False) -> float | None:
    """Parse a possibly censored tumor-marker value into a float or None."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    stripped = str(value).strip()
    if stripped in {"", "-", "/", "?"}:
        return None
    if stripped.startswith(">") or stripped.startswith("<"):
        stripped = stripped[1:].strip()
    if decimal_comma:
        stripped = stripped.replace(",", ".")

    try:
        return float(stripped)
    except ValueError:
        LOGGER.warning("Unparseable tumor-marker value converted to missing.")
        return None


def _load_tumor_markers(metadata_path: Path) -> pd.DataFrame:
    """Load and parse tumor-marker columns from metadata.csv."""
    usecols = [METADATA_COLUMN_MAP["sample_id"], *TUMOR_MARKER_COLUMNS.keys()]
    try:
        tumor_markers = pd.read_csv(metadata_path, usecols=usecols, encoding="utf-8-sig")
    except OSError as exc:
        LOGGER.error("Failed to read tumor-marker metadata at %s.", metadata_path)
        raise OSError(f"Failed to read tumor-marker metadata at {metadata_path}.") from exc

    parsed = pd.DataFrame({"sample_id": tumor_markers["sample_id"]})
    LOGGER.info("Loaded tumor-marker metadata for %s samples.", len(tumor_markers))

    for raw_column, alias in TUMOR_MARKER_COLUMNS.items():
        right_censored = 0
        left_censored = 0
        unparseable = 0
        values: list[float | None] = []

        for raw_value in tumor_markers[raw_column]:
            if raw_value is None or pd.isna(raw_value):
                values.append(None)
                continue

            stripped = str(raw_value).strip()
            if stripped in {"", "-", "/", "?"}:
                values.append(None)
                continue
            if stripped.startswith(">"):
                right_censored += 1
            elif stripped.startswith("<"):
                left_censored += 1

            parsed_value = _parse_truncated_numeric(raw_value, decimal_comma=alias == "ca199")
            if parsed_value is None and stripped not in {"", "-", "/", "?"} and not stripped.startswith((">", "<")):
                unparseable += 1
            values.append(parsed_value)

        parsed_series = pd.Series(values, dtype=float)
        parsed[alias] = parsed_series

        non_missing = parsed_series.dropna()
        if non_missing.empty:
            median_text = "NA"
            q1_text = "NA"
            q3_text = "NA"
        else:
            median_text = f"{non_missing.median():.1f}"
            q1_text = f"{non_missing.quantile(0.25):.1f}"
            q3_text = f"{non_missing.quantile(0.75):.1f}"

        LOGGER.info(
            (
                "Tumor marker %s: right-censored handled=%s, left-censored handled=%s, "
                "missing=%s, unparseable_to_missing=%s, median=%s, q1=%s, q3=%s"
            ),
            alias,
            right_censored,
            left_censored,
            int(parsed_series.isna().sum()),
            unparseable,
            median_text,
            q1_text,
            q3_text,
        )

    return parsed


def _format_median_iqr(values: pd.Series, decimals: int = 2) -> str:
    """Format a continuous summary as median [Q1–Q3]."""
    clean_values = values.dropna().astype(float)
    if clean_values.empty:
        return "—"

    median = clean_values.median()
    q1 = clean_values.quantile(0.25)
    q3 = clean_values.quantile(0.75)
    return f"{median:.{decimals}f} [{q1:.{decimals}f}–{q3:.{decimals}f}]"


def _format_count_percent(values: pd.Series, target_value: object) -> str:
    """Format a categorical summary as n/N (p.p%)."""
    explicit_missing = values.eq("missing")
    if target_value == "missing":
        numerator = int(values.isna().sum() + explicit_missing.sum())
        denominator = int(len(values))
    else:
        clean_values = values.loc[~values.isna() & ~explicit_missing]
        numerator = int((clean_values == target_value).sum())
        denominator = int(len(clean_values))

    if denominator == 0:
        return f"{numerator}/{denominator} (NA%)"

    percentage = 100.0 * numerator / denominator
    return f"{numerator}/{denominator} ({percentage:.1f}%)"


def _omnibus_p_continuous(groups: list[np.ndarray], variable_name: str = "unknown") -> float:
    """Dispatch the continuous omnibus test based on the number of groups."""
    clean_groups = [np.asarray(group, dtype=float)[~np.isnan(np.asarray(group, dtype=float))] for group in groups]
    if any(len(group) < 2 for group in clean_groups):
        LOGGER.warning(
            "Continuous variable %s has fewer than 2 non-missing values in at least one group; returning NaN p value.",
            variable_name,
        )
        return float("nan")

    if len(clean_groups) == 2:
        return float(mannwhitneyu(clean_groups[0], clean_groups[1], alternative="two-sided").pvalue)
    return float(kruskal(*clean_groups).pvalue)


def _omnibus_p_categorical(contingency_table: np.ndarray) -> tuple[float, int]:
    """Compute chi-square p value and the number of expected cells below five."""
    if np.any(contingency_table.sum(axis=0) == 0) or np.any(contingency_table.sum(axis=1) == 0):
        return float("nan"), -1

    _, p_value, _, expected = chi2_contingency(contingency_table)
    return float(p_value), int((expected < 5).sum())


def _compute_max_pairwise_continuous_smd(group_values: dict[str, pd.Series]) -> tuple[float, tuple[str, str]]:
    """Return the maximum pairwise absolute SMD across named continuous groups."""
    best_smd = float("nan")
    best_pair = ("", "")

    for left_name, right_name in combinations(group_values, 2):
        left_values = group_values[left_name].dropna().astype(float)
        right_values = group_values[right_name].dropna().astype(float)
        if len(left_values) < 2 or len(right_values) < 2:
            continue

        smd_value = _compute_continuous_smd(left_values, right_values)
        if np.isnan(best_smd) or smd_value > best_smd:
            best_smd = smd_value
            best_pair = (left_name, right_name)

    return best_smd, best_pair


def _compute_max_one_vs_rest_binary_smd(group_values: dict[str, pd.Series]) -> tuple[float, str]:
    """Return the maximum one-vs-rest absolute SMD across named binary groups."""
    best_smd = float("nan")
    best_group = ""

    for group_name, focal_values in group_values.items():
        rest_series = [values for other_name, values in group_values.items() if other_name != group_name]
        rest_values = pd.concat(rest_series, ignore_index=True) if rest_series else pd.Series(dtype=float)

        focal_clean = focal_values.dropna()
        rest_clean = rest_values.dropna()
        if focal_clean.empty or rest_clean.empty:
            continue

        smd_value = _compute_binary_smd(focal_clean.astype(int), rest_clean.astype(int))
        if np.isnan(best_smd) or smd_value > best_smd:
            best_smd = smd_value
            best_group = group_name

    return best_smd, best_group


def _format_p_value(p_value: float) -> str:
    """Format a p value for table output."""
    return "NA" if np.isnan(p_value) else f"{p_value:.4f}"


def _format_smd_value(smd_value: float) -> str:
    """Format an SMD value for table output."""
    return "NA" if np.isnan(smd_value) else f"{smd_value:.6f}"


def _make_elevated_indicator(values: pd.Series, upper_normal: float) -> pd.Series:
    """Encode an elevated-vs-normal indicator while preserving missing values."""
    indicator = pd.Series(np.nan, index=values.index, dtype=float)
    non_missing = values.notna()
    indicator.loc[non_missing] = (values.loc[non_missing] > upper_normal).astype(int)
    return indicator


def _make_binary_indicator(values: pd.Series, target_value: object) -> pd.Series:
    """Encode a binary row indicator while preserving missing denominators."""
    if target_value == "missing":
        return (values.isna() | values.eq("missing")).astype(int)

    indicator = pd.Series(np.nan, index=values.index, dtype=float)
    valid = ~values.isna() & ~values.eq("missing")
    indicator.loc[valid] = (values.loc[valid] == target_value).astype(int)
    return indicator


def _build_binary_contingency_table(group_values: dict[str, pd.Series]) -> np.ndarray:
    """Build a groups-by-two contingency table from 0/1-valued series."""
    rows: list[list[int]] = []
    for values in group_values.values():
        clean_values = values.dropna()
        if clean_values.empty:
            rows.append([0, 0])
            continue
        target_count = int((clean_values.astype(int) == 1).sum())
        other_count = int(len(clean_values) - target_count)
        rows.append([target_count, other_count])
    return np.asarray(rows, dtype=int)


def _build_variable_rows(
    cohort: pd.DataFrame,
    group_labels: dict[str, np.ndarray],
    comparison_mode: str,
) -> list[dict[str, str]]:
    """Build the cohort-description rows in the required paper order."""
    age_cutpoint = 60
    bmi_cutpoint = 24
    hp_encoded = _encode_hp_infection(cohort["hp_infection"])

    age_ge_60 = (cohort["age"] >= age_cutpoint).astype(int)
    bmi_ge_24 = (cohort["bmi"] >= bmi_cutpoint).astype(int)
    cea_elevated = _make_elevated_indicator(cohort["cea"], CEA_UPPER_NORMAL)
    ca199_elevated = _make_elevated_indicator(cohort["ca199"], CA199_UPPER_NORMAL)
    ca724_elevated = _make_elevated_indicator(cohort["ca724"], CA724_UPPER_NORMAL)

    row_specs: list[dict[str, object]] = [
        {"section": "Demographics"},
        {"label": "Age, years (median [IQR])", "kind": "continuous", "series": cohort["age"], "decimals": 2},
        {"label": "Age ≥60, n (%)", "kind": "categorical", "series": age_ge_60, "target_value": 1},
        {"label": "Sex, female n (%)", "kind": "categorical", "series": cohort["sex"], "target_value": 1},
        {"label": "BMI, kg/m² (median [IQR])", "kind": "continuous", "series": cohort["bmi"], "decimals": 2},
        {"label": "BMI ≥24, n (%)", "kind": "categorical", "series": bmi_ge_24, "target_value": 1},
        {"section": "Lifestyle and family history"},
        {"label": "Current/ever smoker, n (%)", "kind": "categorical", "series": cohort["smoking"], "target_value": 1},
        {"label": "Current/ever drinker, n (%)", "kind": "categorical", "series": cohort["drinking"], "target_value": 1},
        {
            "label": "First-degree family history of GC, n (%)",
            "kind": "categorical",
            "series": cohort["family_history"],
            "target_value": 1,
        },
        {"section": "Helicobacter pylori"},
        {"label": "Hp_infection positive, n (%)", "kind": "categorical", "series": hp_encoded, "target_value": "1"},
        {"label": "Hp_infection missing, n (%)", "kind": "categorical", "series": hp_encoded, "target_value": "missing"},
        {"section": "Clinical setting"},
        {"label": "Inpatient, n (%)", "kind": "categorical", "series": cohort["collection_location"], "target_value": 1},
        {"section": "Serum tumor markers"},
        {"label": "CEA, μg/L (median [IQR])", "kind": "continuous", "series": cohort["cea"], "decimals": 1},
        {"label": "CEA missing, n (%)", "kind": "categorical", "series": cohort["cea"], "target_value": "missing"},
        {"label": "CEA elevated (>5 μg/L), n (%)", "kind": "categorical", "series": cea_elevated, "target_value": 1},
        {"label": "CA19-9, U/mL (median [IQR])", "kind": "continuous", "series": cohort["ca199"], "decimals": 1},
        {"label": "CA19-9 missing, n (%)", "kind": "categorical", "series": cohort["ca199"], "target_value": "missing"},
        {"label": "CA19-9 elevated (>37 U/mL), n (%)", "kind": "categorical", "series": ca199_elevated, "target_value": 1},
        {"label": "CA72-4, U/mL (median [IQR])", "kind": "continuous", "series": cohort["ca724"], "decimals": 1},
        {"label": "CA72-4 missing, n (%)", "kind": "categorical", "series": cohort["ca724"], "target_value": "missing"},
        {"label": "CA72-4 elevated (>10 U/mL), n (%)", "kind": "categorical", "series": ca724_elevated, "target_value": 1},
    ]

    rows: list[dict[str, str]] = []
    group_names = list(group_labels.keys())

    for spec in row_specs:
        if "section" in spec:
            row = {"variable": f"**{spec['section']}**", "total": "", "p_value": "", "smd": ""}
            row.update({group_name: "" for group_name in group_names})
            rows.append(row)
            continue

        label = str(spec["label"])
        kind = str(spec["kind"])
        values = pd.Series(spec["series"])
        row = {"variable": label}

        if kind == "continuous":
            decimals = int(spec["decimals"])
            row["total"] = _format_median_iqr(values, decimals=decimals)
            group_value_map = {group_name: values.loc[group_mask] for group_name, group_mask in group_labels.items()}
            for group_name, group_values in group_value_map.items():
                row[group_name] = _format_median_iqr(group_values, decimals=decimals)

            p_value = _omnibus_p_continuous(
                [group_values.to_numpy(dtype=float) for group_values in group_value_map.values()],
                variable_name=label,
            )
            if comparison_mode == "pos_vs_neg":
                positive_values = group_value_map[group_names[0]].dropna().astype(float)
                negative_values = group_value_map[group_names[1]].dropna().astype(float)
                smd_value = (
                    _compute_continuous_smd(positive_values, negative_values)
                    if len(positive_values) >= 2 and len(negative_values) >= 2
                    else float("nan")
                )
            else:
                smd_value, max_pair = _compute_max_pairwise_continuous_smd(group_value_map)
                if not np.isnan(smd_value):
                    LOGGER.info(
                        "Continuous Table 2 SMD for %s maximized by pair %s vs %s (|SMD|=%.6f).",
                        label,
                        max_pair[0],
                        max_pair[1],
                        smd_value,
                    )

            row["p_value"] = _format_p_value(p_value)
            row["smd"] = _format_smd_value(smd_value)
            rows.append(row)
            continue

        target_value = spec["target_value"]
        row["total"] = _format_count_percent(values, target_value=target_value)

        indicator_series = _make_binary_indicator(values, target_value=target_value)
        group_indicator_map = {group_name: indicator_series.loc[group_mask] for group_name, group_mask in group_labels.items()}

        for group_name, group_values in group_labels.items():
            row[group_name] = _format_count_percent(values.loc[group_values], target_value=target_value)

        contingency_table = _build_binary_contingency_table(group_indicator_map)
        p_value, expected_below_5 = _omnibus_p_categorical(contingency_table)
        if expected_below_5 >= 0:
            total_cells = contingency_table.size
            LOGGER.info(
                "Categorical variable %s chi-square expected cell count < 5: %s/%s cells (%.1f%%).",
                label,
                expected_below_5,
                total_cells,
                100.0 * expected_below_5 / total_cells,
            )
        else:
            LOGGER.info("Categorical variable %s has a zero row/column total; chi-square p value reported as NA.", label)

        if comparison_mode == "pos_vs_neg":
            positive_values = group_indicator_map[group_names[0]].dropna()
            negative_values = group_indicator_map[group_names[1]].dropna()
            smd_value = (
                _compute_binary_smd(positive_values.astype(int), negative_values.astype(int))
                if not positive_values.empty and not negative_values.empty
                else float("nan")
            )
        else:
            smd_value, max_group = _compute_max_one_vs_rest_binary_smd(group_indicator_map)
            if not np.isnan(smd_value):
                LOGGER.info(
                    "Categorical Table 2 SMD for %s maximized by %s vs rest (|SMD|=%.6f).",
                    label,
                    max_group,
                    smd_value,
                )

        row["p_value"] = _format_p_value(p_value)
        row["smd"] = _format_smd_value(smd_value)
        rows.append(row)

    return rows


def _get_git_short_hash() -> str:
    """Return the current git short hash for provenance output."""
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_table_csv(path: Path, table_frame: pd.DataFrame) -> None:
    """Write a cohort-description CSV with informative error handling."""
    try:
        table_frame.to_csv(path, index=False)
    except OSError as exc:
        LOGGER.error("Failed to write cohort-description CSV to %s.", path)
        raise OSError(f"Failed to write cohort-description CSV to {path}.") from exc


def _render_markdown_table(table_frame: pd.DataFrame) -> str:
    """Render the dataframe rows into a markdown table."""
    header = [str(column) for column in table_frame.columns]
    divider = ["---"] * len(header)
    body_lines = [
        "| " + " | ".join(str(row[column]) for column in table_frame.columns) + " |"
        for _, row in table_frame.fillna("").iterrows()
    ]
    return "\n".join(
        [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(divider) + " |",
            *body_lines,
        ]
    )


def _build_markdown_document(
    title: str,
    source_line: str,
    continuous_note: str,
    categorical_note: str,
    extra_notes: list[str],
    table_frame: pd.DataFrame,
) -> str:
    """Build the markdown document for a cohort-description table."""
    timestamp_utc = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    git_short_hash = _get_git_short_hash()
    bullet_lines = [
        f"- Source data: {source_line}",
        f"- UTC timestamp: `{timestamp_utc}`",
        f"- Git short hash: `{git_short_hash}`",
        f"- Continuous variables: {continuous_note}",
        f"- Categorical variables: {categorical_note}",
        *[f"- {note}" for note in extra_notes],
    ]
    footnotes = [
        "- p values are descriptive; no multiple-comparison adjustment was applied. Statistical significance was not interpreted from these tests; primary hypothesis testing uses prespecified analyses described in `docs/protocol/modeling_plan.md`.",
        "- For variables with NaN values (Hp_infection, tumor markers), denominators of n (%) cells exclude NaN unless the row explicitly counts missingness.",
        "- Chi-square Cochran's-rule transparency: see logs for any variable where >20% of contingency-table cells had expected frequency <5; chi-square approximations remain adequate when fewer than 20% of cells have expected count <5 and no expected count is <1.",
        "- |SMD| computed using Cohen's pooled-SD convention as in Phase 2 Table 1a (`src/analysis/data_split.py::_compute_continuous_smd`).",
    ]
    return (
        f"# {title}\n\n"
        + "\n".join(bullet_lines)
        + "\n\n"
        + _render_markdown_table(table_frame)
        + "\n\n## Footnotes\n\n"
        + "\n".join(footnotes)
        + "\n"
    )


def _build_output_frame(rows: list[dict[str, str]], column_order: list[str]) -> pd.DataFrame:
    """Convert row dictionaries into a dataframe with a fixed column order."""
    table_frame = pd.DataFrame(rows)
    return table_frame.reindex(columns=column_order).fillna("")


def _rename_group_columns(rows: list[dict[str, str]], display_names: dict[str, str]) -> list[dict[str, str]]:
    """Rename internal group keys to display labels with counts."""
    renamed_rows: list[dict[str, str]] = []
    for row in rows:
        renamed_row = {"variable": row["variable"], "total": row["total"]}
        for original_name, display_name in display_names.items():
            renamed_row[display_name] = row.get(original_name, "")
        renamed_row["p_value"] = row["p_value"]
        renamed_row["smd"] = row["smd"]
        renamed_rows.append(renamed_row)
    return renamed_rows


def generate_pos_vs_neg_table(config_path: Path) -> None:
    """Generate Table 1: Positive vs Negative baseline characteristics.

    Args:
        config_path: Path to `configs/modeling_global.yaml`.
    """
    config = _load_yaml_config(config_path=config_path)
    cohort = _prepare_cohort_dataframe(config=config)
    tumor_markers = _load_tumor_markers(METADATA_PATH)
    cohort = cohort.merge(tumor_markers, on="sample_id", how="left", validate="one_to_one")

    group_labels = {
        "Positive": cohort["label"].to_numpy(dtype=int) == 1,
        "Negative": cohort["label"].to_numpy(dtype=int) == 0,
    }
    display_names = {
        "Positive": f"Positive (n={int(group_labels['Positive'].sum())})",
        "Negative": f"Negative (n={int(group_labels['Negative'].sum())})",
    }
    rows = _build_variable_rows(cohort=cohort, group_labels=group_labels, comparison_mode="pos_vs_neg")
    renamed_rows = _rename_group_columns(rows, display_names=display_names)
    column_order = ["variable", "total", *display_names.values(), "p_value", "smd"]
    table_frame = _build_output_frame(rows=renamed_rows, column_order=column_order)
    table_frame = table_frame.rename(columns={"total": f"Total (N={len(cohort)})", "p_value": "p value", "smd": "|SMD|"})

    outputs_analysis_dir = _resolve_path(config["paths"]["outputs_analysis_dir"])
    csv_path = outputs_analysis_dir / "table_one_pos_vs_neg.csv"
    markdown_path = outputs_analysis_dir / "table_one_pos_vs_neg.md"
    outputs_analysis_dir.mkdir(parents=True, exist_ok=True)

    _write_table_csv(csv_path, table_frame)
    markdown = _build_markdown_document(
        title="Table 1 — Baseline Characteristics by Composite Endpoint (Full Cohort, N=972)",
        source_line="`data/processed/sample_index.csv` + `data/raw/clinical_data/metadata.csv`",
        continuous_note="median [Q1–Q3]; comparisons by Mann-Whitney U test",
        categorical_note="n (%); comparisons by chi-square test of independence (Cochran's rule note in footer)",
        extra_notes=["Composite endpoint: Positive = HGIN + EGC + AGC; Negative = HC + BGL + LGIN"],
        table_frame=table_frame,
    )
    _write_text(markdown_path, markdown)
    LOGGER.info(
        "Wrote Table 1 cohort-description outputs to %s and %s (rows=%s).",
        _format_path(markdown_path),
        _format_path(csv_path),
        len(rows),
    )


def generate_six_categories_table(config_path: Path) -> None:
    """Generate Table 2: Six disease-category baseline characteristics.

    Args:
        config_path: Path to `configs/modeling_global.yaml`.
    """
    config = _load_yaml_config(config_path=config_path)
    cohort = _prepare_cohort_dataframe(config=config)
    tumor_markers = _load_tumor_markers(METADATA_PATH)
    cohort = cohort.merge(tumor_markers, on="sample_id", how="left", validate="one_to_one")

    group_labels = {
        label: cohort["disease_category"].to_numpy(dtype=int) == code
        for code, label in DISEASE_CATEGORY_LABELS.items()
    }
    display_names = {
        label: f"{label} (n={int(mask.sum())})"
        for label, mask in group_labels.items()
    }
    rows = _build_variable_rows(cohort=cohort, group_labels=group_labels, comparison_mode="six_categories")
    renamed_rows = _rename_group_columns(rows, display_names=display_names)
    column_order = ["variable", "total", *display_names.values(), "p_value", "smd"]
    table_frame = _build_output_frame(rows=renamed_rows, column_order=column_order)
    table_frame = table_frame.rename(columns={"total": f"Total (N={len(cohort)})", "p_value": "p value", "smd": "|SMD|"})

    outputs_analysis_dir = _resolve_path(config["paths"]["outputs_analysis_dir"])
    csv_path = outputs_analysis_dir / "table_two_six_categories.csv"
    markdown_path = outputs_analysis_dir / "table_two_six_categories.md"
    outputs_analysis_dir.mkdir(parents=True, exist_ok=True)

    _write_table_csv(csv_path, table_frame)
    markdown = _build_markdown_document(
        title="Table 2 — Baseline Characteristics by Disease Category (Full Cohort, N=972)",
        source_line="`data/processed/sample_index.csv` + `data/raw/clinical_data/metadata.csv`",
        continuous_note="median [Q1–Q3]; comparisons by Kruskal-Wallis test",
        categorical_note="n (%); comparisons by chi-square test of independence (Cochran's rule note in footer)",
        extra_notes=[],
        table_frame=table_frame,
    )
    _write_text(markdown_path, markdown)
    LOGGER.info(
        "Wrote Table 2 cohort-description outputs to %s and %s (rows=%s).",
        _format_path(markdown_path),
        _format_path(csv_path),
        len(rows),
    )
