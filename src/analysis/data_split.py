from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.config import METADATA_PATH, PROCESSED_DATA_ROOT, PROJECT_ROOT
from src.training.common.early_stopping import create_dl_holdout_indices, save_dl_holdout
from src.training.common.fold_generator import FoldGenerator


LOGGER = logging.getLogger(__name__)
SEED_TIE_TOLERANCE = 1e-6
EXPECTED_FINAL_COHORT_SIZE = 972
REQUIRED_SAMPLE_INDEX_COLUMNS = [
    "sample_id",
    "label",
    "disease_category",
    "fallback_used",
    "padded",
    "in_final_output",
]
METADATA_COLUMN_MAP = {
    "sample_id": "sample_id",
    "label": "metadata_label",
    "disease_category(0HC1BGL2LGIN3HGIN4EGC5AGC)": "metadata_disease_category",
    "age": "age",
    "sex(0male1female)": "sex",
    "BMI": "bmi",
    "smoking(0no1yes)": "smoking",
    "drinking(0no1yes)": "drinking",
    "first_degree_family_history_of_GC(0no1yes)": "family_history",
    "Hp_Infection(0no1yes)": "hp_infection",
    "collection_location(0outpatient1inpatient)": "collection_location",
}
DISEASE_CATEGORY_LABELS = {
    0: "HC",
    1: "BGL",
    2: "LGIN",
    3: "HGIN",
    4: "EGC",
    5: "AGC",
}


@dataclass(frozen=True)
class SeedScore:
    """Store per-seed imbalance metrics for lexicographic selection."""

    seed: int
    label_smd: float
    disease_category_smd: float
    age_smd: float
    bmi_smd: float
    sex_smd: float
    smoking_smd: float
    drinking_smd: float
    family_history_smd: float
    hp_smd: float
    fallback_used_smd: float
    padded_smd: float


def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML config file into a dictionary."""
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except OSError as exc:
        LOGGER.error("Failed to read YAML config at %s.", config_path)
        raise OSError(f"Failed to read YAML config at {config_path}.") from exc

    if not isinstance(config, dict):
        raise ValueError(f"Config at {config_path} must deserialize to a mapping.")
    return config


def _resolve_path(path_str: str) -> Path:
    """Resolve a project-root-relative path string."""
    return PROJECT_ROOT / Path(path_str)


def _format_path(path: Path) -> str:
    """Return a project-root-relative string for tracked outputs."""
    return str(path.relative_to(PROJECT_ROOT))


def _log_dataframe_summary(name: str, dataframe: pd.DataFrame) -> None:
    """Log dataframe shape, dtypes, and missing counts."""
    LOGGER.info(
        "%s summary: shape=%s, dtypes=%s, missing_counts=%s",
        name,
        dataframe.shape,
        {column: str(dtype) for column, dtype in dataframe.dtypes.items()},
        dataframe.isna().sum().to_dict(),
    )


def _log_array_summary(name: str, array: np.ndarray) -> None:
    """Log numpy-array shape and dtype."""
    LOGGER.info("%s summary: shape=%s, dtype=%s", name, array.shape, array.dtype)


def _normalize_bool_series(series: pd.Series, column_name: str) -> pd.Series:
    """Convert a bool-like series to pandas boolean dtype."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")

    normalized = series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if normalized.isna().any():
        raise ValueError(f"Column {column_name} contains non-boolean values after normalization.")
    return normalized.astype("boolean")


def _validate_candidate_seeds(config: dict[str, Any]) -> list[int]:
    """Validate the candidate seed list from the modeling config."""
    candidates = config.get("data_split_seed_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("data_split_seed_candidates must be a non-empty list of integers.")
    if any(isinstance(candidate, bool) or not isinstance(candidate, int) for candidate in candidates):
        raise ValueError("data_split_seed_candidates must contain integers only.")
    return candidates


def _validate_locked_seed(config: dict[str, Any]) -> int:
    """Validate the locked split seed according to the Session 5 guard."""
    candidates = _validate_candidate_seeds(config=config)
    seed = config.get("data_split_seed")
    if seed == "__TBD__":
        raise ValueError(
            "data_split_seed is unlocked ('__TBD__'). Review outputs/analysis/data_split_seed_selection.md "
            "and edit configs/modeling_global.yaml to set data_split_seed to an integer from "
            "data_split_seed_candidates."
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(
            f"data_split_seed must be an integer from {candidates}; received value {seed!r} "
            f"of type {type(seed).__name__}."
        )
    if seed not in candidates:
        raise ValueError(f"data_split_seed={seed} is not in data_split_seed_candidates={candidates}.")
    return seed


def _load_final_sample_index(config: dict[str, Any]) -> pd.DataFrame:
    """Load, filter, and sort the final-cohort sample index."""
    sample_index_path = _resolve_path(config["paths"]["sample_index"])
    try:
        sample_index = pd.read_csv(sample_index_path, usecols=REQUIRED_SAMPLE_INDEX_COLUMNS)
    except OSError as exc:
        LOGGER.error("Failed to read sample index at %s.", sample_index_path)
        raise OSError(f"Failed to read sample index at {sample_index_path}.") from exc

    _log_dataframe_summary(name="sample_index_raw", dataframe=sample_index)

    sample_index["in_final_output"] = _normalize_bool_series(sample_index["in_final_output"], "in_final_output")
    sample_index["fallback_used"] = _normalize_bool_series(sample_index["fallback_used"], "fallback_used")
    sample_index["padded"] = _normalize_bool_series(sample_index["padded"], "padded")

    final_sample_index = (
        sample_index.loc[sample_index["in_final_output"].fillna(False)].copy().sort_values("sample_id").reset_index(drop=True)
    )
    if len(final_sample_index) != EXPECTED_FINAL_COHORT_SIZE:
        raise AssertionError(
            f"Expected {EXPECTED_FINAL_COHORT_SIZE} final-cohort rows in sample_index.csv, found {len(final_sample_index)}."
        )

    _log_dataframe_summary(name="sample_index_final", dataframe=final_sample_index)
    return final_sample_index


def _assert_stage1_alignment(sample_index: pd.DataFrame) -> None:
    """Assert Stage 1 sample-id and label alignment across persisted artifacts."""
    pipeline_a_path = PROCESSED_DATA_ROOT / "sequences" / "pipeline_a_raw.npz"
    pipeline_b_path = PROCESSED_DATA_ROOT / "features" / "pipeline_b_raw.csv"

    try:
        with np.load(pipeline_a_path, allow_pickle=False) as pipeline_a:
            pipeline_a_sample_ids = pipeline_a["sample_id"]
            pipeline_a_labels = pipeline_a["label"]
            pipeline_a_x = pipeline_a["x"]
    except OSError as exc:
        LOGGER.error("Failed to read Pipeline A artifact at %s.", pipeline_a_path)
        raise OSError(f"Failed to read Pipeline A artifact at {pipeline_a_path}.") from exc

    _log_array_summary(name="pipeline_a.x", array=pipeline_a_x)
    _log_array_summary(name="pipeline_a.sample_id", array=pipeline_a_sample_ids)
    _log_array_summary(name="pipeline_a.label", array=pipeline_a_labels)

    try:
        pipeline_b = pd.read_csv(pipeline_b_path, usecols=["sample_id", "label"])
    except OSError as exc:
        LOGGER.error("Failed to read Pipeline B artifact at %s.", pipeline_b_path)
        raise OSError(f"Failed to read Pipeline B artifact at {pipeline_b_path}.") from exc

    _log_dataframe_summary(name="pipeline_b_alignment_subset", dataframe=pipeline_b)

    expected_sample_ids = sample_index["sample_id"].to_numpy(dtype=np.int64)
    expected_labels = sample_index["label"].to_numpy(dtype=np.int64)
    pipeline_b_sample_ids = pipeline_b["sample_id"].to_numpy(dtype=np.int64)
    pipeline_b_labels = pipeline_b["label"].to_numpy(dtype=np.int64)

    assert np.array_equal(expected_sample_ids, pipeline_a_sample_ids), (
        "Pipeline A sample_id order does not match data/processed/sample_index.csv final-cohort order."
    )
    assert np.array_equal(expected_sample_ids, pipeline_b_sample_ids), (
        "Pipeline B sample_id order does not match data/processed/sample_index.csv final-cohort order."
    )
    assert np.array_equal(expected_labels, pipeline_a_labels), (
        "Pipeline A labels do not match data/processed/sample_index.csv final-cohort labels."
    )
    assert np.array_equal(expected_labels, pipeline_b_labels), (
        "Pipeline B labels do not match data/processed/sample_index.csv final-cohort labels."
    )


def _load_metadata() -> pd.DataFrame:
    """Load the metadata columns needed for seed evaluation and Table 1."""
    try:
        metadata = pd.read_csv(METADATA_PATH, usecols=list(METADATA_COLUMN_MAP.keys()), encoding="utf-8-sig")
    except OSError as exc:
        LOGGER.error("Failed to read metadata at %s.", METADATA_PATH)
        raise OSError(f"Failed to read metadata at {METADATA_PATH}.") from exc

    metadata = metadata.rename(columns=METADATA_COLUMN_MAP)
    _log_dataframe_summary(name="metadata_subset", dataframe=metadata)
    return metadata


def _prepare_cohort_dataframe(config: dict[str, Any]) -> pd.DataFrame:
    """Prepare the final-cohort dataframe used by Phase 1 analyses."""
    sample_index = _load_final_sample_index(config=config)
    _assert_stage1_alignment(sample_index=sample_index)
    metadata = _load_metadata()

    cohort = sample_index.merge(metadata, on="sample_id", how="left", validate="one_to_one")
    _log_dataframe_summary(name="cohort_merged", dataframe=cohort)

    if cohort[["age", "sex", "bmi", "smoking", "drinking", "family_history", "collection_location"]].isna().any().any():
        raise AssertionError("Merged cohort has missing metadata values in required seed-selection fields.")

    if not cohort["label"].equals(cohort["metadata_label"].astype(int)):
        raise AssertionError("metadata.csv label column does not match sample_index.csv labels on sample_id alignment.")
    if not cohort["disease_category"].equals(cohort["metadata_disease_category"].astype(int)):
        raise AssertionError(
            "metadata.csv disease_category column does not match sample_index.csv disease_category on sample_id alignment."
        )

    hp_missing_count = int(cohort["hp_infection"].isna().sum())
    LOGGER.info(
        "Hp infection missing count in final cohort=%s; treating missingness as an explicit categorical level.",
        hp_missing_count,
    )
    cohort["fallback_used"] = cohort["fallback_used"].astype(bool).astype(int)
    cohort["padded"] = cohort["padded"].astype(bool).astype(int)
    return cohort


def _compute_continuous_smd(train_values: pd.Series, test_values: pd.Series) -> float:
    """Compute the absolute standardized mean difference for a continuous variable."""
    train_array = train_values.astype(float).to_numpy()
    test_array = test_values.astype(float).to_numpy()
    mean_diff = float(np.mean(train_array) - np.mean(test_array))
    pooled_sd = float(np.sqrt((np.var(train_array, ddof=1) + np.var(test_array, ddof=1)) / 2.0))
    if pooled_sd < 1e-8:
        return 0.0 if abs(mean_diff) < 1e-8 else abs(mean_diff)
    return abs(mean_diff) / pooled_sd


def _compute_binary_smd(train_values: pd.Series, test_values: pd.Series) -> float:
    """Compute the absolute SMD for a binary indicator encoded as 0/1."""
    return _compute_continuous_smd(train_values=train_values.astype(int), test_values=test_values.astype(int))


def _compute_multilevel_smd(train_values: pd.Series, test_values: pd.Series, levels: list[int]) -> float:
    """Compute the maximum one-vs-rest SMD across category levels."""
    level_smds = [
        _compute_binary_smd(train_values=(train_values == level), test_values=(test_values == level)) for level in levels
    ]
    return max(level_smds)


def _encode_hp_infection(values: pd.Series) -> pd.Series:
    """Encode Hp infection with an explicit missing category."""
    return values.apply(lambda value: "missing" if pd.isna(value) else str(int(value)))


def _compute_seed_score(cohort: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, seed: int) -> SeedScore:
    """Compute the prespecified imbalance metrics for one candidate seed."""
    train_frame = cohort.iloc[train_idx].reset_index(drop=True)
    test_frame = cohort.iloc[test_idx].reset_index(drop=True)
    train_hp = _encode_hp_infection(train_frame["hp_infection"])
    test_hp = _encode_hp_infection(test_frame["hp_infection"])

    return SeedScore(
        seed=seed,
        label_smd=_compute_binary_smd(train_frame["label"], test_frame["label"]),
        disease_category_smd=_compute_multilevel_smd(
            train_frame["disease_category"],
            test_frame["disease_category"],
            levels=sorted(DISEASE_CATEGORY_LABELS),
        ),
        age_smd=_compute_continuous_smd(train_frame["age"], test_frame["age"]),
        bmi_smd=_compute_continuous_smd(train_frame["bmi"], test_frame["bmi"]),
        sex_smd=_compute_binary_smd(train_frame["sex"], test_frame["sex"]),
        smoking_smd=_compute_binary_smd(train_frame["smoking"], test_frame["smoking"]),
        drinking_smd=_compute_binary_smd(train_frame["drinking"], test_frame["drinking"]),
        family_history_smd=_compute_binary_smd(train_frame["family_history"], test_frame["family_history"]),
        hp_smd=_compute_multilevel_smd(train_hp, test_hp, levels=["0", "1", "missing"]),
        fallback_used_smd=_compute_binary_smd(train_frame["fallback_used"], test_frame["fallback_used"]),
        padded_smd=_compute_binary_smd(train_frame["padded"], test_frame["padded"]),
    )


def _select_best_seed(seed_scores: list[SeedScore]) -> tuple[SeedScore, int]:
    """Select the best seed by the prespecified lexicographic cascade."""
    if not seed_scores:
        raise ValueError("seed_scores must contain at least one SeedScore record.")

    candidate_scores = list(seed_scores)
    level_extractors = {
        1: lambda score: score.label_smd,
        2: lambda score: score.disease_category_smd,
        3: lambda score: max(score.age_smd, score.bmi_smd),
        4: lambda score: max(
            score.sex_smd,
            score.smoking_smd,
            score.drinking_smd,
            score.family_history_smd,
            score.hp_smd,
        ),
        5: lambda score: max(score.fallback_used_smd, score.padded_smd),
    }

    for level, extractor in level_extractors.items():
        best_value = min(extractor(score) for score in candidate_scores)
        candidate_scores = [
            score for score in candidate_scores if abs(extractor(score) - best_value) <= SEED_TIE_TOLERANCE
        ]
        if len(candidate_scores) == 1:
            return candidate_scores[0], level

    return min(candidate_scores, key=lambda score: score.seed), 6


def _selection_level_description(level: int) -> str:
    """Return the plain-English description for the winning cascade level."""
    descriptions = {
        1: "Level 1: label |SMD|",
        2: "Level 2: disease_category max one-vs-rest |SMD|",
        3: "Level 3: max |SMD| across age and BMI",
        4: "Level 4: max |SMD| across sex, smoking, drinking, family history, and Hp infection",
        5: "Level 5: max |SMD| across fallback_used and padded",
        6: "Level 6: lowest seed integer after ties across levels 1-5",
    }
    return descriptions[level]


def _format_seed_table(seed_scores: list[SeedScore]) -> str:
    """Format the per-seed metric table as markdown."""
    header = [
        "seed",
        "label_SMD",
        "disease_category_SMD",
        "age_SMD",
        "BMI_SMD",
        "sex_SMD",
        "smoking_SMD",
        "drinking_SMD",
        "family_history_SMD",
        "Hp_SMD",
        "fallback_used_SMD",
        "padded_SMD",
    ]
    divider = ["---"] * len(header)
    rows = [
        [
            str(score.seed),
            f"{score.label_smd:.6f}",
            f"{score.disease_category_smd:.6f}",
            f"{score.age_smd:.6f}",
            f"{score.bmi_smd:.6f}",
            f"{score.sex_smd:.6f}",
            f"{score.smoking_smd:.6f}",
            f"{score.drinking_smd:.6f}",
            f"{score.family_history_smd:.6f}",
            f"{score.hp_smd:.6f}",
            f"{score.fallback_used_smd:.6f}",
            f"{score.padded_smd:.6f}",
        ]
        for score in sorted(seed_scores, key=lambda item: item.seed)
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(divider) + " |",
        *["| " + " | ".join(row) + " |" for row in rows],
    ]
    return "\n".join(lines)


def _get_git_short_hash() -> str:
    """Return the current git short hash for report provenance."""
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _build_seed_selection_markdown(seed_scores: list[SeedScore], selected_score: SeedScore, selected_level: int) -> str:
    """Build the Phase 1 seed-selection markdown report."""
    timestamp_utc = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    git_short_hash = _get_git_short_hash()
    selected_level_text = _selection_level_description(level=selected_level)
    level_six_notice = (
        "\n> WARNING: selection reached Level 6, so the candidate seeds remained tied through levels 1-5 and the "
        "lowest seed integer was chosen.\n"
        if selected_level == 6
        else ""
    )

    return (
        "# Data Split Seed Selection\n\n"
        f"- UTC timestamp: `{timestamp_utc}`\n"
        f"- Git short hash: `{git_short_hash}`\n"
        "- Prespecification anchor: `modeling-prespecified`\n\n"
        "## Algorithm\n\n"
        "The candidate seeds are evaluated with a six-level lexicographic imbalance cascade where lower values are "
        "better.\n\n"
        "1. Compare the absolute train-vs-test SMD for the binary label.\n"
        "2. If Level 1 is tied within `1e-6`, compare the maximum one-vs-rest absolute SMD across the six "
        "disease-category levels.\n"
        "3. If still tied, compare the maximum absolute SMD across the continuous variables age and BMI.\n"
        "4. If still tied, compare the maximum absolute SMD across the demographic variables sex, smoking, "
        "drinking, first-degree family history of GC, and Hp infection.\n"
        "5. If still tied, compare the maximum absolute SMD across the engineering flags `fallback_used` and `padded`.\n"
        "6. If all five levels remain tied, choose the lowest seed integer.\n"
        f"{level_six_notice}\n"
        "## Per-Seed Score Table\n\n"
        f"{_format_seed_table(seed_scores=seed_scores)}\n\n"
        "## Selection Result\n\n"
        f"- Selected seed: `{selected_score.seed}`\n"
        f"- Selection resolved at: `{selected_level_text}`\n"
        "- Output report path: `outputs/analysis/data_split_seed_selection.md`\n"
        "- Lock target config path: `configs/modeling_global.yaml`\n\n"
        "## Lock Declaration (to be appended in Phase 2)\n\n"
        "Phase 2 will append the lock declaration here after the user edits configs/modeling_global.yaml.\n"
    )


def _write_text(path: Path, content: str) -> None:
    """Write UTF-8 text content to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        LOGGER.error("Failed to write text output to %s.", path)
        raise OSError(f"Failed to write text output to {path}.") from exc


def _save_primary_split_arrays(output_dir: Path, train_idx: np.ndarray, test_idx: np.ndarray) -> None:
    """Persist convenience split files consumed by Session 5 analyses."""
    try:
        np.savez_compressed(output_dir / "train_idx.npz", train_idx=train_idx.astype(int))
        np.savez_compressed(output_dir / "test_idx.npz", test_idx=test_idx.astype(int))
    except OSError as exc:
        LOGGER.error("Failed to save convenience split artifacts under %s.", output_dir)
        raise OSError(f"Failed to save convenience split artifacts under {output_dir}.") from exc


def evaluate_candidate_seeds(config_path: Path) -> None:
    """Evaluate candidate split seeds and write the prespecified markdown report.

    Args:
        config_path: Path to `configs/modeling_global.yaml`.

    Raises:
        AssertionError: If the final-cohort alignment contract fails.
        OSError: If required inputs or outputs cannot be read or written.
        ValueError: If the config schema is invalid.
    """
    config = _load_yaml_config(config_path=config_path)
    candidate_seeds = _validate_candidate_seeds(config=config)
    cohort = _prepare_cohort_dataframe(config=config)

    labels = cohort["label"].to_numpy(dtype=int)
    disease_categories = cohort["disease_category"].to_numpy(dtype=int)
    fold_generator = FoldGenerator(
        train_fraction=float(config["split"]["train_fraction"]),
        outer_n_folds=int(config["cv"]["outer_folds"]),
        inner_n_folds=int(config["cv"]["inner_folds"]),
        dl_holdout_fraction=float(config["dl"]["holdout_fraction"]),
    )

    seed_scores: list[SeedScore] = []
    for seed in candidate_seeds:
        train_idx, test_idx = fold_generator.generate_train_test_split(
            labels=labels,
            disease_categories=disease_categories,
            seed=seed,
        )
        seed_score = _compute_seed_score(cohort=cohort, train_idx=train_idx, test_idx=test_idx, seed=seed)
        seed_scores.append(seed_score)
        LOGGER.info(
            (
                "Seed %s score breakdown: label=%.6f, disease_category=%.6f, age=%.6f, BMI=%.6f, sex=%.6f, "
                "smoking=%.6f, drinking=%.6f, family_history=%.6f, Hp=%.6f, fallback_used=%.6f, padded=%.6f"
            ),
            seed_score.seed,
            seed_score.label_smd,
            seed_score.disease_category_smd,
            seed_score.age_smd,
            seed_score.bmi_smd,
            seed_score.sex_smd,
            seed_score.smoking_smd,
            seed_score.drinking_smd,
            seed_score.family_history_smd,
            seed_score.hp_smd,
            seed_score.fallback_used_smd,
            seed_score.padded_smd,
        )

    selected_score, selected_level = _select_best_seed(seed_scores=seed_scores)
    report_path = _resolve_path(config["paths"]["outputs_analysis_dir"]) / "data_split_seed_selection.md"
    report = _build_seed_selection_markdown(
        seed_scores=seed_scores,
        selected_score=selected_score,
        selected_level=selected_level,
    )
    _write_text(path=report_path, content=report)
    LOGGER.info(
        "Selected data_split_seed=%s at %s and wrote report to %s.",
        selected_score.seed,
        _selection_level_description(level=selected_level),
        _format_path(report_path),
    )


def generate_and_save_splits(config_path: Path) -> None:
    """Generate and persist the locked train/test, nested-CV, and DL holdout splits.

    Args:
        config_path: Path to `configs/modeling_global.yaml`.

    Raises:
        AssertionError: If the Stage 1 alignment contract fails.
        OSError: If split artifacts cannot be written.
        ValueError: If `data_split_seed` is unlocked or invalid.
    """
    config = _load_yaml_config(config_path=config_path)
    locked_seed = _validate_locked_seed(config=config)
    cohort = _prepare_cohort_dataframe(config=config)

    labels = cohort["label"].to_numpy(dtype=int)
    disease_categories = cohort["disease_category"].to_numpy(dtype=int)
    fold_generator = FoldGenerator(
        train_fraction=float(config["split"]["train_fraction"]),
        outer_n_folds=int(config["cv"]["outer_folds"]),
        inner_n_folds=int(config["cv"]["inner_folds"]),
        dl_holdout_fraction=float(config["dl"]["holdout_fraction"]),
    )

    train_idx, test_idx = fold_generator.generate_train_test_split(
        labels=labels,
        disease_categories=disease_categories,
        seed=locked_seed,
    )
    outer_folds = fold_generator.generate_outer_folds(
        train_labels=labels[train_idx],
        train_disease_categories=disease_categories[train_idx],
        seed=int(config["random_seed"]),
    )
    for fold in outer_folds:
        absolute_outer_train_idx = train_idx[fold.train_idx]
        fold_generator.generate_dl_holdout(
            train_idx=absolute_outer_train_idx,
            disease_categories=disease_categories,
            seed=int(config["random_seed"]) + fold.outer_fold,
        )

    # Use a distinct deterministic seed offset for the final Stage 4b holdout. [project-reproducibility-convention]
    final_dl_train_idx, final_dl_holdout_idx = create_dl_holdout_indices(
        train_idx=train_idx,
        disease_categories=disease_categories,
        holdout_fraction=float(config["dl"]["holdout_fraction"]),
        seed=int(config["random_seed"]) + 100,
    )

    splits_dir = _resolve_path(config["paths"]["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)
    _save_primary_split_arrays(output_dir=splits_dir, train_idx=train_idx, test_idx=test_idx)
    fold_generator.save_splits(output_dir=splits_dir)
    save_dl_holdout(
        path=splits_dir / "final_dl_es_holdout.npz",
        train_idx=final_dl_train_idx,
        holdout_idx=final_dl_holdout_idx,
    )
    LOGGER.info("Saved locked split artifacts under %s.", _format_path(splits_dir))
