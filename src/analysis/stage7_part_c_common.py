from __future__ import annotations

from dataclasses import dataclass
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

from src.analysis.stage7_part_a_common import (
    BOOTSTRAP_CONVENTION_TEXT,
    _format_path,
    _get_git_commit_hash,
    _load_global_config,
    _load_model_payload,
    _load_view_prediction_frame,
    _resolve_path,
    _timestamp_utc,
)
from src.analysis.stage7_part_b_common import _write_dataframe_csv
from src.config import PROJECT_ROOT, RANDOM_SEED


LOGGER = logging.getLogger(__name__)
RAW_METADATA_RENAME = {
    "collection_location(0outpatient1inpatient)": "collection_location",
    "disease_category(0HC1BGL2LGIN3HGIN4EGC5AGC)": "disease_category",
    "sex(0male1female)": "sex",
    "BMI": "bmi",
    "smoking(0no1yes)": "smoking",
    "drinking(0no1yes)": "drinking",
    "first_degree_family_history_of_GC(0no1yes)": "family_history",
    "Hp_Infection(0no1yes)": "hp_infection",
    "CEA(0-5μg/L)": "cea",
    "CA199(0.1-37μ/ml)": "ca199",
    "CA724(0.1-10μ/ml)": "ca724",
}
TUMOR_MARKER_COLUMNS = ("cea", "ca199", "ca724")
MARKER_FEATURE_NAMES = ("cea", "ca199", "ca724")
FUSION_FEATURE_NAMES = ("voc_p_raw", "cea", "ca199", "ca724")
SELECTION_BIAS_VARIABLES = (
    "age",
    "sex",
    "bmi",
    "smoking",
    "drinking",
    "family_history",
    "hp_infection",
    "disease_category",
    "collection_location",
    "label",
)


@dataclass(frozen=True)
class TumorMarkerSubsetBundle:
    """Bundle the complete-case train/test subsets and supporting cohort frames."""

    full_cohort: pd.DataFrame
    train_cohort: pd.DataFrame
    test_cohort: pd.DataFrame
    tm_train_subset: pd.DataFrame
    tm_test_subset: pd.DataFrame


@dataclass(frozen=True)
class TumorMarkerModelArtifacts:
    """Persistable outputs from one LR-based tumor-marker model."""

    model_name: str
    feature_names: tuple[str, ...]
    final_model: LogisticRegression
    oof_frame: pd.DataFrame
    test_frame: pd.DataFrame
    calibrator: LogisticRegression
    threshold_p_raw: float
    threshold_p_final_equivalent: float
    model_path: Path
    summary: dict[str, Any]


def _log_dataframe_summary(frame: pd.DataFrame, context: str) -> None:
    """Log shape, dtypes, and missing counts for one dataframe."""
    LOGGER.info(
        "Loaded %s: shape=%s dtypes=%s missing_counts=%s",
        context,
        frame.shape,
        {column: str(dtype) for column, dtype in frame.dtypes.items()},
        frame.isna().sum().to_dict(),
    )


def _read_raw_metadata() -> pd.DataFrame:
    """Read raw metadata and rename to the locked Session 13 aliases."""
    metadata_path = PROJECT_ROOT / "data" / "raw" / "clinical_data" / "metadata.csv"
    try:
        metadata = pd.read_csv(metadata_path)
    except OSError as exc:
        LOGGER.error("Failed to read raw metadata at %s.", metadata_path)
        raise OSError(f"Failed to read raw metadata at {metadata_path}.") from exc

    renamed = metadata.rename(columns=RAW_METADATA_RENAME).copy()
    renamed["sample_id"] = pd.to_numeric(renamed["sample_id"], errors="raise").astype(np.int64)
    renamed["label"] = pd.to_numeric(renamed["label"], errors="raise").astype(np.int64)
    renamed["collection_location"] = pd.to_numeric(renamed["collection_location"], errors="raise").astype(np.int64)
    renamed["disease_category"] = pd.to_numeric(renamed["disease_category"], errors="raise").astype(np.int64)
    renamed["sex"] = pd.to_numeric(renamed["sex"], errors="coerce")
    renamed["age"] = pd.to_numeric(renamed["age"], errors="coerce")
    renamed["bmi"] = pd.to_numeric(renamed["bmi"], errors="coerce")
    renamed["smoking"] = pd.to_numeric(renamed["smoking"], errors="coerce")
    renamed["drinking"] = pd.to_numeric(renamed["drinking"], errors="coerce")
    renamed["family_history"] = pd.to_numeric(renamed["family_history"], errors="coerce")
    renamed["hp_infection"] = pd.to_numeric(renamed["hp_infection"], errors="coerce")
    renamed["cea"] = pd.to_numeric(renamed["cea"], errors="coerce")
    renamed["ca199"] = _coerce_ca199_series(renamed["ca199"])
    renamed["ca724"] = pd.to_numeric(renamed["ca724"], errors="coerce")
    _log_dataframe_summary(renamed, "raw metadata with Session 13 aliases")
    return renamed


def _load_final_sample_index(config: dict[str, Any]) -> pd.DataFrame:
    """Load the locked final-cohort sample index."""
    sample_index_path = _resolve_path(config["paths"]["sample_index"])
    try:
        sample_index = pd.read_csv(sample_index_path)
    except OSError as exc:
        LOGGER.error("Failed to read sample index at %s.", sample_index_path)
        raise OSError(f"Failed to read sample index at {sample_index_path}.") from exc

    final_sample_index = (
        sample_index.loc[sample_index["in_final_output"] == True]
        .copy()
        .sort_values("sample_id", kind="mergesort")
        .reset_index(drop=True)
    )
    final_sample_index["sample_id"] = pd.to_numeric(final_sample_index["sample_id"], errors="raise").astype(np.int64)
    if len(final_sample_index) != 972:
        raise AssertionError(f"Expected 972 rows in the final cohort sample index, found {len(final_sample_index)}.")
    _log_dataframe_summary(final_sample_index, "final sample index")
    return final_sample_index


def _coerce_ca199_series(series: pd.Series) -> pd.Series:
    """Coerce CA19-9 values with European decimals into float, otherwise NaN."""
    string_values = series.astype("string")
    cleaned = string_values.str.replace(",", ".", regex=False)
    return pd.to_numeric(cleaned, errors="coerce").astype(float)


def _compute_complete_case_mask(frame: pd.DataFrame) -> pd.Series:
    """Return the locked complete-case mask for all three tumor markers."""
    return frame.loc[:, list(TUMOR_MARKER_COLUMNS)].notna().all(axis=1)


def _partition_frame_by_indices(
    frame: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition one final-cohort dataframe using locked split indices."""
    train_frame = frame.iloc[train_idx].copy().reset_index(drop=True)
    test_frame = frame.iloc[test_idx].copy().reset_index(drop=True)
    return train_frame, test_frame


def _build_complete_case_subsets(
    cohort_frame: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply locked train/test indices then the tumor-marker complete-case filter."""
    train_frame, test_frame = _partition_frame_by_indices(frame=cohort_frame, train_idx=train_idx, test_idx=test_idx)
    train_subset = train_frame.loc[_compute_complete_case_mask(train_frame)].copy().reset_index(drop=True)
    test_subset = test_frame.loc[_compute_complete_case_mask(test_frame)].copy().reset_index(drop=True)
    return train_subset, test_subset


def _load_primary_split_indices(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Load the locked train/test split indices."""
    splits_dir = _resolve_path(config["paths"]["splits_dir"])
    try:
        with np.load(splits_dir / "train_idx.npz", allow_pickle=False) as train_npz:
            train_idx = train_npz["train_idx"].astype(np.int64)
        with np.load(splits_dir / "test_idx.npz", allow_pickle=False) as test_npz:
            test_idx = test_npz["test_idx"].astype(np.int64)
    except OSError as exc:
        LOGGER.error("Failed to read locked primary split arrays under %s.", splits_dir)
        raise OSError(f"Failed to read locked primary split arrays under {splits_dir}.") from exc
    return train_idx, test_idx


def _load_outer_fold_definitions(config: dict[str, Any]) -> list[dict[str, np.ndarray]]:
    """Load the five locked outer-fold artifacts used for restricted OOF fits."""
    splits_dir = _resolve_path(config["paths"]["splits_dir"])
    folds: list[dict[str, np.ndarray]] = []
    for outer_fold in range(int(config["cv"]["outer_folds"])):
        path = splits_dir / f"outer_fold_{outer_fold}.npz"
        try:
            with np.load(path, allow_pickle=True) as fold_npz:
                folds.append(
                    {
                        "outer_fold": fold_npz["outer_fold"].astype(np.int64),
                        "train_idx": fold_npz["train_idx"].astype(np.int64),
                        "val_idx": fold_npz["val_idx"].astype(np.int64),
                    }
                )
        except OSError as exc:
            LOGGER.error("Failed to read outer-fold artifact at %s.", path)
            raise OSError(f"Failed to read outer-fold artifact at {path}.") from exc
    return folds


def _prepare_final_cohort_frame(config: dict[str, Any]) -> pd.DataFrame:
    """Build the locked 972-row final cohort with Session 13 covariates."""
    metadata = _read_raw_metadata()
    final_sample_index = _load_final_sample_index(config)
    final_sample_ids = final_sample_index["sample_id"].to_numpy(dtype=np.int64)
    filtered = (
        metadata.loc[metadata["sample_id"].isin(final_sample_ids)]
        .copy()
        .sort_values("sample_id", kind="mergesort")
        .reset_index(drop=True)
    )
    if not np.array_equal(filtered["sample_id"].to_numpy(dtype=np.int64), final_sample_ids):
        raise AssertionError("Final metadata sample_id ordering does not match the locked sample index ordering.")
    _log_dataframe_summary(filtered, "final 972-row cohort metadata")
    return filtered


def _attach_voc_prediction_columns(
    *,
    best_model: str,
    tm_train_subset: pd.DataFrame,
    tm_test_subset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join VOC OOF raw predictions to train subset and test raw/calibrated predictions to test subset."""
    voc_oof = _load_view_prediction_frame(model_name=best_model, view_name="nested_cv_oof").rename(
        columns={"p_raw": "voc_p_raw", "y_true": "label"}
    )
    voc_test = _load_view_prediction_frame(model_name=best_model, view_name="test").rename(
        columns={"p_raw": "voc_p_raw", "p_cal": "voc_p_cal", "y_true": "label"}
    )

    train_merged = tm_train_subset.merge(
        voc_oof.loc[:, ["sample_id", "label", "voc_p_raw"]],
        on=["sample_id", "label"],
        how="left",
        validate="one_to_one",
    )
    test_merged = tm_test_subset.merge(
        voc_test.loc[:, ["sample_id", "label", "voc_p_raw", "voc_p_cal"]],
        on=["sample_id", "label"],
        how="left",
        validate="one_to_one",
    )
    if train_merged["voc_p_raw"].isna().any():
        raise AssertionError("VOC OOF merge left missing voc_p_raw values in tm_train_subset.")
    if test_merged[["voc_p_raw", "voc_p_cal"]].isna().any().any():
        raise AssertionError("VOC test merge left missing voc_p_raw or voc_p_cal values in tm_test_subset.")
    return train_merged, test_merged


def _assemble_tumor_marker_subsets(config_path: Path, best_model: str) -> TumorMarkerSubsetBundle:
    """Assemble the locked complete-case tumor-marker train/test subsets."""
    config = _load_global_config(config_path=config_path)
    cohort_frame = _prepare_final_cohort_frame(config)
    train_idx, test_idx = _load_primary_split_indices(config)
    train_cohort, test_cohort = _partition_frame_by_indices(frame=cohort_frame, train_idx=train_idx, test_idx=test_idx)
    tm_train_subset, tm_test_subset = _build_complete_case_subsets(
        cohort_frame=cohort_frame,
        train_idx=train_idx,
        test_idx=test_idx,
    )
    tm_train_subset, tm_test_subset = _attach_voc_prediction_columns(
        best_model=best_model,
        tm_train_subset=tm_train_subset,
        tm_test_subset=tm_test_subset,
    )
    LOGGER.info(
        "Assembled tumor-marker subsets for model=%s: tm_train_subset_n=%s tm_train_pos=%s tm_test_subset_n=%s tm_test_pos=%s.",
        best_model,
        len(tm_train_subset),
        int(tm_train_subset["label"].sum()),
        len(tm_test_subset),
        int(tm_test_subset["label"].sum()),
    )
    return TumorMarkerSubsetBundle(
        full_cohort=cohort_frame,
        train_cohort=train_cohort,
        test_cohort=test_cohort,
        tm_train_subset=tm_train_subset,
        tm_test_subset=tm_test_subset,
    )


def _assert_expected_subset_sizes(tm_train_subset: pd.DataFrame, tm_test_subset: pd.DataFrame) -> None:
    """Validate the locked expected subset counts within the allowed tolerance."""
    if abs(len(tm_train_subset) - 516) > 2:
        raise AssertionError(f"tm_train_subset n_total drifted from 516 by more than 2: {len(tm_train_subset)}")
    if abs(int(tm_train_subset["label"].sum()) - 293) > 2:
        raise AssertionError(
            f"tm_train_subset n_pos drifted from 293 by more than 2: {int(tm_train_subset['label'].sum())}"
        )
    if abs(len(tm_test_subset) - 135) > 2:
        raise AssertionError(f"tm_test_subset n_total drifted from 135 by more than 2: {len(tm_test_subset)}")
    if abs(int(tm_test_subset["label"].sum()) - 76) > 2:
        raise AssertionError(
            f"tm_test_subset n_pos drifted from 76 by more than 2: {int(tm_test_subset['label'].sum())}"
        )


def _assert_no_nan_matrix(x_matrix: np.ndarray, context: str) -> None:
    """Fail fast if a tumor-marker design matrix contains NaN."""
    if np.isnan(x_matrix).any():
        raise AssertionError(f"{context} contains NaN values; tumor-marker analysis must remain complete-case only.")


def _create_logistic_regression_estimator() -> LogisticRegression:
    """Create the locked Session 13 logistic-regression estimator."""
    return LogisticRegression(
        class_weight="balanced",
        random_state=RANDOM_SEED,
        max_iter=1000,
        solver="lbfgs",
    )


def _fit_logistic_regression_model(
    estimator: Any,
    feature_frame: pd.DataFrame,
    y_true: np.ndarray,
) -> Any:
    """Fit one estimator after enforcing the no-NaN complete-case contract."""
    x_matrix = feature_frame.to_numpy(dtype=float, copy=True)
    _assert_no_nan_matrix(x_matrix, "Tumor-marker feature matrix passed to fit()")
    fitted_estimator = estimator.fit(x_matrix, y_true.astype(np.int64, copy=False))
    return fitted_estimator


def _predict_logistic_regression_probabilities(estimator: LogisticRegression, feature_frame: pd.DataFrame) -> np.ndarray:
    """Predict positive-class probabilities with a no-NaN guard."""
    x_matrix = feature_frame.to_numpy(dtype=float, copy=True)
    _assert_no_nan_matrix(x_matrix, "Tumor-marker feature matrix passed to predict_proba()")
    probabilities = estimator.predict_proba(x_matrix)[:, 1]
    return np.asarray(probabilities, dtype=float)


def _fit_youden_threshold(y_true: np.ndarray, p_raw: np.ndarray) -> float:
    """Fit the locked raw-space Youden threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, p_raw)
    youden = tpr - fpr
    finite_mask = np.isfinite(thresholds)
    if finite_mask.any():
        best_value = np.max(youden[finite_mask])
        best_idx = np.flatnonzero(finite_mask & (youden == best_value))[0]
    else:
        best_idx = int(np.argmax(youden))
    return float(thresholds[best_idx])


def _build_outer_fold_oof_predictions(
    *,
    train_cohort: pd.DataFrame,
    tm_train_subset: pd.DataFrame,
    feature_names: tuple[str, ...],
    outer_folds: list[dict[str, np.ndarray]],
) -> pd.DataFrame:
    """Generate restricted OOF predictions using the locked outer folds."""
    oof_rows: list[dict[str, Any]] = []
    train_lookup = train_cohort.loc[:, ["sample_id"]].copy()
    for fold in outer_folds:
        outer_fold = int(fold["outer_fold"][0])
        fold_train_sample_ids = set(train_lookup.iloc[fold["train_idx"]]["sample_id"].to_numpy(dtype=np.int64))
        fold_val_sample_ids = set(train_lookup.iloc[fold["val_idx"]]["sample_id"].to_numpy(dtype=np.int64))
        restricted_train = (
            tm_train_subset.loc[tm_train_subset["sample_id"].isin(fold_train_sample_ids)]
            .copy()
            .sort_values("sample_id", kind="mergesort")
            .reset_index(drop=True)
        )
        restricted_val = (
            tm_train_subset.loc[tm_train_subset["sample_id"].isin(fold_val_sample_ids)]
            .copy()
            .sort_values("sample_id", kind="mergesort")
            .reset_index(drop=True)
        )
        if restricted_train.empty or restricted_val.empty:
            raise AssertionError(f"Restricted outer fold {outer_fold} produced an empty train or validation subset.")
        if restricted_train["label"].nunique() != 2:
            raise AssertionError(f"Restricted outer fold {outer_fold} lost class balance in the training subset.")

        estimator = _create_logistic_regression_estimator()
        _fit_logistic_regression_model(
            estimator=estimator,
            feature_frame=restricted_train.loc[:, list(feature_names)],
            y_true=restricted_train["label"].to_numpy(dtype=np.int64),
        )
        fold_probabilities = _predict_logistic_regression_probabilities(
            estimator=estimator,
            feature_frame=restricted_val.loc[:, list(feature_names)],
        )
        for sample_id, label, p_raw in zip(
            restricted_val["sample_id"].to_numpy(dtype=np.int64),
            restricted_val["label"].to_numpy(dtype=np.int64),
            fold_probabilities,
            strict=True,
        ):
            oof_rows.append(
                {
                    "sample_id": int(sample_id),
                    "y_true": int(label),
                    "p_raw": float(p_raw),
                    "outer_fold": outer_fold,
                }
            )

    oof_frame = pd.DataFrame(oof_rows).sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    if len(oof_frame) != len(tm_train_subset):
        raise AssertionError(
            f"Restricted OOF row count mismatch: expected {len(tm_train_subset)}, found {len(oof_frame)}."
        )
    if oof_frame["sample_id"].duplicated().any():
        raise AssertionError("Restricted OOF generation produced duplicate sample_id rows.")
    return oof_frame


def _compute_continuous_smd(group_a: pd.Series, group_b: pd.Series) -> float:
    """Compute the absolute standardized mean difference for a continuous variable."""
    values_a = group_a.dropna().astype(float).to_numpy()
    values_b = group_b.dropna().astype(float).to_numpy()
    if len(values_a) == 0 or len(values_b) == 0:
        return float(np.nan)
    mean_difference = float(np.mean(values_a) - np.mean(values_b))
    pooled_sd = float(np.sqrt((np.var(values_a, ddof=1) + np.var(values_b, ddof=1)) / 2.0))
    if pooled_sd < 1e-8:
        return 0.0 if abs(mean_difference) < 1e-8 else abs(mean_difference)
    return abs(mean_difference) / pooled_sd


def _compute_binary_smd(group_a: pd.Series, group_b: pd.Series) -> float:
    """Compute the absolute standardized mean difference for a binary variable."""
    values_a = group_a.dropna().astype(float).to_numpy()
    values_b = group_b.dropna().astype(float).to_numpy()
    if len(values_a) == 0 or len(values_b) == 0:
        return float(np.nan)
    p_a = float(np.mean(values_a))
    p_b = float(np.mean(values_b))
    pooled = float(np.sqrt(((p_a * (1.0 - p_a)) + (p_b * (1.0 - p_b))) / 2.0))
    if pooled < 1e-8:
        return 0.0 if abs(p_a - p_b) < 1e-8 else abs(p_a - p_b)
    return abs(p_a - p_b) / pooled


def _selection_bias_value_series(frame: pd.DataFrame, variable_name: str) -> pd.Series:
    """Return the SMD-ready value series for one prespecified selection-bias variable."""
    if variable_name == "hp_infection":
        return frame["hp_infection"].dropna().astype(float).eq(1.0).astype(float)
    return frame[variable_name]


def _selection_bias_summary_value(frame: pd.DataFrame, variable_name: str) -> float:
    """Return the reported summary value for one selection-bias variable."""
    series = _selection_bias_value_series(frame, variable_name)
    if variable_name in {"age", "bmi", "disease_category"}:
        return float(series.dropna().astype(float).mean())
    return float(series.dropna().astype(float).mean())


def _compute_selection_bias_rows(
    tm_test_subset: pd.DataFrame,
    full_test_frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Compute the locked 10-variable SMD table for tumor-marker selection bias."""
    rows: list[dict[str, Any]] = []
    for variable_name in SELECTION_BIAS_VARIABLES:
        tm_series = _selection_bias_value_series(tm_test_subset, variable_name)
        full_series = _selection_bias_value_series(full_test_frame, variable_name)
        if variable_name in {"age", "bmi", "disease_category"}:
            smd = _compute_continuous_smd(tm_series, full_series)
        else:
            smd = _compute_binary_smd(tm_series, full_series)
        rows.append(
            {
                "variable": variable_name,
                "value_in_tm_subset": _selection_bias_summary_value(tm_test_subset, variable_name),
                "value_in_full_test": _selection_bias_summary_value(full_test_frame, variable_name),
                "smd": float(smd),
                "n_non_missing_tm": int(tm_series.notna().sum()),
                "n_non_missing_full": int(full_series.notna().sum()),
                "interpretation": "SMD >= 0.1 is conventionally a notable imbalance.",
            }
        )
    return rows


def _save_final_pkl_compliant(path: Path, payload: dict[str, Any]) -> None:
    """Write a Stage 4b-style PKL payload using the Session 10 serialization pattern."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("wb") as handle:
            pickle.dump(payload, handle)
    except OSError as exc:
        LOGGER.error("Failed to write PKL output to %s.", path)
        raise OSError(f"Failed to write PKL output to {path}.") from exc
    LOGGER.info("Wrote PKL artifact to %s.", _format_path(path))


def _build_final_pkl_payload(
    *,
    model_name: str,
    final_model: LogisticRegression,
    calibrator: LogisticRegression,
    threshold_p_raw: float,
    threshold_p_final_equivalent: float,
    feature_names: tuple[str, ...],
    training_sample_ids: np.ndarray,
) -> dict[str, Any]:
    """Build the locked 9-key final-model PKL payload for Session 13 LR models."""
    return {
        "transformer": None,
        "normalizer": None,
        "model": final_model,
        "calibrator": calibrator,
        "threshold_p_raw": float(threshold_p_raw),
        "threshold_p_final_equivalent": float(threshold_p_final_equivalent),
        "feature_names": list(feature_names),
        "training_sample_ids": training_sample_ids.astype(np.int64, copy=False),
        "metadata": {
            "model_name": model_name,
            "final_hparams": {
                "class_weight": "balanced",
                "random_state": RANDOM_SEED,
                "max_iter": 1000,
                "solver": "lbfgs",
            },
            "oof_source": "restricted_locked_outer_folds",
            "git_commit_hash": _get_git_commit_hash(),
            "training_timestamp_utc": _timestamp_utc(),
        },
    }


__all__ = [
    "BOOTSTRAP_CONVENTION_TEXT",
    "FUSION_FEATURE_NAMES",
    "MARKER_FEATURE_NAMES",
    "PROJECT_ROOT",
    "RAW_METADATA_RENAME",
    "RANDOM_SEED",
    "SELECTION_BIAS_VARIABLES",
    "TUMOR_MARKER_COLUMNS",
    "TumorMarkerModelArtifacts",
    "TumorMarkerSubsetBundle",
    "_assemble_tumor_marker_subsets",
    "_assert_expected_subset_sizes",
    "_assert_no_nan_matrix",
    "_build_complete_case_subsets",
    "_build_final_pkl_payload",
    "_build_outer_fold_oof_predictions",
    "_coerce_ca199_series",
    "_compute_binary_smd",
    "_compute_complete_case_mask",
    "_compute_continuous_smd",
    "_compute_selection_bias_rows",
    "_create_logistic_regression_estimator",
    "_fit_logistic_regression_model",
    "_fit_youden_threshold",
    "_load_global_config",
    "_load_model_payload",
    "_load_outer_fold_definitions",
    "_load_primary_split_indices",
    "_load_view_prediction_frame",
    "_partition_frame_by_indices",
    "_predict_logistic_regression_probabilities",
    "_save_final_pkl_compliant",
    "_timestamp_utc",
    "_write_dataframe_csv",
]
