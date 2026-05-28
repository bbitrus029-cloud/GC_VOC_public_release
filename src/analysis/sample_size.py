"""Post-hoc supplementary sample-size reconstruction for Plan 1 Session 1.

This module reconstructs Riley's binary-outcome B1-B4 sample-size criteria from
the prospectively registered inputs locked in Plan 1 §3.3 and produces the
structured outputs required by Session 1 Sub-phases D and E.
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from src.analysis.data_split import _format_path, _load_yaml_config, _resolve_path, _write_text
from src.config import PROJECT_ROOT, RANDOM_SEED


LOGGER = logging.getLogger(__name__)

POST_HOC_DECLARATION = (
    "This analysis is post-hoc, conducted after the `modeling-frozen` tag, and is provided as "
    "supplementary evidence only. It does not modify or replace the prespecified primary modeling "
    "analysis, selected model, model parameters, thresholds, calibrators, normalizers, transformers, "
    "train/test split, fold structure, or any primary-analysis artifact."
)
MMOE_DISCLOSURE_TEXT = (
    "The prospective registration specified mmoe = 1.1, which is a continuous-outcome pmsampsize "
    "option and is non-operative for the binary Riley B1-B4 criteria used in this reconstruction. "
    "The value is preserved in the registration record for audit completeness but does not enter "
    "the computation."
)
SHORTFALL_LIMITATION_TEXT = (
    "This shortfall is documented as a Discussion-section limitation per project-conventions.mdc "
    "post-hoc condition 3; it does not modify any frozen primary-analysis artifact."
)

ACTUAL_COHORT_N = 972  # [domain-physical-constant: frozen Stage 1 / modeling run artifacts]
ACTUAL_COHORT_EVENTS = 413  # [domain-physical-constant: frozen Stage 1 / modeling run artifacts]
ACTUAL_COHORT_EPV = 16.5  # [runtime-determined-by-prespecified-algorithm: cohort event count divided by registered candidate predictor parameters]
FROZEN_POST_PLATT_BRIER = 0.1211  # [runtime-determined-by-prespecified-algorithm: docs/modeling/validated_modeling_run_summary.md]
FROZEN_CALIBRATION_SLOPE = 0.9046  # [runtime-determined-by-prespecified-algorithm: docs/modeling/validated_modeling_run_summary.md]
FROZEN_CALIBRATION_INTERCEPT = 0.3397  # [runtime-determined-by-prespecified-algorithm: docs/modeling/validated_modeling_run_summary.md]
NORMAL_QUANTILE_975 = 1.96  # [ml-methodology-literature: conventional two-sided 95% normal quantile]
REGISTERED_MMOE = 1.1  # [prospectively-registered-protocol-parameter; non-operative for binary B1-B4]

TOP_LEVEL_CONFIG_KEYS = {
    "provenance_schema_version",
    "stage",
    "best_model",
    "random_seed",
    "temporal_anchor_tag",
    "sample_size",
    "optimism_correction",
    "threshold_uncertainty",
    "calibration_tests",
    "paths",
}
SAMPLE_SIZE_CONFIG_KEYS = {
    "anticipated_auc",
    "candidate_predictor_parameters",
    "anticipated_prevalence",
    "target_shrinkage",
    "target_mape",
    "prevalence_margin_of_error",
    "target_r2_nagelkerke_optimism",
    "cox_snell_simulation_n",
}
OPTIMISM_CORRECTION_CONFIG_KEYS = {"n_boot", "metric"}
THRESHOLD_UNCERTAINTY_CONFIG_KEYS = {"n_boot", "ci_percentiles", "tie_break"}
CALIBRATION_TEST_CONFIG_KEYS = {"hosmer_lemeshow_groups"}
PATHS_CONFIG_KEYS = {
    "outputs_analysis_dir",
    "outputs_logs_dir",
    "outputs_figures_modeling_dir",
    "post_hoc_docs",
}


@dataclass(frozen=True)
class SampleSizeResult:
    """Structured sample-size reconstruction result for downstream reporting.

    Attributes:
        locked_inputs: Ordered mapping of locked inputs used in the reconstruction.
        r2_cs: Simulated Cox-Snell R-squared.
        r2_cs_max: Maximum attainable Cox-Snell R-squared at the locked prevalence.
        n_b1: Riley criterion B1 required sample size.
        n_b2: van Smeden / Riley criterion B2 required sample size.
        n_b3: Riley criterion B3 required sample size.
        n_b4: Riley criterion B4 required sample size.
        n_required: Maximum of the four criteria.
        required_events: Ceiling of `n_required * anticipated_prevalence`.
        required_epv: Required events per candidate predictor parameter.
        cohort_meets_all_criteria: Whether the actual cohort meets all four criteria.
        shortfall_criteria: Unmet criteria with their numerical shortfalls.
    """

    locked_inputs: dict[str, float | int]
    r2_cs: float
    r2_cs_max: float
    n_b1: int
    n_b2: int
    n_b3: int
    n_b4: int
    n_required: int
    required_events: int
    required_epv: float
    cohort_meets_all_criteria: bool
    shortfall_criteria: list[dict[str, str | int]]

    def to_json_payload(
        self,
        *,
        provenance_schema_version: int,
        stage: str,
        best_model: str,
        random_seed: int,
        temporal_anchor_tag: str,
        generated_at_utc: str,
        git_commit_hash: str,
    ) -> dict[str, Any]:
        """Convert the result into the JSON payload required by Sub-phase E.

        Args:
            provenance_schema_version: Locked payload schema version.
            stage: Locked stage identifier.
            best_model: Locked best-model name.
            random_seed: Locked reproducibility seed.
            temporal_anchor_tag: Locked temporal anchor tag.
            generated_at_utc: ISO-8601 UTC timestamp captured at script entry.
            git_commit_hash: Current `HEAD` commit hash.

        Returns:
            JSON-serializable payload with deterministic key/value content.
        """
        return {
            "provenance_schema_version": provenance_schema_version,
            "stage": stage,
            "best_model": best_model,
            "random_seed": random_seed,
            "temporal_anchor_tag": temporal_anchor_tag,
            "generated_at_utc": generated_at_utc,
            "git_commit_hash": git_commit_hash,
            "locked_inputs": self.locked_inputs,
            "r2_cs": self.r2_cs,
            "r2_cs_max": self.r2_cs_max,
            "n_b1": self.n_b1,
            "n_b2": self.n_b2,
            "n_b3": self.n_b3,
            "n_b4": self.n_b4,
            "n_required": self.n_required,
            "required_events": self.required_events,
            "required_epv": self.required_epv,
            "actual_cohort": {
                "n": ACTUAL_COHORT_N,
                "events": ACTUAL_COHORT_EVENTS,
                "epv": ACTUAL_COHORT_EPV,
            },
            "cohort_meets_all_criteria": self.cohort_meets_all_criteria,
            "shortfall_criteria": self.shortfall_criteria,
            "mmoe_disclosure_flag": True,
        }


def _resolve_config_path(config_path: Path) -> Path:
    """Resolve a config path relative to the project root when needed."""
    return config_path if config_path.is_absolute() else PROJECT_ROOT / config_path


def _timestamp_utc() -> str:
    """Return an ISO-8601 UTC timestamp with trailing `Z`."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _get_git_commit_hash() -> str:
    """Return the current full git commit hash."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a UTF-8 JSON file with deterministic formatting.

    Args:
        path: Output path to create or overwrite.
        payload: JSON-serializable mapping.

    Raises:
        OSError: If the file cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    except OSError as exc:
        LOGGER.error("Failed to write JSON output to %s.", path)
        raise OSError(f"Failed to write JSON output to {path}.") from exc


def _validate_exact_keys(mapping: dict[str, Any], expected_keys: set[str], context: str) -> None:
    """Assert that one mapping contains exactly the required keys.

    Args:
        mapping: Mapping to validate.
        expected_keys: Required key set.
        context: Human-readable context for error messages.

    Raises:
        ValueError: If any key is missing or unknown.
    """
    missing_keys = sorted(expected_keys.difference(mapping))
    unknown_keys = sorted(set(mapping).difference(expected_keys))
    if missing_keys:
        raise ValueError(f"{context} is missing required keys: {missing_keys}.")
    if unknown_keys:
        raise ValueError(f"{context} contains unknown keys: {unknown_keys}.")


def _require_int(mapping: dict[str, Any], key: str, context: str) -> int:
    """Return one integer config value with bool rejection."""
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context}.{key} must be an integer, received {type(value).__name__}.")
    return value


def _require_float(mapping: dict[str, Any], key: str, context: str) -> float:
    """Return one numeric config value as float with bool rejection."""
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context}.{key} must be numeric, received {type(value).__name__}.")
    return float(value)


def _require_str(mapping: dict[str, Any], key: str, context: str) -> str:
    """Return one string config value."""
    value = mapping[key]
    if not isinstance(value, str):
        raise TypeError(f"{context}.{key} must be a string, received {type(value).__name__}.")
    return value


def load_post_hoc_config(config_path: Path) -> dict[str, Any]:
    """Load and validate the exact Plan 1 post-hoc supplementary config.

    Args:
        config_path: Path to `configs/post_hoc_supplementary.yaml`.

    Returns:
        Validated config mapping.

    Raises:
        ValueError: If required keys are missing, unknown keys exist, or locked
            literal values do not match Plan 1 §5.
        TypeError: If config field types are invalid.
    """
    resolved_config_path = _resolve_config_path(config_path)
    config = _load_yaml_config(resolved_config_path)
    _validate_exact_keys(config, TOP_LEVEL_CONFIG_KEYS, "post_hoc_supplementary config")
    _validate_exact_keys(config["sample_size"], SAMPLE_SIZE_CONFIG_KEYS, "post_hoc_supplementary.sample_size")
    _validate_exact_keys(
        config["optimism_correction"],
        OPTIMISM_CORRECTION_CONFIG_KEYS,
        "post_hoc_supplementary.optimism_correction",
    )
    _validate_exact_keys(
        config["threshold_uncertainty"],
        THRESHOLD_UNCERTAINTY_CONFIG_KEYS,
        "post_hoc_supplementary.threshold_uncertainty",
    )
    _validate_exact_keys(
        config["calibration_tests"],
        CALIBRATION_TEST_CONFIG_KEYS,
        "post_hoc_supplementary.calibration_tests",
    )
    _validate_exact_keys(config["paths"], PATHS_CONFIG_KEYS, "post_hoc_supplementary.paths")

    provenance_schema_version = _require_int(config, "provenance_schema_version", "post_hoc_supplementary")
    random_seed = _require_int(config, "random_seed", "post_hoc_supplementary")
    _require_str(config, "stage", "post_hoc_supplementary")
    _require_str(config, "best_model", "post_hoc_supplementary")
    _require_str(config, "temporal_anchor_tag", "post_hoc_supplementary")

    for key in SAMPLE_SIZE_CONFIG_KEYS:
        if key == "candidate_predictor_parameters" or key == "cox_snell_simulation_n":
            _require_int(config["sample_size"], key, "post_hoc_supplementary.sample_size")
        else:
            _require_float(config["sample_size"], key, "post_hoc_supplementary.sample_size")

    _require_int(config["optimism_correction"], "n_boot", "post_hoc_supplementary.optimism_correction")
    _require_int(config["threshold_uncertainty"], "n_boot", "post_hoc_supplementary.threshold_uncertainty")
    percentiles = config["threshold_uncertainty"]["ci_percentiles"]
    if not isinstance(percentiles, list) or len(percentiles) != 2:
        raise TypeError("post_hoc_supplementary.threshold_uncertainty.ci_percentiles must be a two-element list.")
    for percentile in percentiles:
        if isinstance(percentile, bool) or not isinstance(percentile, (int, float)):
            raise TypeError("ci_percentiles values must be numeric.")
    _require_int(config["calibration_tests"], "hosmer_lemeshow_groups", "post_hoc_supplementary.calibration_tests")

    if provenance_schema_version != 1:
        raise ValueError(f"provenance_schema_version must equal 1, received {provenance_schema_version}.")
    if random_seed != RANDOM_SEED:
        raise ValueError(
            f"random_seed must equal src.config.RANDOM_SEED={RANDOM_SEED}, received {random_seed}."
        )
    if config["stage"] != "post_hoc_supplementary_plan_1":
        raise ValueError(f"stage must equal 'post_hoc_supplementary_plan_1', received {config['stage']!r}.")
    if config["best_model"] != "minirocket_ridge":
        raise ValueError(f"best_model must equal 'minirocket_ridge', received {config['best_model']!r}.")
    if config["temporal_anchor_tag"] != "modeling-frozen":
        raise ValueError(
            f"temporal_anchor_tag must equal 'modeling-frozen', received {config['temporal_anchor_tag']!r}."
        )
    if config["optimism_correction"]["metric"] != "auc_roc":
        raise ValueError("optimism_correction.metric must equal 'auc_roc'.")
    if config["threshold_uncertainty"]["tie_break"] != "smallest_threshold":
        raise ValueError("threshold_uncertainty.tie_break must equal 'smallest_threshold'.")

    return config


def _bernoulli_deviance(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Compute Bernoulli deviance for one vector of probabilities."""
    tiny = np.finfo(float).tiny
    clipped = np.clip(np.asarray(probabilities, dtype=float), tiny, 1.0 - np.finfo(float).eps)
    y_float = np.asarray(y_true, dtype=float)
    log_likelihood = np.sum(y_float * np.log(clipped) + (1.0 - y_float) * np.log(1.0 - clipped))
    return float(-2.0 * log_likelihood)


def _simulate_cox_snell_r_squared(
    *,
    anticipated_auc: float,
    anticipated_prevalence: float,
    cox_snell_simulation_n: int,
    random_seed: int,
) -> float:
    """Simulate Cox-Snell R-squared via the Plan 1 §3.3 procedure.

    Args:
        anticipated_auc: Prospectively registered anticipated AUC.
        anticipated_prevalence: Prospectively registered anticipated prevalence.
        cox_snell_simulation_n: Locked simulation size.
        random_seed: Locked reproducibility seed.

    Returns:
        Deterministic Cox-Snell R-squared estimate.

    Raises:
        ValueError: If inputs are outside their valid ranges or if the result is
            non-finite / non-positive.
    """
    if not 0.0 < anticipated_auc < 1.0:
        raise ValueError(f"anticipated_auc must lie in (0, 1); received {anticipated_auc}.")
    if not 0.0 < anticipated_prevalence < 1.0:
        raise ValueError(f"anticipated_prevalence must lie in (0, 1); received {anticipated_prevalence}.")
    if cox_snell_simulation_n <= 0:
        raise ValueError(f"cox_snell_simulation_n must be positive, received {cox_snell_simulation_n}.")

    rng = np.random.default_rng(random_seed)
    mu = math.sqrt(2.0) * norm.ppf(anticipated_auc)  # [ml-methodology-literature: van Calster c-statistic to Cox-Snell approximation under normal score distributions]
    n_events = math.ceil(cox_snell_simulation_n * anticipated_prevalence)
    n_nonevents = cox_snell_simulation_n - n_events
    event_scores = rng.normal(loc=mu, scale=1.0, size=n_events)
    nonevent_scores = rng.normal(loc=0.0, scale=1.0, size=n_nonevents)
    linear_predictor_scores = np.concatenate([event_scores, nonevent_scores]).reshape(-1, 1)
    y_true = np.concatenate(
        [
            np.ones(n_events, dtype=np.int64),
            np.zeros(n_nonevents, dtype=np.int64),
        ]
    )

    model = LogisticRegression(penalty=None, fit_intercept=True, solver="lbfgs")
    model.fit(linear_predictor_scores, y_true)
    p_model = model.predict_proba(linear_predictor_scores)[:, 1]
    p_null = np.full_like(p_model, fill_value=float(np.mean(y_true)), dtype=float)
    null_deviance = _bernoulli_deviance(y_true=y_true, probabilities=p_null)
    model_deviance = _bernoulli_deviance(y_true=y_true, probabilities=p_model)
    r2_cs = 1.0 - math.exp(-(null_deviance - model_deviance) / cox_snell_simulation_n)  # [ml-methodology-literature: pmsampsize cstat2rsq reference implementation]
    if not math.isfinite(r2_cs):
        raise ValueError("Cox-Snell simulation produced a non-finite R2_cs value.")
    if r2_cs <= 0.0:
        raise ValueError(f"R2_cs must be positive per Plan 1 §3.3 halt rule; received {r2_cs}.")
    return float(r2_cs)


def _compute_r2_cs_max(anticipated_prevalence: float) -> float:
    """Compute the maximum attainable Cox-Snell R-squared at one prevalence."""
    if not 0.0 < anticipated_prevalence < 1.0:
        raise ValueError(f"anticipated_prevalence must lie in (0, 1); received {anticipated_prevalence}.")
    phi = anticipated_prevalence
    r2_cs_max = 1.0 - math.exp(2.0 * (phi * math.log(phi) + (1.0 - phi) * math.log(1.0 - phi)))  # [ml-methodology-literature: Riley 2020 B4]
    if not math.isfinite(r2_cs_max) or r2_cs_max <= 0.0:
        raise ValueError(f"R2_cs_max must be positive and finite, received {r2_cs_max}.")
    return float(r2_cs_max)


def _compute_n_b1(*, anticipated_prevalence: float, prevalence_margin_of_error: float) -> int:
    """Compute criterion B1: precise prevalence."""
    if not 0.0 < anticipated_prevalence < 1.0:
        raise ValueError(f"anticipated_prevalence must lie in (0, 1); received {anticipated_prevalence}.")
    if prevalence_margin_of_error <= 0.0:
        raise ValueError(f"prevalence_margin_of_error must be positive, received {prevalence_margin_of_error}.")
    n_b1 = math.ceil(
        ((NORMAL_QUANTILE_975 / prevalence_margin_of_error) ** 2)
        * anticipated_prevalence
        * (1.0 - anticipated_prevalence)
    )  # [ml-methodology-literature: Riley 2020 B1]
    return int(n_b1)


def _compute_n_b2(*, candidate_predictor_parameters: int, anticipated_prevalence: float, target_mape: float) -> int:
    """Compute criterion B2: small MAPE under van Smeden's closed form."""
    if candidate_predictor_parameters > 30:
        raise ValueError(
            f"Criterion B2 is only valid for p <= 30; received candidate_predictor_parameters={candidate_predictor_parameters}."
        )
    if candidate_predictor_parameters <= 0:
        raise ValueError(
            f"candidate_predictor_parameters must be positive, received {candidate_predictor_parameters}."
        )
    if not 0.0 < anticipated_prevalence < 1.0:
        raise ValueError(f"anticipated_prevalence must lie in (0, 1); received {anticipated_prevalence}.")
    if target_mape <= 0.0:
        raise ValueError(f"target_mape must be positive, received {target_mape}.")
    n_b2 = math.ceil(
        math.exp(
            (
                -0.508
                + (0.259 * math.log(anticipated_prevalence))
                + (0.504 * math.log(candidate_predictor_parameters))
                - math.log(target_mape)
            )
            / 0.544
        )
    )  # [ml-methodology-literature: van Smeden MAPE formula, valid for p <= 30]
    return int(n_b2)


def _compute_n_b3(*, candidate_predictor_parameters: int, target_shrinkage: float, r2_cs: float) -> int:
    """Compute criterion B3: target shrinkage."""
    if candidate_predictor_parameters <= 0:
        raise ValueError(
            f"candidate_predictor_parameters must be positive, received {candidate_predictor_parameters}."
        )
    if not 0.0 < target_shrinkage < 1.0:
        raise ValueError(f"target_shrinkage must lie in (0, 1); received {target_shrinkage}.")
    if r2_cs >= target_shrinkage:
        raise ValueError(
            f"R2_cs must be strictly smaller than target_shrinkage per Plan 1 §3.3 halt rule; received {r2_cs} vs {target_shrinkage}."
        )
    denominator = (target_shrinkage - 1.0) * math.log(1.0 - (r2_cs / target_shrinkage))
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise ValueError(f"Criterion B3 denominator must be positive and finite, received {denominator}.")
    n_b3 = math.ceil(candidate_predictor_parameters / denominator)  # [ml-methodology-literature: Riley 2020 and pmsampsize default shrinkage recommendation]
    return int(n_b3)


def _compute_s_b4(*, r2_cs: float, target_r2_nagelkerke_optimism: float, r2_cs_max: float) -> float:
    """Derive the B4 shrinkage analogue from Riley's optimism criterion."""
    if target_r2_nagelkerke_optimism <= 0.0:
        raise ValueError(
            f"target_r2_nagelkerke_optimism must be positive, received {target_r2_nagelkerke_optimism}."
        )
    if r2_cs_max <= 0.0:
        raise ValueError(f"R2_cs_max must be positive, received {r2_cs_max}.")
    s_b4 = r2_cs / (r2_cs + (target_r2_nagelkerke_optimism * r2_cs_max))  # [ml-methodology-literature: Riley 2020 B4]
    if not math.isfinite(s_b4) or not 0.0 < s_b4 < 1.0:
        raise ValueError(f"S_b4 must lie in (0, 1), received {s_b4}.")
    return float(s_b4)


def _compute_n_b4(*, candidate_predictor_parameters: int, s_b4: float, r2_cs: float) -> int:
    """Compute criterion B4: small Nagelkerke R-squared optimism."""
    if candidate_predictor_parameters <= 0:
        raise ValueError(
            f"candidate_predictor_parameters must be positive, received {candidate_predictor_parameters}."
        )
    denominator = (s_b4 - 1.0) * math.log(1.0 - (r2_cs / s_b4))
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise ValueError(f"Criterion B4 denominator must be positive and finite, received {denominator}.")
    n_b4 = math.ceil(candidate_predictor_parameters / denominator)  # [ml-methodology-literature: Riley 2020 B4]
    return int(n_b4)


def _build_locked_inputs(config: dict[str, Any]) -> dict[str, float | int]:
    """Assemble the ordered locked-input mapping used in JSON and markdown."""
    sample_size_config = config["sample_size"]
    return {
        "anticipated_auc": float(sample_size_config["anticipated_auc"]),
        "candidate_predictor_parameters": int(sample_size_config["candidate_predictor_parameters"]),
        "anticipated_prevalence": float(sample_size_config["anticipated_prevalence"]),
        "target_shrinkage": float(sample_size_config["target_shrinkage"]),
        "target_mape": float(sample_size_config["target_mape"]),
        "prevalence_margin_of_error": float(sample_size_config["prevalence_margin_of_error"]),
        "target_r2_nagelkerke_optimism": float(sample_size_config["target_r2_nagelkerke_optimism"]),
        "normal_quantile_975": NORMAL_QUANTILE_975,
        "cox_snell_simulation_n": int(sample_size_config["cox_snell_simulation_n"]),
        "mmoe": REGISTERED_MMOE,
    }


def analyze_sample_size(config_path: Path) -> SampleSizeResult:
    """Run the deterministic Plan 1 §3.3 reconstruction without writing outputs.

    Args:
        config_path: Path to `configs/post_hoc_supplementary.yaml`.

    Returns:
        Structured sample-size reconstruction result.

    Raises:
        ValueError: If any Plan 1 §3.3 halt condition is triggered.
        TypeError: If config types are invalid.
    """
    config = load_post_hoc_config(config_path)
    locked_inputs = _build_locked_inputs(config)
    sample_size_config = config["sample_size"]

    LOGGER.info("Running sample-size reconstruction with random_seed=%s.", config["random_seed"])
    LOGGER.info("Locked inputs: %s", locked_inputs)

    r2_cs = _simulate_cox_snell_r_squared(
        anticipated_auc=float(sample_size_config["anticipated_auc"]),
        anticipated_prevalence=float(sample_size_config["anticipated_prevalence"]),
        cox_snell_simulation_n=int(sample_size_config["cox_snell_simulation_n"]),
        random_seed=int(config["random_seed"]),
    )
    r2_cs_max = _compute_r2_cs_max(anticipated_prevalence=float(sample_size_config["anticipated_prevalence"]))
    if r2_cs > r2_cs_max:
        raise ValueError(
            f"R2_cs must not exceed R2_cs_max per Plan 1 §3.3 halt rule; received {r2_cs} vs {r2_cs_max}."
        )

    n_b1 = _compute_n_b1(
        anticipated_prevalence=float(sample_size_config["anticipated_prevalence"]),
        prevalence_margin_of_error=float(sample_size_config["prevalence_margin_of_error"]),
    )
    n_b2 = _compute_n_b2(
        candidate_predictor_parameters=int(sample_size_config["candidate_predictor_parameters"]),
        anticipated_prevalence=float(sample_size_config["anticipated_prevalence"]),
        target_mape=float(sample_size_config["target_mape"]),
    )
    n_b3 = _compute_n_b3(
        candidate_predictor_parameters=int(sample_size_config["candidate_predictor_parameters"]),
        target_shrinkage=float(sample_size_config["target_shrinkage"]),
        r2_cs=r2_cs,
    )
    s_b4 = _compute_s_b4(
        r2_cs=r2_cs,
        target_r2_nagelkerke_optimism=float(sample_size_config["target_r2_nagelkerke_optimism"]),
        r2_cs_max=r2_cs_max,
    )
    n_b4 = _compute_n_b4(
        candidate_predictor_parameters=int(sample_size_config["candidate_predictor_parameters"]),
        s_b4=s_b4,
        r2_cs=r2_cs,
    )

    n_required = max(n_b1, n_b2, n_b3, n_b4)
    required_events = math.ceil(n_required * float(sample_size_config["anticipated_prevalence"]))
    required_epv = float(required_events / int(sample_size_config["candidate_predictor_parameters"]))

    criterion_requirements = {
        "B1": n_b1,
        "B2": n_b2,
        "B3": n_b3,
        "B4": n_b4,
    }
    shortfall_criteria = [
        {
            "criterion": criterion,
            "n_required_for_criterion": required_n,
            "shortfall": required_n - ACTUAL_COHORT_N,
        }
        for criterion, required_n in criterion_requirements.items()
        if required_n > ACTUAL_COHORT_N
    ]
    cohort_meets_all_criteria = len(shortfall_criteria) == 0

    LOGGER.info(
        "Derived R2 metrics: r2_cs=%.10f, r2_cs_max=%.10f.",
        r2_cs,
        r2_cs_max,
    )
    LOGGER.info(
        "Sample-size criteria: n_b1=%s, n_b2=%s, n_b3=%s, n_b4=%s, n_required=%s, required_events=%s, required_epv=%.2f.",
        n_b1,
        n_b2,
        n_b3,
        n_b4,
        n_required,
        required_events,
        required_epv,
    )
    LOGGER.info(
        "Actual cohort comparison: n=%s, events=%s, epv=%.1f, cohort_meets_all_criteria=%s, shortfall_criteria=%s.",
        ACTUAL_COHORT_N,
        ACTUAL_COHORT_EVENTS,
        ACTUAL_COHORT_EPV,
        cohort_meets_all_criteria,
        shortfall_criteria,
    )

    return SampleSizeResult(
        locked_inputs=locked_inputs,
        r2_cs=float(r2_cs),
        r2_cs_max=float(r2_cs_max),
        n_b1=int(n_b1),
        n_b2=int(n_b2),
        n_b3=int(n_b3),
        n_b4=int(n_b4),
        n_required=int(n_required),
        required_events=int(required_events),
        required_epv=float(required_epv),
        cohort_meets_all_criteria=cohort_meets_all_criteria,
        shortfall_criteria=shortfall_criteria,
    )


def _locked_input_markdown_lines(result: SampleSizeResult) -> list[str]:
    """Render the locked input bullets with inline L17 labels."""
    return [
        f"- `anticipated_auc = {result.locked_inputs['anticipated_auc']:.2f}` `[prospectively-registered-protocol-parameter]`",
        "- `candidate_predictor_parameters = 25` `[prospectively-registered-protocol-parameter]`",
        f"- `anticipated_prevalence = {result.locked_inputs['anticipated_prevalence']:.2f}` `[prospectively-registered-protocol-parameter]`",
        f"- `target_shrinkage = {result.locked_inputs['target_shrinkage']:.2f}` `[ml-methodology-literature: Riley 2020 and pmsampsize default shrinkage recommendation]`",
        f"- `target_mape = {result.locked_inputs['target_mape']:.2f}` `[ml-methodology-literature: Riley 2020 B2 target mean absolute prediction error]`",
        f"- `prevalence_margin_of_error = {result.locked_inputs['prevalence_margin_of_error']:.2f}` `[ml-methodology-literature: Riley 2020 B1]`",
        f"- `target_r2_nagelkerke_optimism = {result.locked_inputs['target_r2_nagelkerke_optimism']:.2f}` `[ml-methodology-literature: Riley 2020 B4]`",
        f"- `normal_quantile_975 = {result.locked_inputs['normal_quantile_975']:.2f}` `[ml-methodology-literature: conventional two-sided 95% normal quantile]`",
        f"- `cox_snell_simulation_n = {int(result.locked_inputs['cox_snell_simulation_n'])}` `[ml-methodology-literature: pmsampsize cstat2rsq reference implementation]`",
        f"- `mmoe = {result.locked_inputs['mmoe']:.1f}` `[prospectively-registered-protocol-parameter; non-operative for binary B1-B4]`",
    ]


def _shortfall_subsection(result: SampleSizeResult) -> str:
    """Render the mandatory shortfall disclosure subsection when needed."""
    if result.cohort_meets_all_criteria:
        return ""

    unmet_criteria = ", ".join(entry["criterion"] for entry in result.shortfall_criteria)
    title = (
        f"## Sample-size reconstruction: shortfall against criterion {unmet_criteria}"
        if len(result.shortfall_criteria) == 1
        else f"## Sample-size reconstruction: shortfall against criteria {unmet_criteria}"
    )
    shortfall_lines = [
        f"- `(a)` The actual development cohort (`n = {ACTUAL_COHORT_N}`) does not meet {entry['criterion']}: "
        f"`n_required_for_{entry['criterion'].lower()} = {entry['n_required_for_criterion']}` and "
        f"`shortfall = {entry['shortfall']}`."
        for entry in result.shortfall_criteria
    ]
    shortfall_lines.extend(
        [
            "- `(b)` The locked input parameters producing this shortfall are:",
            *[f"  {line}" for line in _locked_input_markdown_lines(result)],
            "- `(c)` Calibration-focused criteria, especially the van Smeden B2 MAPE requirement, are widely unmet in published clinical prediction-model studies, including studies appearing in high-impact journals, because the Riley 2020 framework post-dates much of the existing literature and uptake is uneven. This disclosure follows the methodological context described by van Smeden et al. 2019 (*Statistical Methods in Medical Research*) and Riley et al. 2020 (*BMJ* m441).",
            f"- `(d)` The frozen-stage empirical calibration evidence remains the actual basis for judging calibration adequacy in the developed model: post-Platt Brier `{FROZEN_POST_PLATT_BRIER:.4f}`, calibration slope `{FROZEN_CALIBRATION_SLOPE:.4f}`, calibration intercept `{FROZEN_CALIBRATION_INTERCEPT:.4f}` (source: `docs/modeling/validated_modeling_run_summary.md`).",
            f"- `(e)` {SHORTFALL_LIMITATION_TEXT}",
        ]
    )
    return "\n".join([title, "", *shortfall_lines, ""])


def _build_markdown_report(
    *,
    result: SampleSizeResult,
    config: dict[str, Any],
    generated_at_utc: str,
    git_commit_hash: str,
    markdown_path: Path,
    json_path: Path,
) -> str:
    """Build the Session 1 sample-size reconstruction markdown report."""
    locked_input_lines = "\n".join(_locked_input_markdown_lines(result))
    shortfall_section = _shortfall_subsection(result)
    shortfall_summary = (
        "All four criteria are met by the actual development cohort."
        if result.cohort_meets_all_criteria
        else "The actual development cohort meets B1, B3, and B4 but falls short of B2."
    )

    return (
        "# Post-Hoc Sample-Size Reconstruction\n\n"
        f"> {POST_HOC_DECLARATION}\n\n"
        f"- `generated_at_utc = {generated_at_utc}`\n"
        f"- `git_commit_hash = {git_commit_hash}`\n"
        f"- `temporal_anchor_tag = {config['temporal_anchor_tag']}`\n"
        f"- `output_markdown = {_format_path(markdown_path)}`\n"
        f"- `output_json = {_format_path(json_path)}`\n\n"
        "## Locked Input Parameters\n\n"
        f"{locked_input_lines}\n\n"
        "## Reconstruction Results\n\n"
        f"- `R2_cs = {result.r2_cs:.10f}`\n"
        f"- `R2_cs_max = {result.r2_cs_max:.10f}`\n"
        f"- `n_b1 = {result.n_b1}`\n"
        f"- `n_b2 = {result.n_b2}`\n"
        f"- `n_b3 = {result.n_b3}`\n"
        f"- `n_b4 = {result.n_b4}`\n"
        f"- `n_required = {result.n_required}`\n"
        f"- `required_events = {result.required_events}`\n"
        f"- `required_epv = {result.required_epv:.2f}`\n\n"
        "## Actual Cohort Comparison\n\n"
        f"- Actual development cohort: `n = {ACTUAL_COHORT_N}`, `events = {ACTUAL_COHORT_EVENTS}`, `EPV = {ACTUAL_COHORT_EPV:.1f}`\n"
        f"- `cohort_meets_all_criteria = {result.cohort_meets_all_criteria}`\n"
        f"- Summary: {shortfall_summary}\n\n"
        f"{shortfall_section}"
        "## MMOE Disclosure\n\n"
        f"{MMOE_DISCLOSURE_TEXT}\n"
    )


def generate_sample_size_outputs(
    config_path: Path,
    *,
    generated_at_utc: str | None = None,
    git_commit_hash: str | None = None,
) -> SampleSizeResult:
    """Run the analysis and write the Session 1 markdown and JSON outputs.

    Args:
        config_path: Path to `configs/post_hoc_supplementary.yaml`.
        generated_at_utc: Optional externally supplied script-entry timestamp.
        git_commit_hash: Optional externally supplied `HEAD` hash.

    Returns:
        Structured sample-size reconstruction result.
    """
    config = load_post_hoc_config(config_path)
    result = analyze_sample_size(config_path)
    timestamp_value = generated_at_utc or _timestamp_utc()
    git_hash_value = git_commit_hash or _get_git_commit_hash()

    outputs_analysis_dir = _resolve_path(config["paths"]["outputs_analysis_dir"])
    markdown_path = outputs_analysis_dir / "posthoc_sample_size_reconstruction.md"
    json_path = outputs_analysis_dir / "posthoc_sample_size_reconstruction.json"

    payload = result.to_json_payload(
        provenance_schema_version=int(config["provenance_schema_version"]),
        stage=str(config["stage"]),
        best_model=str(config["best_model"]),
        random_seed=int(config["random_seed"]),
        temporal_anchor_tag=str(config["temporal_anchor_tag"]),
        generated_at_utc=timestamp_value,
        git_commit_hash=git_hash_value,
    )
    markdown = _build_markdown_report(
        result=result,
        config=config,
        generated_at_utc=timestamp_value,
        git_commit_hash=git_hash_value,
        markdown_path=markdown_path,
        json_path=json_path,
    )

    _write_json(json_path, payload)
    _write_text(markdown_path, markdown)
    LOGGER.info(
        "Wrote sample-size reconstruction outputs to %s and %s.",
        _format_path(markdown_path),
        _format_path(json_path),
    )
    return result
