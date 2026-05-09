"""Label-blind cohort aggregation primitives for preprocessing discovery."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.stats import gaussian_kde
from skimage.filters import threshold_otsu


LOGGER = logging.getLogger(__name__)
_STD_FLOOR = 1e-8


def _validate_finite_1d_array(values: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    """Validate and coerce a one-dimensional finite numeric array."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _validate_aligned_arrays(
    baseline_means: NDArray[np.float64],
    t_samples: NDArray[np.float64],
    h_samples: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Validate three aligned one-dimensional finite arrays."""
    baseline_array = _validate_finite_1d_array(baseline_means, "baseline_means")
    t_array = _validate_finite_1d_array(t_samples, "t_samples")
    h_array = _validate_finite_1d_array(h_samples, "h_samples")

    if not (baseline_array.size == t_array.size == h_array.size):
        raise ValueError("baseline_means, t_samples, and h_samples must have the same length.")

    return baseline_array, t_array, h_array


def _safe_standard_deviation(values: NDArray[np.float64], name: str) -> float:
    """Return a numerically stable standard deviation for correlation calculations."""
    std_value = float(np.std(values, ddof=0))
    if std_value < _STD_FLOOR:
        LOGGER.warning("%s standard deviation below %.1e; clamping to 1.0.", name, _STD_FLOOR)
        return 1.0
    return std_value


def _compute_r2(observed: NDArray[np.float64], predicted: NDArray[np.float64]) -> float:
    """Compute coefficient of determination with a constant-response guard."""
    residual_sum_of_squares = float(np.sum((observed - predicted) ** 2))
    total_sum_of_squares = float(np.sum((observed - np.mean(observed)) ** 2))
    if total_sum_of_squares < _STD_FLOOR:
        return 0.0
    return 1.0 - (residual_sum_of_squares / total_sum_of_squares)


def _compute_adjusted_r2(r2: float, n_samples: int, n_predictors: int) -> float:
    """Compute adjusted R-squared for a fitted linear model."""
    if n_samples <= n_predictors + 1:
        raise ValueError(
            "Adjusted R-squared requires more samples than predictors plus intercept degrees of freedom."
        )
    return 1.0 - (1.0 - r2) * (n_samples - 1) / (n_samples - n_predictors - 1)


def _fit_univariate_regression(
    response: NDArray[np.float64], predictor: NDArray[np.float64]
) -> tuple[float, float, float]:
    """Fit an intercept-plus-slope univariate linear regression."""
    design_matrix = np.column_stack([np.ones(response.size, dtype=float), predictor])
    coefficients, _, _, _ = np.linalg.lstsq(design_matrix, response, rcond=None)
    fitted = design_matrix @ coefficients
    r2 = _compute_r2(response, fitted)
    adjusted_r2 = _compute_adjusted_r2(r2=r2, n_samples=response.size, n_predictors=1)
    intercept = float(coefficients[0])
    slope = float(coefficients[1])
    return intercept, slope, adjusted_r2


def detect_bimodality_kde(values: NDArray[np.float64]) -> bool:
    """Detect whether a one-dimensional distribution is bimodal under the prespecified KDE rule.

    The algorithm uses `scipy.stats.gaussian_kde` with Scott's-rule bandwidth, evaluates the density on the
    sorted unique observed values, counts strict interior local maxima, and returns `True` when at least two
    peaks are present.

    Args:
        values: One-dimensional finite cohort-level summary values.

    Returns:
        `True` if the KDE evaluation yields at least two strict interior peaks; otherwise `False`. This is a
        `[runtime-determined-by-prespecified-algorithm]` result.

    Raises:
        ValueError: If `values` is empty, not one-dimensional, or contains non-finite entries.
    """
    array = _validate_finite_1d_array(values, "values")
    unique_values = np.unique(array)
    if unique_values.size < 3:
        return False

    try:
        density = gaussian_kde(array)(unique_values)
    except (np.linalg.LinAlgError, ValueError):
        return False

    if density.size < 3:
        return False

    local_maxima = (density[1:-1] > density[:-2]) & (density[1:-1] > density[2:])
    peak_count = int(np.count_nonzero(local_maxima))
    return peak_count >= 2


def compute_otsu_threshold(values: NDArray[np.float64]) -> float:
    """Compute an Otsu threshold for a finite one-dimensional sample.

    This function is a thin wrapper around `skimage.filters.threshold_otsu`, which is the canonical
    scikit-image implementation of Otsu's between-class variance thresholding method.

    Args:
        values: One-dimensional finite values to threshold.

    Returns:
        The Otsu threshold as a float. This is a `[runtime-determined-by-prespecified-algorithm]` value.

    Raises:
        ValueError: If `values` is empty, not one-dimensional, contains non-finite entries, or contains fewer
            than two unique values.
    """
    array = _validate_finite_1d_array(values, "values")
    if np.unique(array).size < 2:
        raise ValueError("values must contain at least two unique values for Otsu thresholding.")

    try:
        return float(threshold_otsu(array))
    except ValueError as exc:
        raise ValueError(f"Failed to compute Otsu threshold: {exc}") from exc


def compute_tukey_upper_inner_fence(values: NDArray[np.float64]) -> float:
    """Compute the Tukey upper inner fence `Q3 + 1.5 * IQR`.

    Args:
        values: One-dimensional finite values used to compute the fence.

    Returns:
        The upper inner fence as a float. This is a `[runtime-determined-by-prespecified-algorithm]` value.

    Raises:
        ValueError: If `values` is empty, not one-dimensional, or contains non-finite entries.
    """
    array = _validate_finite_1d_array(values, "values")
    q1 = float(np.percentile(array, 25))
    q3 = float(np.percentile(array, 75))
    iqr = q3 - q1
    return float(q3 + 1.5 * iqr)


def compute_tukey_lower_inner_fence(values: NDArray[np.float64], clip_at_zero: bool = False) -> float:
    """Compute the Tukey lower inner fence `Q1 - 1.5 * IQR`.

    Args:
        values: One-dimensional finite values used to compute the fence.
        clip_at_zero: Whether to clip negative fence values to `0.0`.

    Returns:
        The lower inner fence as a float. This is a `[runtime-determined-by-prespecified-algorithm]` value.

    Raises:
        ValueError: If `values` is empty, not one-dimensional, or contains non-finite entries.
    """
    array = _validate_finite_1d_array(values, "values")
    q1 = float(np.percentile(array, 25))
    q3 = float(np.percentile(array, 75))
    iqr = q3 - q1
    fence = float(q1 - 1.5 * iqr)
    if clip_at_zero and fence < 0.0:
        return 0.0
    return fence


def compute_kofn_fallback_median(triggered_t_baseline_end: NDArray[np.float64]) -> float:
    """Compute the K-of-N baseline fallback median over triggered samples.

    Args:
        triggered_t_baseline_end: One-dimensional finite array of triggered-sample `t_baseline_end` values.

    Returns:
        The median triggered `t_baseline_end` value as a float. This is a
        `[runtime-determined-by-prespecified-algorithm]` value.

    Raises:
        ValueError: If `triggered_t_baseline_end` is empty, not one-dimensional, or contains non-finite
            entries.
    """
    array = _validate_finite_1d_array(triggered_t_baseline_end, "triggered_t_baseline_end")
    return float(np.median(array))


def compute_pearson_correlation(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """Compute a numerically stable Pearson correlation coefficient.

    Args:
        x: One-dimensional finite predictor values.
        y: One-dimensional finite predictor values aligned to `x`.

    Returns:
        The Pearson correlation coefficient as a float. This is a
        `[runtime-determined-by-prespecified-algorithm]` value.

    Raises:
        ValueError: If the arrays are empty, not one-dimensional, contain non-finite entries, have unequal
            lengths, or contain fewer than two samples.
    """
    x_array = _validate_finite_1d_array(x, "x")
    y_array = _validate_finite_1d_array(y, "y")

    if x_array.size != y_array.size:
        raise ValueError("x and y must have the same length.")
    if x_array.size < 2:
        raise ValueError("x and y must contain at least two samples.")

    x_centered = x_array - np.mean(x_array)
    y_centered = y_array - np.mean(y_array)
    x_std = _safe_standard_deviation(x_array, "x")
    y_std = _safe_standard_deviation(y_array, "y")
    covariance = float(np.mean(x_centered * y_centered))
    correlation = covariance / (x_std * y_std)
    return float(np.clip(correlation, -1.0, 1.0))


def fit_environmental_regression(
    baseline_means: NDArray[np.float64],
    t_samples: NDArray[np.float64],
    h_samples: NDArray[np.float64],
) -> dict[str, float]:
    """Fit the prespecified multivariable environmental regression.

    Args:
        baseline_means: One-dimensional finite baseline means for one gas channel.
        t_samples: One-dimensional finite temperature summaries aligned to `baseline_means`.
        h_samples: One-dimensional finite humidity summaries aligned to `baseline_means`.

    Returns:
        A dictionary with keys `alpha`, `beta_T`, `beta_H`, and `adj_r2`. Each numeric value is
        `[runtime-determined-by-prespecified-algorithm]`.

    Raises:
        ValueError: If the inputs are empty, not one-dimensional, contain non-finite entries, have unequal
            lengths, or do not provide enough samples for adjusted R-squared with two predictors.
    """
    baseline_array, t_array, h_array = _validate_aligned_arrays(baseline_means, t_samples, h_samples)
    if baseline_array.size <= 3:
        raise ValueError("Environmental regression requires at least four aligned samples.")

    design_matrix = np.column_stack([np.ones(baseline_array.size, dtype=float), t_array, h_array])
    coefficients, _, _, _ = np.linalg.lstsq(design_matrix, baseline_array, rcond=None)
    fitted = design_matrix @ coefficients
    r2 = _compute_r2(baseline_array, fitted)
    adj_r2 = _compute_adjusted_r2(r2=r2, n_samples=baseline_array.size, n_predictors=2)

    return {
        "alpha": float(coefficients[0]),
        "beta_T": float(coefficients[1]),
        "beta_H": float(coefficients[2]),
        "adj_r2": float(adj_r2),
    }


def select_univariate_fallback(
    baseline_means: NDArray[np.float64],
    t_samples: NDArray[np.float64],
    h_samples: NDArray[np.float64],
) -> dict[str, float]:
    """Select the higher-adjusted-R-squared univariate environmental fallback model.

    Args:
        baseline_means: One-dimensional finite baseline means for one gas channel.
        t_samples: One-dimensional finite temperature summaries aligned to `baseline_means`.
        h_samples: One-dimensional finite humidity summaries aligned to `baseline_means`.

    Returns:
        A dictionary with keys `alpha`, `beta_T`, `beta_H`, and `adj_r2`. The dropped predictor coefficient is
        set to `0.0`. Each numeric value is `[runtime-determined-by-prespecified-algorithm]`.

    Raises:
        ValueError: If the inputs are empty, not one-dimensional, contain non-finite entries, have unequal
            lengths, or do not provide enough samples for adjusted R-squared with one predictor.
    """
    baseline_array, t_array, h_array = _validate_aligned_arrays(baseline_means, t_samples, h_samples)
    if baseline_array.size <= 2:
        raise ValueError("Univariate fallback requires at least three aligned samples.")

    temp_alpha, temp_beta, temp_adj_r2 = _fit_univariate_regression(baseline_array, t_array)
    humid_alpha, humid_beta, humid_adj_r2 = _fit_univariate_regression(baseline_array, h_array)

    if temp_adj_r2 >= humid_adj_r2:
        return {
            "alpha": float(temp_alpha),
            "beta_T": float(temp_beta),
            "beta_H": 0.0,
            "adj_r2": float(temp_adj_r2),
        }

    return {
        "alpha": float(humid_alpha),
        "beta_T": 0.0,
        "beta_H": float(humid_beta),
        "adj_r2": float(humid_adj_r2),
    }


def compute_clip_length(n_available: NDArray[np.float64]) -> int:
    """Compute the prespecified clip length from the low-tail response-length distribution.

    Args:
        n_available: One-dimensional finite non-negative response lengths.

    Returns:
        The clip length `L` as an integer `floor(percentile(n_available, 5))`. This is a
        `[runtime-determined-by-prespecified-algorithm]` value.

    Raises:
        ValueError: If `n_available` is empty, not one-dimensional, contains non-finite entries, or contains
            negative values.
    """
    array = _validate_finite_1d_array(n_available, "n_available")
    if np.any(array < 0):
        raise ValueError("n_available must contain only non-negative values.")
    return int(np.floor(np.percentile(array, 5)))


def compute_padding_deficit_threshold(relative_deficits: NDArray[np.float64]) -> float | None:
    """Compute the Tukey upper inner fence for relative padding deficits.

    Args:
        relative_deficits: One-dimensional finite non-negative relative deficits.

    Returns:
        The Tukey upper inner fence as a float, or `None` when the input is empty. Any numeric return is a
        `[runtime-determined-by-prespecified-algorithm]` value.

    Raises:
        ValueError: If `relative_deficits` is not one-dimensional, contains non-finite entries, or contains
            negative values.
    """
    array = np.asarray(relative_deficits, dtype=float)
    if array.ndim != 1:
        raise ValueError("relative_deficits must be a one-dimensional array.")
    if array.size == 0:
        return None
    if not np.isfinite(array).all():
        raise ValueError("relative_deficits must contain only finite values.")
    if np.any(array < 0):
        raise ValueError("relative_deficits must contain only non-negative values.")
    return compute_tukey_upper_inner_fence(array)
