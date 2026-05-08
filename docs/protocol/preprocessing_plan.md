# Preprocessing Protocol Snapshot

This frozen protocol snapshot is derived from `.cursor/plans/preprocessing_plan_v1_539bc1eb.plan.md` at commit `TBD-at-commit-time`. It is prespecified before any preprocessing implementation commit and is frozen on the annotated tag `preprocessing-prespecified`.

This document is immutable after tagging. Deviations from the locked methodology are recorded only as errata in the operational plan under `.cursor/plans/`; this snapshot is never edited after lock.

## L17 Provenance Legend

- `[domain-physical-constant]`: derived from acquisition SOP, hardware structure, source-file structure, or pre-authored repository policy.
- `[signal-processing-literature]`: drawn from established signal-processing, statistical quality-control, electronic-nose, gas-sensor, or statistical-methodology literature cited in this plan.
- `[runtime-determined-by-prespecified-algorithm]`: computed label-blind by an algorithm specified before cohort-wide computation.
- `[independent-pilot-or-engineering-bench-test]`: reserved for values from pre-cohort bench measurements documented outside this cohort. Currently unused.
- `[sensor-manufacturer-datasheet]`: reserved for values drawn from a manufacturer datasheet. Currently unused.
- `[runtime-determined; not preregistered]`: used only if a value cannot be assigned to one of the categories above.

## 1. Cohort And Signal Facts

### 1.1 Cohort

- Cohort size: `1040` prospectively collected exhaled-breath samples `[domain-physical-constant: source metadata/file count supplied before plan authoring]`.
- Disease distribution: HC=`163`, BGL=`378`, LGIN=`53`, HGIN=`76`, EGC=`137`, AGC=`233` `[domain-physical-constant: source metadata aggregate supplied before plan authoring]`.
- Binary label distribution: label `0` = HC+BGL+LGIN = `594`; label `1` = HGIN+EGC+AGC = `446` `[domain-physical-constant: source metadata aggregate supplied before plan authoring]`.
- Collection period: `[PRIVATE_LINKAGE_REDACTED]` to `[PRIVATE_LINKAGE_REDACTED]` `[domain-physical-constant: metadata aggregate supplied before plan authoring]`.
- Metadata contains sample identifiers, filename mapping, disease category, binary label, demographics, and tumor-marker fields. Tumor markers are documented but are neither imputed nor used during preprocessing.
- Tumor-marker missingness is approximately `31%` `[domain-physical-constant: metadata aggregate supplied before plan authoring]`; CA19-9 includes a malformed value `14,07` `[domain-physical-constant: metadata fact supplied before plan authoring]`.

### 1.2 Acquisition SOP

- Acquisition device: one MOS MEMS gas sensor array `[domain-physical-constant: user-confirmed SOP]`.
- Device warm-up: approximately `15` minutes `[domain-physical-constant: user-confirmed SOP]`.
- Subject blow duration: approximately `45-60` seconds `[domain-physical-constant: user-confirmed SOP]`.
- Passive interval: approximately `60-75` seconds `[domain-physical-constant: user-confirmed SOP]`.
- Total per-sample acquisition: approximately `120` seconds `[domain-physical-constant: user-confirmed SOP]`.
- Recording begins before the participant blows, so each trace starts with a short baseline segment.
- Temperature and humidity vary between samples and are handled only by the prespecified label-blind environmental-compensation procedure.

### 1.3 Raw Files And Sampling

- Uniform sampling rate is assumed `[domain-physical-constant: acquisition SOP]`.
- `1038` of `1040` files have exactly `1000` rows `[domain-physical-constant: structural file-count fact supplied before plan authoring]`.
- `2` files have fewer than `1000` rows `[domain-physical-constant: structural file-count fact supplied before plan authoring]` and are excluded by QC-2 if `n_rows < 1000`.
- The `1000`-row figure is a hardware/SOP constant `[domain-physical-constant: user-confirmed acquisition structure]`.
- Implied sampling rate is approximately `8.33` Hz `[domain-physical-constant: 1000 rows divided by approximately 120 seconds from SOP]`.
- Raw breath CSV files use UTF-8 with BOM and must be read with `encoding='utf-8-sig'` `[domain-physical-constant: file-format fact supplied before plan authoring]`.

### 1.4 Channel Layout And Traceability

- Raw breath CSV files contain `14` columns S0-S13 `[domain-physical-constant: hardware channel layout]`.
- S0 is a diagnostic ADC-range indicator. Its maximum observed value is `4095` `[domain-physical-constant: structural hardware fact supplied before plan authoring]`; a `12`-bit interpretation remains `[sensor-manufacturer-datasheet: not used; bit depth not treated as a confirmed parameter]`. S0 is excluded from downstream computation.
- S1-S11 are the `11` gas-sensing channels `[domain-physical-constant: hardware channel layout]`.
- S12 is the ambient temperature channel `[domain-physical-constant: hardware channel layout]`.
- S13 is the ambient humidity channel `[domain-physical-constant: hardware channel layout]`.
- Filenames follow `{YYYY-MM-DD}-{sample_id}-{disease_category}-{label}.csv` `[domain-physical-constant: source naming convention supplied before plan authoring]`.
- The filename-to-`sample_id` mapping is taken from `metadata.csv::filename`; `sample_id` is never parsed from filename strings.

## 2. Pipeline Overview

Stage 1 preprocessing is a single full-cohort, label-blind pass that converts raw breath sensor CSV files and metadata into two model-ready but still unnormalized outputs: a sign-preserving time-series tensor for deep learning and a `99`-feature table `[signal-processing-literature: 9 response-curve features × 11 gas channels; Yan et al. feature taxonomy context]` for traditional machine learning.

The prespecified sequence is:

1. Load metadata and raw CSV files.
2. Separate channels: drop S0; retain gas channels S1-S11; retain S12 and S13 as environmental covariates.
3. Run label-blind Phase 1 discovery to compute thresholds and calibration values only.
4. Detect onset with K-of-N voting across all gas channels and assign a fallback onset when needed.
5. Apply six prespecified QC checks.
6. Smooth only the gas channels.
7. Fit and apply environmental compensation using baseline summaries from S12 and S13.
8. Apply fractional baseline correction while preserving response sign.
9. Trim each response to a common length `L` `[runtime-determined-by-prespecified-algorithm]`, with prespecified padding or exclusion rules.
10. Write Pipeline A (§4), Pipeline B (§5), and audit artifacts (§6).

Disease labels are kept only for traceability and downstream alignment. They are never used to choose thresholds, smoothing settings, environmental coefficients, clip length, or feature definitions.

## 3. Prespecified Preprocessing Methodology

### 3.1 Raw Loading And Channel Separation

For each metadata row, the referenced CSV file is loaded using UTF-8 with BOM. The raw table must contain S0-S13, that is `14` columns `[domain-physical-constant: hardware channel layout]`. Downstream computation retains the `11` gas channels `[domain-physical-constant: hardware channel layout]` and the two environmental channels; S0 remains diagnostic only. Immediately after loading, the implementation logs shape, dtype, missingness, and `sample_id` traceability at INFO level.

### 3.2 Phase-1 Label-Blind Discovery

Phase 1 computes thresholds and calibration values only. It does not make final QC exclusion decisions and does not write Pipeline A or Pipeline B outputs. The following quantities are determined label-blind and stored as `[runtime-determined-by-prespecified-algorithm]` values in `data/processed/discovery_report.json` and `configs/preprocessing.yaml`:

- QC-1 threshold or disabled flag
- QC-3 baseline CV threshold
- QC-4 response-amplitude threshold
- QC-5 saturation threshold
- QC-6 spike-count threshold
- environmental multicollinearity flag
- per-channel environmental coefficients and adjusted `R^2`
- K-of-N fallback median
- clip length `L`
- padding deficit threshold

### 3.3 Onset Detection: K-Of-N Voting, No Sentinel

Baseline statistics are computed independently for each gas channel `i in {1, ..., 11}` where `11` is `[domain-physical-constant: gas-channel count]`.

- `t_baseline_window_init = 5` samples `[domain-physical-constant: SOP-derived early baseline window; acquisition starts before subject blow]`.
- `mu_i = mean(x_i[0:t_baseline_window_init])`.
- `sigma_i = std(x_i[0:t_baseline_window_init])`.
- If `sigma_i < 1e-8`, set `sigma_i = 1.0` `[signal-processing-literature: numerical stability convention for degenerate variance; implementation guard, not biological threshold]` and log WARNING.

A per-channel onset event occurs when:

- `abs(x_i[t] - mu_i) > sigma_multiplier * sigma_i` for `consecutive_exceed` consecutive samples.
- `sigma_multiplier = 3.0` `[signal-processing-literature: Shewhart 3-sigma control limit; NIST SPC handbook]`.
- `consecutive_exceed = 3` `[signal-processing-literature: short sustained run confirmation adapted from control-chart run-rule logic; Western Electric 1956; Montgomery SPC]`.
- Search window is `[t_baseline_end_min, t_baseline_end_max]`.
- `t_baseline_end_min = 5` samples `[domain-physical-constant: same SOP-derived minimum early baseline bound as t_baseline_window_init]`.
- `t_baseline_end_max = 50` samples `[domain-physical-constant: approximately 6 seconds at 8.33 Hz, from prompt user-confirmed SOP requiring prompt blow onset]`.

Sample-level onset uses weak-consensus voting:

- `K = ceil(11 / 3) = 4` channels `[signal-processing-literature: weak-consensus multisensor voting rule; 11 is domain channel count, denominator 3 encodes more-than-one-third quorum without sentinel-channel selection]`.
- `t_baseline_end` is the smallest `t` at which at least `K` channels have triggered.
- If fewer than `K` channels trigger in the search window, record `fallback_used=True`.

Fallback is deterministic and cohort-wide:

- Pass 1 computes onset for samples with valid K-of-N triggers.
- `fallback_median = median(t_baseline_end)` among triggered samples `[runtime-determined-by-prespecified-algorithm]`.
- Pass 2 assigns `t_baseline_end = fallback_median` to samples with `fallback_used=True`.

By definition, `t_baseline_end` is the first post-baseline sample. Therefore `[0, t_baseline_end)` is the baseline interval and `[t_baseline_end, n_rows)` is the response interval.

### 3.4 Quality Control

There are `6` QC checks `[signal-processing-literature: fixed taxonomy prespecified for this study; not data-derived]`. QC-1 is evaluated first. If QC-1 fails, QC-2 through QC-6 diagnostics are set to NaN. If QC-1 passes, QC-2 through QC-6 are all evaluated and failures accumulate. A sample is excluded from both pipelines if any QC check fails or if response trimming fails.

#### QC-1 Environmental Sensor Fault

- `ENV_SENSOR_BASELINE_POINTS = t_baseline_window_init = 5` samples `[domain-physical-constant: same SOP-derived early baseline window as §3.3 onset detection; consistent definition of "device-just-started, no breath yet" interval used across both QC-1 environmental fault detection and gas-channel onset detection]`.
- `s12_baseline_mean = mean(v_temp[0:ENV_SENSOR_BASELINE_POINTS])`.
- `ENV_SENSOR_FAULT_S12_THRESHOLD = [runtime-determined-by-prespecified-algorithm]` via Otsu's method.
- Otsu's method is justified as `[signal-processing-literature: Otsu 1979]`.
- KDE bandwidth follows Scott's rule `[signal-processing-literature: Scott 1992 default bandwidth selection rule]`.
- The KDE grid is the sorted unique observed `s12_baseline_mean` values `[runtime-determined-by-prespecified-algorithm: grid is determined entirely by the cohort's observed environmental-baseline statistic values, not by an unverified fixed resolution]`.
- QC-1 is enabled if the KDE peak count is at least `2` `[signal-processing-literature: bimodality requires at least two modes by definition]`.
- QC-1 is disabled if the KDE peak count is `1` `[signal-processing-literature: unimodality has exactly one mode by definition]` or `0` `[signal-processing-literature: no interior mode detected under the prespecified KDE evaluation]`.

#### QC-2 Acquisition Truncated

- `MIN_ROWS_QC2 = 1000` rows `[domain-physical-constant: hardware/SOP acquisition length]`.
- The sample fails if `n_rows < MIN_ROWS_QC2`.

#### QC-3 Unstable Baseline

- `cv_i = sigma_i / abs(mu_i)`.
- If `abs(mu_i) < 1e-8`, use denominator `1.0` for diagnostics `[signal-processing-literature: numerical stability convention for degenerate denominator]`.
- `BASELINE_CV_THRESHOLD = [runtime-determined-by-prespecified-algorithm]` using Tukey upper inner fence `Q3 + 1.5 × IQR`.
- `1.5 × IQR` is `[signal-processing-literature: Tukey 1977 inner-fence mild-outlier rule]`.
- `N_UNSTABLE_THRESHOLD = floor(11 / 2) + 1 = 6` channels `[signal-processing-literature: majority-vote multisensor reliability rule; 11 is domain channel count]`.
- The sample fails if `n_unstable_channels >= N_UNSTABLE_THRESHOLD`.

#### QC-4 Insufficient Response

- `delta_max = max(x_i[t_baseline_end:]) - mu_i`.
- `delta_min = min(x_i[t_baseline_end:]) - mu_i`.
- `delta_i = delta_max` if `abs(delta_max) >= abs(delta_min)`, otherwise `delta_i = delta_min`.
- `r_i = delta_i / abs(mu_i)` with denominator guard `1e-8` `[signal-processing-literature: numerical stability convention]`.
- `RESPONSE_AMPLITUDE_THRESHOLD = [runtime-determined-by-prespecified-algorithm]`.
- If the pooled `abs(r_i)` distribution is bimodal, thresholding uses Otsu `[signal-processing-literature: Otsu 1979]`.
- Otherwise it uses Tukey lower inner fence `Q1 - 1.5 × IQR`, clipped at `0.0` `[signal-processing-literature: Tukey 1977; nonnegative magnitude boundary]`.
- `N_RESPOND_THRESHOLD = ceil(11 / 4) = 3` channels `[signal-processing-literature: minority quorum for multisensor informativeness; 11 is domain channel count and denominator 4 encodes at-least-one-quartile response]`.
- The sample fails if `n_responding_channels < N_RESPOND_THRESHOLD`.

#### QC-5 Sensor Saturation

- A point is saturated if it is within `1` ADC unit of the channel-specific trace maximum `[domain-physical-constant: minimum integer ADC step in recorded data representation]`.
- `SATURATION_THRESHOLD = [runtime-determined-by-prespecified-algorithm]` using Tukey upper inner fence `Q3 + 1.5 × IQR`.
- If the pooled saturation-fraction distribution has no variability and the common value is greater than `0.0` `[signal-processing-literature: zero saturation is the no-artifact null]`, execution halts for user review.
- The structural hardware ceiling `4095` remains `[domain-physical-constant: structural hardware fact supplied before plan authoring]`; a `12`-bit interpretation remains `[sensor-manufacturer-datasheet: not used; bit depth not treated as a confirmed parameter]`.
- The saturation threshold is a fraction in `[0, 1]` `[signal-processing-literature: saturation fraction bounds]`.

#### QC-6 Excessive Transient Spikes

- First differences are `diff_i[t] = x_i[t] - x_i[t-1]`.
- Candidate spikes are detected within each channel and sample by Tukey thresholding of `abs(diff_i)` at `Q3 + 1.5 × IQR` `[signal-processing-literature: Tukey 1977 robust outlier fence]`.
- `SPIKE_COUNT_THRESHOLD = [runtime-determined-by-prespecified-algorithm]` using Tukey upper inner fence `Q3 + 1.5 × IQR`.

### 3.5 Savitzky-Golay Smoothing

Smoothing is applied only to the `11` gas channels `[domain-physical-constant: hardware channel layout]`. S12 and S13 are not smoothed.

- `window_length = 11` samples `[signal-processing-literature: odd-window Savitzky-Golay local polynomial smoothing; at approximately 8.33 Hz this is approximately 1.32 seconds, much shorter than MOS response/recovery dynamics on the order of 10-30 seconds; Savitzky and Golay 1964; Press et al.; MOS denoising practice]`.
- `polyorder = 3` `[signal-processing-literature: cubic local polynomial commonly used to preserve peak/curve shape while smoothing; Savitzky and Golay 1964; Press et al.]`.
- Implementation uses `scipy.signal.savgol_filter(axis=0, mode='interp')`.

The smoothing step is defensively incompatible with traces shorter than `window_length`. Under `MIN_ROWS_QC2 = 1000` `[domain-physical-constant: hardware/SOP acquisition length]` and `window_length = 11` `[signal-processing-literature: odd-window Savitzky-Golay local polynomial smoothing]`, this halt path is not expected to occur.

### 3.6 Environmental Compensation

Environmental compensation is fit only on the QC-1/QC-2 passing cohort and never uses labels.

For each gas channel:

- `T_sample = mean(v_temp[0:t_baseline_end])`.
- `H_sample = mean(v_humid[0:t_baseline_end])`.
- `baseline_mean_i = mean(x_smoothed_i[0:t_baseline_end])`.
- Fit `baseline_mean_i = alpha_i + beta_T_i * T_sample + beta_H_i * H_sample`.

Reference conditions are cohort medians:

- `T_ref = median(T_sample)` `[runtime-determined-by-prespecified-algorithm]`.
- `H_ref = median(H_sample)` `[runtime-determined-by-prespecified-algorithm]`.

Multicollinearity and applicability rules are:

- Use Pearson correlation between `T_sample` and `H_sample`.
- If `abs(r) > 0.80` `[signal-processing-literature: common severe pairwise-correlation screening heuristic, applied cautiously because VIF rules are context-dependent; O'Brien 2007]`, use univariate fallback.
- In the fallback, the dropped coefficient is set to `0.0` `[signal-processing-literature: zero coefficient encodes omitted predictor]`.
- Compensation applies only if adjusted `R^2_i > R2_THRESHOLD`.
- `R2_THRESHOLD = 0.0` adjusted `R^2` `[signal-processing-literature: adjusted R² greater than zero indicates improvement over intercept-only model; replaces unsupported 0.10 threshold]`.

The corrected trace is:

- `x_compensated_i(t) = x_smoothed_i(t) - beta_T_i * (T_sample - T_ref) - beta_H_i * (H_sample - H_ref)`.

The YAML snapshot records `alpha_i`, `beta_T_i`, `beta_H_i`, adjusted `R^2_i`, `applies_compensation_i`, `T_ref`, `H_ref`, and `multicollinearity_fallback_used`.

### 3.7 Fractional Baseline Correction And Trimming

For each compensated gas channel:

- `S0_i = mean(x_compensated_i[0:t_baseline_end])`.
- `x_corrected_i(t) = (x_compensated_i(t) - S0_i) / S0_i` `[signal-processing-literature: fractional baseline manipulation removes additive baseline offset and expresses per-baseline change while preserving sign; Gutierrez-Osuna 2002; Yan et al. 2015]`.
- Log WARNING if `abs(S0_i) < 10` ADC units `[domain-physical-constant: engineering guard tied to integer ADC-like raw scale supplied in source facts; warning only, not auto-exclusion]`.

Trimming uses the onset-aligned response interval:

- `t_start = t_baseline_end` `[domain-physical-constant: per §3.3 convention, t_baseline_end is the index of the first post-baseline response sample; the response interval is the half-open range [t_baseline_end, n_rows)]`.
- `n_available = n_rows - t_start`.
- `clip_length_L = floor(percentile(n_available, 5))` over QC-passing samples `[runtime-determined-by-prespecified-algorithm: 5th percentile is a robust low-tail summary preserving at least 95% without truncation; percentile convention follows robust descriptive-statistics practice]`.

Per-sample handling is prespecified:

- If `n_available >= L`, trim to `[t_start, t_start + L)`.
- If `n_available < L`, compute relative deficit `(L - n_available) / L`.
- `PADDING_RELATIVE_DEFICIT_THRESHOLD = [runtime-determined-by-prespecified-algorithm]` via Tukey upper inner fence `Q3 + 1.5 × IQR` on relative deficits among otherwise QC-passing samples shorter than `L`.
- If the relative deficit is at or below `PADDING_RELATIVE_DEFICIT_THRESHOLD`, tail-pad by repeating the last available value and record `padded=True`.
- If the relative deficit exceeds `PADDING_RELATIVE_DEFICIT_THRESHOLD`, exclude the sample and add `clip-length insufficient` to `fail_reasons`.
- If no otherwise QC-passing sample is shorter than `L`, the padding threshold is stored as null and no padding is performed.

No fixed `10%` padding rule is used.

## 4. Pipeline A — Deep Learning Time-Series Tensor

Pipeline A is the unnormalized, sign-preserving response tensor consumed by deep-learning models in the modeling stage. It carries the fractional-corrected (per §3.7), response-trimmed (per §3.7) gas-channel signal for every sample in `in_final_output=True`.

Output path: `data/processed/sequences/pipeline_a_raw.npz`.

The NPZ archive contains:

- `x`: float array with shape `[N, L, 11]` where `11` is `[domain-physical-constant: gas-channel count]`; `N` and `L` are `[runtime-determined-by-prespecified-algorithm]`.
- `sample_id`: sample identifiers aligned to the first axis of `x`.
- `label`: binary labels aligned to the first axis of `x`. Labels are stored for downstream traceability only and are not used to compute any preprocessing parameter.
- `channel_names`: the ordered list S1, S2, ..., S11 `[domain-physical-constant: gas-channel layout]`.
- `clip_length_L`: `L` `[runtime-determined-by-prespecified-algorithm]`.

Pipeline A values are fractional-corrected per §3.7, response-trimmed per §3.7, sign-preserving per §7, and **unnormalized**. No z-score or min-max scaling is applied at preprocessing time. Z-score normalization is deferred to the modeling stage and must use training-fold statistics only.

Normalization utilities are implemented in the preprocessing codebase but are not invoked by Stage 1 preprocessing:

- `compute_pipeline_a_stats(x_train)` returns training-only per-channel mean and standard deviation.
- `apply_pipeline_a_normalization(x, stats)` applies supplied training statistics without mutating its input.
- Standard deviations below `1e-8` are clamped to `1.0` `[signal-processing-literature: numerical stability convention for degenerate variance]`.

## 5. Pipeline B — Engineered-Feature Table

Pipeline B is the unnormalized, sign-preserving engineered-feature table consumed by traditional machine-learning models in the modeling stage.

Output path: `data/processed/features/pipeline_b_raw.csv`.

Shape and columns:

- Feature count: `99` `[signal-processing-literature: 9 feature families × 11 gas channels based on e-nose response-curve feature taxonomy]`.
- Two traceability columns are appended: `sample_id` and `label`. The CSV therefore has `99 + 2` columns `[domain-physical-constant: traceability contract plus feature taxonomy]`.
- Rows correspond exactly to `sample_index.csv` rows where `in_final_output=True`, sorted by `sample_id`.

For each fractional-corrected, response-trimmed gas-channel signal `s` of length `L`, the following nine features are computed (yielding `9 × 11 = 99` columns when stacked across S1–S11):

1. `MPD` — max positive displacement: `max(s)` if any value of `s` is greater than `0.0`, else `0.0` `[signal-processing-literature: maximum response features in e-nose taxonomy; zero is sign boundary]`.
2. `MND` — max negative displacement: `min(s)` if any value of `s` is less than `0.0`, else `0.0` `[signal-processing-literature: sign-preserving counterpart to maximum response features; zero is sign boundary]`.
3. `AUC` — signed area under curve: trapezoidal integral over the response window `[signal-processing-literature: integrals/area values are standard original-response e-nose features; Yan et al. 2015]`.
4. `Mean` — interval mean response: arithmetic mean of `s` over the full `L` `[signal-processing-literature: interval mean response summaries are standard original-response e-nose features; Yan et al. 2015]`.
5. `MAS` — max ascending slope: `max(np.diff(s))` `[signal-processing-literature: derivative/slope features are standard original-response e-nose features; Yan et al. 2015]`.
6. `MDS` — max descending slope: `min(np.diff(s))` `[signal-processing-literature: derivative/slope features are standard original-response e-nose features; Yan et al. 2015]`.
7. `T90` — response time: the first index `t` where `abs(s[t]) >= 0.90 × max(abs(s))`; if this magnitude is never reached, T90 is set to `L` `[signal-processing-literature: T0-90%/T90 response-time features are standard gas-sensor response descriptors; Yan et al. 2015]`.
8. `ERR` — early response rate: `mean(s[:ceil(L × 0.20)]) / mean(s)` with denominator guard `1e-8`; if the denominator magnitude is below the guard, ERR is set to `0.0` `[signal-processing-literature: windowed time-slicing and early transient summaries are established e-nose features; 0.20 is retained as a prespecified one-fifth early-window partition, not cohort-derived]`.
9. `LMR` — late-segment mean: `mean(s[-ceil(L × 0.50):])` `[signal-processing-literature: interval mean/windowed time-slicing response summaries are established e-nose features; 0.50 is retained as a prespecified half-window late-response partition, not cohort-derived]`.

Feature values preserve sign wherever the feature has signed meaning. Absolute values are used only inside the T90 thresholding step, because T90 measures the time to reach the magnitude of peak response.

Normalization utilities are implemented in the preprocessing codebase but are not invoked by Stage 1 preprocessing:

- `compute_pipeline_b_stats(x_train_df)` returns training-only per-feature mean and standard deviation.
- `apply_pipeline_b_normalization(x_df, stats)` applies supplied training statistics without mutating its input and without normalizing the `sample_id` or `label` columns.
- Standard deviations below `1e-8` are clamped to `1.0` `[signal-processing-literature: numerical stability convention for degenerate variance]`.

## 6. Output Contracts And Audit Artifacts

### 6.1 Protocol Snapshot

`docs/protocol/preprocessing_plan.md` is a tracked audit artifact. It is authored independently from the operational plan, frozen on `preprocessing-prespecified`, and not edited afterward.

### 6.2 `data/processed/sample_index.csv`

Required columns:

- `sample_id`
- `filename`
- `collection_time` `[domain-physical-constant: traceability column for downstream temporal analyses; raw value sourced from metadata.csv::collection_time]`
- `label`
- `disease_category`
- `n_rows`
- `env_sensor_ok`
- `qc_status`
- `fail_reasons`
- `fallback_used`
- `padded`
- `in_final_output`

Rows: `1040` `[domain-physical-constant: source cohort size supplied before plan authoring]`.

### 6.3 `data/processed/qc_reports/qc_summary.csv`

Required columns:

- `sample_id`
- `qc1_env_sensor_fault_pass`
- `qc2_acquisition_truncated_pass`
- `qc3_unstable_baseline_pass`
- `qc4_insufficient_response_pass`
- `qc5_sensor_saturation_pass`
- `qc6_excessive_spikes_pass`
- `clip_length_pass`
- `fail_reasons`
- `qc_status`

Rows: `1040` `[domain-physical-constant: source cohort size supplied before plan authoring]`.

### 6.4 `data/processed/qc_reports/qc_diagnostics.csv`

Required diagnostic groups:

- `sample_id`, `n_rows`, `t_baseline_end`, `fallback_used`
- S12 and S13 baseline summaries
- per-channel baseline means, standard deviations, and CVs
- per-channel signed response amplitudes
- per-channel saturation fractions
- per-channel spike counts
- environmental compensation diagnostics
- trimming diagnostics including `n_available`, `clip_length_L`, relative deficit, and `padded`

Rows: `1040` `[domain-physical-constant: source cohort size supplied before plan authoring]`. QC-1 failures carry NaN values for QC-2 through QC-6 diagnostics.

### 6.5 `configs/preprocessing.yaml`

This YAML file is populated during Phase 1 and must contain:

- `provenance_schema_version`: `1` `[domain-physical-constant: first preprocessing provenance schema]`
- `random_seed`: `42` `[domain-physical-constant: project reproducibility convention]`
- `channels`
- `baseline_detection`
- `qc_thresholds`
- `smoothing`
- `environmental_compensation`
- `correction_and_trimming`
- `pipeline_b_features`
- `provenance`

Every numeric YAML value must carry a provenance label, and every literature-derived value must carry its citation key.

### 6.6 `data/processed/discovery_report.json`

Required top-level keys:

- `run_metadata` including seed `42` `[domain-physical-constant: project convention]`
- `label_blinding_statement`
- `baseline_detection`
- `qc1_env_sensor_fault`
- `qc3_baseline_cv`
- `qc4_response_amplitude`
- `qc5_saturation`
- `qc6_spikes`
- `environmental_compensation`
- `trimming`
- `halt_or_warning_events`

### 6.7 `data/processed/sequences/pipeline_a_raw.npz`

Required NPZ arrays per §4:

- `x` with shape `[N, L, 11]` where `11` is `[domain-physical-constant: gas-channel count]` and `N`, `L` are `[runtime-determined-by-prespecified-algorithm]`
- `sample_id`
- `label`
- `channel_names`: ordered S1–S11 `[domain-physical-constant: gas-channel layout]`
- `clip_length_L`

Row count `N` equals the number of `in_final_output=True` rows in `sample_index.csv`. Values are fractional-corrected, response-trimmed, sign-preserving, and unnormalized.

### 6.8 `data/processed/features/pipeline_b_raw.csv`

Per §5:

- `99` feature columns `[signal-processing-literature: 9 feature families × 11 gas channels based on e-nose response-curve feature taxonomy]` — the nine families MPD, MND, AUC, Mean, MAS, MDS, T90, ERR, LMR computed for each gas channel S1–S11
- two traceability columns: `sample_id`, `label`
- total: `99 + 2` columns `[domain-physical-constant: traceability contract plus feature taxonomy]`

Rows correspond to `sample_index.csv` filtered to `in_final_output=True`, sorted by `sample_id`. Values are fractional-corrected, response-trimmed, sign-preserving, and unnormalized.

## 7. Binding Methodological Constraints

- Disease labels are never used to compute preprocessing thresholds, smoothing choices, environmental coefficients, onset fallback values, clip length, or feature definitions.
- Response sign is preserved in Pipeline A and Pipeline B.
- Z-score normalization is deferred to modeling and must use training-fold statistics only.
- The study uses a single-device. Prohibited terms: `device_regime`, `Regime A`, `Regime B`.
- Onset detection uses all `11` gas channels `[domain-physical-constant: hardware channel layout]`; no sentinel channel is selected.
- `RANDOM_SEED = 42` `[domain-physical-constant: project convention]` is logged in every run, even when no random operation occurs.
- Numeric provenance uncertainty is a halt trigger, not an erratum candidate.

## References

- Cohen, J. *Statistical Power Analysis for the Behavioral Sciences*. 2nd ed. Lawrence Erlbaum Associates, 1988. Used only to reject the unsupported `R² = 0.10` draft threshold; Cohen's conventional regression effect sizes are not adopted as preprocessing thresholds.
- Gutierrez-Osuna, R. "Signal conditioning and preprocessing." In Pearce, T. C., Schiffman, S. S., Nagle, H. T., and Gardner, J. W., eds. *Handbook of Machine Olfaction: Electronic Nose Technology*. Wiley-VCH, 2002.
- Montgomery, D. C. *Introduction to Statistical Quality Control*. Wiley.
- NIST/SEMATECH e-Handbook of Statistical Methods. Sections on control charts and outlier detection.
- O'Brien, R. M. "A caution regarding rules of thumb for variance inflation factors." *Quality & Quantity*, 2007.
- Otsu, N. "A threshold selection method from gray-level histograms." *IEEE Transactions on Systems, Man, and Cybernetics*, 1979.
- Press, W. H., Teukolsky, S. A., Vetterling, W. T., and Flannery, B. P. *Numerical Recipes: The Art of Scientific Computing*. Cambridge University Press.
- Savitzky, A., and Golay, M. J. E. "Smoothing and differentiation of data by simplified least squares procedures." *Analytical Chemistry*, 1964.
- Scott, D. W. *Multivariate Density Estimation: Theory, Practice, and Visualization*. Wiley, 1992.
- Tukey, J. W. *Exploratory Data Analysis*. Addison-Wesley, 1977.
- Western Electric Company. *Statistical Quality Control Handbook*, 1956.
- Yan, J., Guo, X., Duan, S., Jia, P., Wang, L., Peng, C., and Zhang, S. "Electronic nose feature extraction methods: A review." *Sensors*, 2015.
