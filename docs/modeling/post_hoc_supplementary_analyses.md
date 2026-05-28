# Post-Hoc Supplementary Analyses

This document records the post-hoc supplementary analyses conducted after the frozen modeling stage of the GC VOC clinical study. It supplements the locked primary analysis with reviewer-facing uncertainty quantification and methodological context, while leaving the prespecified modeling-stage results, selected model, thresholds, calibrators, normalizers, transformers, train/test split, and fold structure unchanged. Frozen upstream facts in this document are copied from `docs/modeling/validated_modeling_run_summary.md`, and later sections are populated only after the corresponding Session 1-4 Plan 1 outputs are generated.

## Post-Hoc Declaration

> This analysis is post-hoc, conducted after the `modeling-frozen` tag, and is provided as supplementary evidence only. It does not modify or replace the prespecified primary modeling analysis, selected model, model parameters, thresholds, calibrators, normalizers, transformers, train/test split, fold structure, or any primary-analysis artifact.

## Upstream Frozen-Model Facts

The frozen modeling-stage record in `docs/modeling/validated_modeling_run_summary.md` establishes the following upstream facts for Plan 1:

- Best model: `minirocket_ridge`
- Locked train/test split: `n_train = 777`, `n_test = 195`
- Best-model OOF AUC-ROC: `0.9246` with `95%` CI `0.9048-0.9427`
- Best-model test AUC-ROC: `0.9223` with `95%` CI `0.8831-0.9585`
- Best-model test Brier: pre-Platt `0.1193`, post-Platt `0.1211`, delta `0.0018`
- Calibration slope: `0.9046`
- Calibration intercept: `0.3397`
- Raw Youden threshold: `threshold_p_raw = 0.341894`
- Calibrated equivalent threshold: `threshold_p_final_equivalent = 0.219589`
- Test-set threshold-dependent counts at `threshold_p_raw`: TP/TN/FP/FN = `74 / 97 / 16 / 8`
- Actual event-per-candidate-parameter ratio: `413 / 25 = 16.5` EPV

## Sample Size Reconstruction

Reconstruction under the prospectively registered parameters (`anticipated_auc=0.90`, `candidate_predictor_parameters=25`, `anticipated_prevalence=0.50`) yields `n_b1=385`, `n_b2=1374`, `n_b3=368`, and `n_b4=489`, with `n_required=1374` driven by the van Smeden B2 MAPE criterion. The actual development cohort (`n=972`, `413` events, `EPV=16.5`) meets B1, B3, and B4 but falls short of B2 by `402` samples. See report subsection "Sample-size reconstruction: shortfall against criterion B2" for full disclosure.

## Bootstrap Optimism-Corrected Validation

Under the Plan 1 §3.4 transient-refit procedure for `minirocket_ridge`, the apparent discrimination on the locked 777-sample training set was `auc_full_apparent=1.000000`. The mean bootstrap optimism was `0.020074`, yielding a supplementary optimism-corrected AUC-ROC of `0.979926` with a descriptive 2.5/97.5 percentile interval of `0.971265-0.988045`. This corrected value remains higher than the frozen nested-CV OOF AUC `0.9246` and frozen test AUC `0.9223`, but those frozen OOF and test estimates remain the primary discrimination evidence; the optimism-corrected value is supplementary context only.

## Decision-Curve Threshold-Uncertainty Sensitivity

(Pending — to be populated by Session 3 of Plan 1.)

## Calibration Goodness-Of-Fit Tests

(Pending — to be populated by Session 4 of Plan 1.)

## Execution Ledger

| Session | Date (UTC) | Commit hash | Artifacts produced |
|---|---|---|---|
| 1 | 2026-05-28T03:06:01Z | `<placeholder; user fills after commit>` | `outputs/analysis/posthoc_sample_size_reconstruction.md`; `outputs/analysis/posthoc_sample_size_reconstruction.json`; `docs/modeling/post_hoc_supplementary_analyses.md` |
| 2 | 2026-05-28T03:46:22Z | `<placeholder; user fills after commit>` | `outputs/analysis/posthoc_optimism_correction_minirocket_ridge.csv`; `outputs/analysis/posthoc_optimism_correction_minirocket_ridge.md`; `outputs/logs/posthoc_optimism_correction.log`; `docs/modeling/post_hoc_supplementary_analyses.md` |
