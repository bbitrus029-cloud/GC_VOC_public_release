# Post-Hoc Supplementary Analyses

This document records the post-hoc supplementary analyses conducted after the frozen modeling stage of the GC VOC clinical study. It supplements the locked primary analysis with reviewer-facing uncertainty quantification and methodological context, while leaving the prespecified modeling-stage results, selected model, thresholds, calibrators, normalizers, transformers, train/test split, and fold structure unchanged. Frozen upstream facts in this document are copied from `docs/modeling/validated_modeling_run_summary.md`, and the four analysis sections below were populated across Sessions 1–4 of Plan 1 and are now complete.

**Intended use and target population.** The model's intended use is risk stratification / triage within a high-risk or symptomatic population already within the gastroenterology diagnostic pathway, consistent with the endoscopy-and-histology-enriched study cohort, for which the `0.10` and higher prevalence scenarios are the most relevant. The model is not intended for, and does not demonstrate significant net benefit in, natural-population mass screening; the `0.01` and `0.05` prevalence scenarios are reported as robustness boundaries.

**Plan 1 status.** Plan 1 (post-hoc supplementary analyses) is complete as of this consolidation. All four analysis sections are now populated. Figure rendering is deferred to Plan 2, which will consume the persisted data feeds in `outputs/analysis/posthoc_threshold_uncertainty_minirocket_ridge_dca_curve_data.csv`, `outputs/analysis/posthoc_threshold_uncertainty_minirocket_ridge.csv`, `outputs/analysis/posthoc_sample_size_reconstruction.json`, `outputs/analysis/posthoc_optimism_correction_minirocket_ridge.csv`, and `outputs/analysis/posthoc_calibration_gof_minirocket_ridge.csv`.

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

### Registered sample-size basis (study-protocol context)

The study protocol registered a sample-size justification using two parallel tracks. Track 1 considered the statistical power of single-group AUC significance testing (Power = 0.90, alpha = 0.05, anticipated AUC approximately 0.90). Track 2 applied the EPV >= 10 model-stability principle: with approximately 25 candidate predictor parameters, a minimum of 10 x 25 = 250 events was required in the training set; at an anticipated high-risk-group prevalence of approximately 50% this corresponds to a 500-sample training set, and as the training set constitutes 80% of the cohort, to a net cohort of 500 / 0.80 = 625. Track 2 (625) exceeded Track 1, so 625 was adopted as the net required sample size; allowing for an approximately 25% potential dropout rate gives a planned recruitment target of 625 / (1 - 0.25) = 834. The actual in-final-output development cohort (`n=972`, `413` events, `EPV=16.5`) exceeds both the registered net requirement (625) and the registered recruitment target (834), and exceeds the EPV >= 10 standard on which the registration was based. The post-hoc Riley 2020 reconstruction reported above applies a stricter, more recent four-criterion framework; the B2 (MAPE-precision) shortfall reflects the higher demands of that post-2020 framework relative to the EPV-based criterion used at registration, not a failure to meet the registered target. The protocol additionally described an optional preliminary-validation sub-study of approximately 100 additional subjects for SOP establishment; this sub-study was not conducted, as a sample of that size was judged insufficient to yield a meaningful preliminary validation. That sub-study was separate from, and did not enter, the main-cohort sample-size calculation, so its omission does not affect the adequacy argument for the main development cohort.

## Bootstrap Optimism-Corrected Validation

Under the Plan 1 §3.4 transient-refit procedure for `minirocket_ridge`, the apparent discrimination on the locked 777-sample training set was `auc_full_apparent=1.000000`. The mean bootstrap optimism was `0.020074`, yielding a supplementary optimism-corrected AUC-ROC of `0.979926` with a descriptive 2.5/97.5 percentile interval of `0.971265-0.988045`. This corrected value remains higher than the frozen nested-CV OOF AUC `0.9246` and frozen test AUC `0.9223`, but those frozen OOF and test estimates remain the primary discrimination evidence; the optimism-corrected value is supplementary context only.

The apparent AUC reaching `1.000000` indicates that the locked 777-sample training set is fully separable in the approximately 10,000-dimensional MiniRocket feature space. Under such separability, standard Harrell optimism correction tends to underestimate true optimism because bootstrap apparent performances are compressed against the `1.0` ceiling. That is the methodological reason the supplementary optimism-corrected AUC-ROC (`0.979926`) remains substantially higher than the frozen nested-CV OOF AUC (`0.9246`): the OOF estimate is derived from genuine out-of-sample predictions and is therefore the more trustworthy generalization estimate. Accordingly, the frozen OOF and independent-test AUC values, not the optimism-corrected value, remain the primary discrimination evidence.

## Decision-Curve Threshold-Uncertainty Sensitivity

Recomputing the OOF Youden threshold from `binary_minirocket_ridge_nested_cv_oof.csv` reproduced the frozen mapping value exactly within the required tolerance (`threshold_p_raw = 0.341893878614765`, agreement delta `0.000000000000000` vs `binary_minirocket_ridge_threshold_mapping.json`). The stratified `n_boot=2000` percentile bootstrap yielded a 95% CI of `0.341893878614765-0.421007456988883`, with the lower bound coinciding numerically with the point estimate in this run.

Across the three prevalence scenarios, the operative threshold is unchanged and the fixed-threshold sensitivity analysis remains directionally stable over the CI band. At prevalence `0.01` and `0.05`, model net benefit remained negative for all three operating points but was consistently much less negative than the treat-all strategy; at prevalence `0.10`, model net benefit remained positive across the band (`0.0240` at the frozen point estimate, `0.0031` at the upper-CI threshold). Moving from the frozen point estimate to the upper-CI threshold traded some sensitivity (`0.9024` to `0.8415`) for a modest reduction in false-positive rate (`0.1416` to `0.1239`), without changing the conclusion that the frozen `threshold_p_raw` remains the operative threshold and this post-hoc sensitivity analysis does not modify any frozen primary-analysis artifact.

This prevalence pattern has a straightforward intended-use interpretation. Model net benefit is positive across the full threshold-CI band at prevalence `0.10`, but negative at `0.01` and `0.05`. The negative values at very low prevalence reflect the intrinsic attenuation of decision-curve net benefit when an operating point optimized on an enriched cohort is transported to a very-low-prevalence population, not a deficiency in discrimination, which is quantified separately by the frozen OOF/test AUC-ROC values of approximately `0.92`. Even where model net benefit is negative it remains far above the treat-all strategy at every prevalence; for example, at prevalence `0.01` the frozen operating-point comparison is `-0.0637992071` for the model versus `-0.5043166563` for treat-all. The `0.01` and `0.05` scenarios therefore function as robustness boundaries, whereas the `0.10` scenario is more aligned with the model's intended high-risk / symptomatic deployment context.

The threshold-uncertainty operating-point table and the background DCA-curve table are intentionally orthogonal outputs: the former captures uncertainty around the three fixed OOF-derived operating points, whereas the latter stores the standard per-prevalence DCA curves over the 98-point threshold grid for downstream visualization. Plan 1 renders no figure; Plan 2 will consume `outputs/analysis/posthoc_threshold_uncertainty_minirocket_ridge_dca_curve_data.csv` and overlay the three operating points from `outputs/analysis/posthoc_threshold_uncertainty_minirocket_ridge.csv`.

## Calibration Goodness-Of-Fit Tests

Using the locked `p_cal` values, the Hosmer-Lemeshow grouped test produced a chi-square statistic of `29.4500913288` with `df = 8` and `p = 0.0002642873`, whereas the Spiegelhalter test produced `z = 1.7883973234` with `p = 0.0737119278`. These two tests assess different aspects of calibration and are therefore not contradictory: Spiegelhalter is an overall mean-calibration test and does not indicate clear systematic over- or under-prediction, while Hosmer-Lemeshow is a grouped decile-based goodness-of-fit test and is more sensitive to local stratum-level deviation. Platt calibration is monotonic and does not change AUC-ROC or AUC-PR; it may change Brier score, calibration slope, and calibration intercept.

Taken together with the frozen post-Platt Brier `0.1211`, calibration slope `0.9046`, and calibration intercept `0.3397`, the supplementary calibration evidence is most consistent with mild over-confidence rather than gross calibration failure: the grouped test flags some decile-level deviation, but the overall-bias test is non-significant and the continuous calibration summaries remain the more informative primary descriptors. As throughout Plan 1, these GOF tests are descriptive supplementary evidence only, not pass/fail gates.

## Execution Ledger

| Session | Date (UTC) | Commit hash | Artifacts produced |
|---|---|---|---|
| 1 | 2026-05-28T03:06:01Z | `73dd2bc` | `outputs/analysis/posthoc_sample_size_reconstruction.md`; `outputs/analysis/posthoc_sample_size_reconstruction.json`; `docs/modeling/post_hoc_supplementary_analyses.md` |
| 2 | 2026-05-28T03:46:22Z | `beacefd` | `outputs/analysis/posthoc_optimism_correction_minirocket_ridge.csv`; `outputs/analysis/posthoc_optimism_correction_minirocket_ridge.md`; `outputs/logs/posthoc_optimism_correction.log`; `docs/modeling/post_hoc_supplementary_analyses.md` |
| 3 | 2026-05-28T08:34:50Z | `2d89ba6` | `outputs/analysis/posthoc_threshold_uncertainty_minirocket_ridge.csv`; `outputs/analysis/posthoc_threshold_uncertainty_minirocket_ridge_dca_curve_data.csv`; `outputs/analysis/posthoc_threshold_uncertainty_minirocket_ridge.md`; `docs/modeling/post_hoc_supplementary_analyses.md` |
| 4A | 2026-05-28T09:09:01Z | `<placeholder; user fills after commit>` | `outputs/analysis/posthoc_calibration_gof_minirocket_ridge.csv`; `outputs/analysis/posthoc_calibration_gof_minirocket_ridge.md` |
| 4B | 2026-05-28T09:27:41Z | `<placeholder; user fills after commit>` | `docs/modeling/post_hoc_supplementary_analyses.md` |
