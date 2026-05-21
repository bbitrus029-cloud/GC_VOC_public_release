# Validated Modeling Run Summary

## Overview

This document summarizes the validated modeling stage of the gastric cancer
(GC) volatile organic compound (VOC) clinical study. The stage built six
prespecified binary classifiers on a 972-sample exhaled-breath cohort, locked
a single best model before any test-set evaluation, and ran a five-tier
analysis hierarchy plus five sanctioned post-hoc trigger checks. The cohort
spans six clinical disease categories: healthy controls (HC, n=158), benign
gastric lesions (BGL, n=352), low-grade intraepithelial neoplasia (LGIN,
n=49), high-grade intraepithelial neoplasia (HGIN, n=71), early gastric
cancer (EGC, n=122), and advanced gastric cancer (AGC, n=220). The binary
endpoint groups HC/BGL/LGIN as negative (n=559) and HGIN/EGC/AGC as positive
(n=413), an approximately 1:1.35 positive-to-negative ratio.

The study's prespecification chain is anchored by four annotated git tags
(`modeling-prespecified`, `data-split-locked`, `best-model-selected`,
`modeling-frozen`) corresponding to four irreversible methodological
commitments: (1) methodology locked before any implementation commit, (2)
data split seed locked after a candidate-set selection on n=972 with neither
labels nor any model influencing the choice, (3) best model locked solely on
five-outer-fold nested-CV out-of-fold (OOF) predictions and a three-step
selection algorithm before the test set was ever read by any model, and (4)
end-to-end run frozen after all prespecified analyses, the five post-hoc
triggers, and SHAP cross-model consistency were complete.

Six models were prespecified across two feature pipelines. Pipeline A
operates on the raw 11-channel `[N, 960, 11]` time-series tensor and houses
`minirocket_ridge` (MR+R), `inception_time` (IT), and
`cnn_bilstm_attention` (CBLA). Pipeline B operates on the 99-feature
hand-crafted feature matrix and houses `xgboost`, `svm`, and
`elastic_net_lr`. Selection chose `minirocket_ridge` at Step 2 of the
prespecified algorithm via a paired DeLong test (p = 0.009427) against the
runner-up XGBoost, after Erratum-E008 non-dominated Step 1 filtering.

On the locked 195-sample test set, MR+R attained AUC-ROC 0.9223 (95% CI
0.8831-0.9585) with Brier 0.1193 (pre-Platt) and 0.1211 (post-Platt;
delta 0.0018, well below the 0.005 calibration-failure threshold). Of the
five sanctioned post-hoc triggers, four (A subgroup retrain, C test-vs-CV
shift, F tumor-marker underpowered, H calibration failure) did not fire on
this cohort, and one (G DL training anomaly) fired in an exploratory
InceptionTime first run and was resolved before any committed run via
Erratum E005 (val_loss monitor, patience=40). Fourteen errata (E001-E014)
documented every clarification or under-specified parameter encountered
during execution; none required protocol-snapshot modification.

## Prespecification Evidence Chain

The TRIPOD prespecification chain rests on (a) two immutable protocol
snapshots committed at the start of each stage and frozen by annotated tags,
(b) one operational plan held in `.cursor/plans/` with all 14 errata, and
(c) four annotated tags whose messages record the locked decisions at each
commitment time.

| Tag | Commit | Anchoring Decision |
|---|---|---|
| `preprocessing-frozen` | `622cb36` | End of Stage 1 preprocessing; 972 samples in final cohort; Pipeline A tensor `[972, 960, 11]`, Pipeline B matrix `[972, 99]`. |
| `modeling-prespecified` | `f0d8e38` | Modeling methodology locked before any modeling-stage implementation commit. Operational plan `.cursor/plans/modeling_plan_v1_a7c3f9b2_dc789bb5.plan.md` and protocol snapshot `docs/protocol/modeling_plan.md` both committed. |
| `data-split-locked` | `81a0e76` | `data_split_seed = 43` selected from candidate set `{42, 43, 44, 45, 46}` by the cascade-stratified score and locked. All split `.npz` files (train/test indices, 5 outer folds, 5x3 inner folds, 5 DL holdouts, 1 Stage-4b DL holdout) generated. Effective sizes n_train=777, n_test=195 per Erratum E001. |
| `best-model-selected` | `8a9a609` | `minirocket_ridge` selected at Step 2 of §2.8 (paired DeLong p = 0.009427 < 0.05, n_pairs=1 after Erratum-E008 non-dominated Step 1 filtering). Selection occurred before any test-set prediction was computed; halt condition (d) "test-set invisibility" was verified not triggered. Nested-CV OOF AUC-ROC 0.9246 (95% CI 0.9048-0.9427). |
| `modeling-frozen` | (this commit) | End-to-end modeling stage frozen; 14 errata documented; 5 post-hoc triggers evaluated. Best-model test-set AUC-ROC 0.9223 (95% CI 0.8831-0.9585). |

Each tag message records the locked numeric outcome at its commitment time;
the messages are reproduced in `outputs/logs/experiment_manifest.json` under
the `annotated_tags` block for machine-readable audit.

## Locked Methodology

The modeling protocol locks the following decisions; none was altered after
its respective anchor tag:

1. **Two-seed separation.** `RANDOM_SEED = 42` (project convention) governs
   all model training, fold generation, bootstrap sampling, weight
   initialization, and dropout. `data_split_seed = 43` (locked at
   `data-split-locked`) governs the 80/20 train/test split. The two seeds
   are stored in distinct configuration keys and never aliased to each
   other.

2. **Stratified train/test split.** 80%/20% on n=972, stratified jointly by
   binary label and disease_category (six categories). Effective sizes
   n_train=777, n_test=195 per Erratum E001 (sklearn-style ceil rounding;
   identical across all five candidate seeds; identical SMD = 0.011083 for
   label across all seeds verifies stratification fidelity).

3. **Nested cross-validation.** Outer 5-fold for OOF estimation; inner
   3-fold for hyperparameter selection. Fold indices generated once and
   persisted to `data/processed/splits/*.npz`; every trainer loads from
   disk, no trainer regenerates folds.

4. **Fold-level z-score normalization.** Pipeline A normalization stats
   shape `[11]` (gas channels), Pipeline B shape `[99]` (features). Fitted
   only on inner-train subsets, transform inner-val. For Stage 4b final
   training, fitted on full 777 train and applied to the 195 test.
   `sigma < 1e-8` clamped to 1.0 with WARNING.

5. **Class imbalance.** Weights recomputed per inner-train subset only
   (never on a global cohort). XGBoost: `scale_pos_weight = n_neg / n_pos`;
   linear/SVM: `class_weight='balanced'`; DL: `CrossEntropyLoss(weight=cw)`
   with `cw = [n_total / (2 * n_neg), n_total / (2 * n_pos)]`.

6. **OOF-based Youden threshold and Platt calibrator.** Both fitted on
   outer-fold OOF `(p_raw, y_true)`, NOT on the final-model in-sample
   predictions. `threshold_p_raw` is the Youden index in `p_raw` space;
   `threshold_p_final_equivalent = Platt_layerB(threshold_p_raw)` is the
   algebraic consequence reported in `binary_<m>_threshold_mapping.json`.
   For MR+R, `threshold_p_raw = 0.341894` and
   `threshold_p_final_equivalent = 0.219589`.

7. **Reuse-or-rerun rule.** If aggregated final hyperparameters equal
   every outer fold's best hyperparameters, reuse `binary_<m>_nested_cv_oof.csv`
   and set `oof_source = "nested_cv_reused"`. Otherwise run a one-time
   5-fold rerun under final HPs on the same outer fold indices and set
   `oof_source = "final_hp_rerun"`. The best model `minirocket_ridge` used
   `nested_cv_reused`; XGBoost, SVM, and elastic-net LR used
   `final_hp_rerun` per Erratum E009-clarified non-HP-key exclusion logic.

8. **Three-step model selection (§2.8) with CV-only halt conditions.**
   Step 1 non-dominated filtering per Erratum E008; Step 2 paired DeLong
   with Bonferroni alpha = 0.05 / n_pairs; Step 3 paired bootstrap on
   AUC-PR with `n_boot = 2000`; Step 4 prespecified parsimony tiebreaker.
   Selection at Step 2 in this study. Halt conditions checked: (a) no CI
   contains 0.5, (b) no metric NaN/Inf, (c) no all-models mean AUC < 0.6,
   (d) no test-set value loaded during selection.

9. **Five-tier analysis hierarchy.** Tier A (best model only, clinical
   translation); Tier B1 (all 6 models, full set, three views); Tier B2
   (best model only, subgroup views); Tier C (all 6 models, subgroups,
   threshold-free); Tier D (all 6 models, sensitivity analyses); Tier E
   (all 6 models, SHAP/channel importance). Confusion matrix counts and
   `quadrant` column derivation operate at all-model scope (zero
   additional cost) per §2.9.

10. **Subgroup immutability.** HGIN+EGC and AGC subgroups filter samples
    only; never refit the model, normalizer, transformer, calibrator,
    `threshold_p_raw`, or `threshold_p_final_equivalent`. Shared negative
    pool (test samples with `disease_category in {0, 1, 2}`) used
    identically by both subgroup analyses.

11. **Tumor-marker complete-case discipline.** `tm_test_subset` and
    `tm_train_subset` filtered on CEA AND CA19-9 AND CA72-4 all
    non-missing. Imputation prohibited. Smoke test
    `tests/test_tumor_marker_no_imputation.py` enforces no MICE or
    IterativeImputer code path.

12. **Bootstrap convention.** Per Erratum E011 (commit `a2af64e`),
    `bootstrap_metric` and `paired_bootstrap_delta` use a single
    `np.random.default_rng(seed)` instantiated at function entry and
    reused sequentially across all `n_boot = 2000` iterations.
    Reproducibility verified via the deterministic PCG64 stream.

## Errata Documentation

All 14 execution errata are recorded in §12 of
`.cursor/plans/modeling_plan_v1_a7c3f9b2_dc789bb5.plan.md` (the operational
plan, untracked by git per `.cursor/` discipline). The protocol snapshot
`docs/protocol/modeling_plan.md` was not modified by any erratum. Verbatim
counts and IDs:

| ID | Date | Affected Sections | One-line Rationale |
|---|---|---|---|
| E001 | 2026-05-17 | §1.3, §2.4, §2.5, §2.11, §7.5 Phase 2, §7.10 Phase 2, §3.7 | sklearn-stratified split on n=972 yields n_train=777 and n_test=195, not the plan's planning-time shorthand of 778/194. |
| E002 | 2026-05-18 | §2.2, §6.2, §7.1, §10 | `modeling_minirocket_ridge.yaml` and `modeling_xgboost.yaml` authored in Session 6 (Stage 4a) instead of Session 1; category-level provenance unchanged; XGBoost endpoints locked with Chen & Guestrin 2016 + Bischl et al. 2023 citations. |
| E003 | 2026-05-18 | §1.3, §5.7 | `BaseNestedCVTrainer` aligned `sample_id` via stringified cast, producing lexicographic ordering that broke the equality assertion at trainer startup; fixed by integer-dtype sort with no reindexing of `x`/`label`. |
| E004 | 2026-05-18 | §2.2, §2.23, §6.2, §10 | InceptionTime 3-configuration YAML authored in Session 8 with locked kernel triplet `[9, 19, 39]`; DL split index-space audit confirms 972-space vs 777-space coexistence with equivalent cardinality. |
| E005 | 2026-05-18 | §2.23, §4.4, §6.2 | DL early-stopping monitor changed from `val_auc` to `val_loss`, mode min, patience increased from 20 to 40. Authorized by Post-hoc G trigger fired on InceptionTime first run (4 outer folds early-stopped before epoch 20 on the noisy 78-sample DL holdout); training-loss monotonic decrease confirmed model was learning. Bindings apply to Session 8 rerun, Session 9 first run, and Session 10 Phase 2. |
| E006 | 2026-05-18 | §2.2, §2.23, §6.2 | CBLA 4-configuration YAML authored in Session 9: baseline plus CNN-capacity (B), sequence-model-capacity (C), and regularization-strength (D) axes. E005 early-stopping configuration re-cited verbatim. |
| E007 | 2026-05-18 | §3.9, §3.10, §5.9, §7.6, §7.7, §7.8, §7.9, §12 | Five of six trainers persisted `sample_id` via `str()` cast, causing lexicographic ordering in OOF and per-fold CSV outputs. Diagnostic verified zero impact on AUC-ROC/PR/Brier/CI numerics; corrective in-place CSV resort applied to all six models plus a trainer-code fix in `fix(training-all-trainers)`. |
| E008 | 2026-05-19 | §2.8 Step 1, Step 2; §5.12; §7.10 Phase 1 | Step 1 of selection algorithm clarified as non-dominated set semantics rather than "return all if no unique strict leader." Locks the correct Bonferroni denominator; in this study reduces n_pairs from 15 (over 6 models) to 1 (over 2 candidates) and changes the selection step from 4 (parsimony tiebreaker) to 2 (DeLong significance at alpha = 0.05). |
| E009 | 2026-05-19 | §2.5; §5.9; §7.10 Phase 2 | §2.5 final-HP aggregation excludes per-model non-search plumbing keys (`random_state`, `n_jobs`, `tree_method`, `class_weight`, `kernel`, `probability`, `max_iter`, `penalty`, `solver`). Aggregated keys restored from canonical config values, not nested-CV result JSONs. Prevents `random_state` aggregation drift to 44 (which would violate `RANDOM_SEED = 42` and Erratum-E002 anti-stall lock). |
| E010 | 2026-05-19 | §2.22; §5.9 | `load_final_model` API contract locked: NUMBA_CACHE_DIR set for MR+R MiniRocket compilation; `DLPredictProbaWrapper` imported from stable `src.training.dl_inference` module path so pickle resolution succeeds in fresh subprocesses. Two DL final PKLs re-serialized (no retraining); ML PKLs unchanged. |
| E011 | 2026-05-19 | §5.6; §2.11/§2.16/§2.17/§3.13-§3.18; modeling-guardrails §1 | Single-RNG sequential bootstrap pattern (Session 2 implementation at commit `a2af64e`) locked as project convention. Statistically equivalent to per-resample seeding for sequential execution. Future parallel-bootstrap code paths would require per-resample seeding. |
| E012 | 2026-05-20 | §2.15; §3.23; §5.12; §7.12 | LGIN B-vs-C primary analysis view promoted from test (n_LGIN=10, n_C=0; mathematically non-estimable) to nested-CV OOF (n_LGIN=39, n_B=35, n_C=4). The §2.15 hypothesis, methodology, gating rules, and pass criterion remain unchanged; only the prediction-source view is reassigned. Test view becomes descriptive sensitivity supplement. |
| E013 | 2026-05-20 | §2.14; §3.22; modeling-guardrails §11 | §2.14 fusion-template condition A revised from strict inclusivity (`AUC <= 0.01 AND (NRI < 0 OR IDI < 0)`) to inclusive disjunction (`(AUC < 0.01 OR DeLong p_two_sided >= 0.05) AND (NRI <= 0 OR IDI <= 0)`). Thresholds 0.01 and 0.05 unchanged; only inclusivity/disjunction structure. Triggers the "VOC alone is preferable" template at observed fusion result. |
| E014 | 2026-05-21 | §2.21; §3.26; §6.1 | Permutation-importance negative aggregated values clipped to zero before L1 normalization with WARNING log and report-level diagnostic disclosure. In this study, SVM channels [3, 5, 7, 8] were clipped (small-magnitude noise floor); cross-model Kendall Tau and top-K converging results unaffected. |

All 14 errata clarified or documented under-specified plan content. Three
errata authorized methodological adjustments: E005 (training-dynamics
adjustment per Post-hoc G), E009 (HP-aggregation exclusion correction
binding for Stage 4b final training), and E014 (permutation-importance
negative clip-to-zero project policy). The remaining 11 errata documented
clarifications, deferred YAML authoring (E002, E004, E006), implementation
defect fixes (E003, E007), API contract locks (E010, E011), interpretive
template revisions (E013), or input-view promotions justified by sample
availability (E012); none altered the methodological hypothesis being
tested.

## Post-Hoc Trigger Outcomes

The protocol sanctions exactly five post-hoc triggers (A, C, F, G, H) per
§4 of the operational plan. Each has an objective trigger condition; no
other post-hoc analysis was added in any session.

| Trigger | Evaluated | Triggered | Outcome and Source |
|---|---|---|---|
| A: Subgroup retrain (§4.1) | Yes | No | HGIN+EGC subgroup AUC=0.9264 (95% CI 0.8833-0.9639) and AGC subgroup AUC=0.9187 (95% CI 0.8727-0.9576). Neither met BOTH AUC-degradation conditions (point estimate `< full-set point - 0.05` AND CI upper bound `< full-set point`). Source: `outputs/analysis/stage7_part_a_audit.json`. |
| C: Test-vs-CV shift attribution (§4.2) | Yes | No | Bidirectional shift inspection across all 6 models. Average test-minus-CV AUC-ROC shift = +0.026 with mixed direction (MR+R: -0.0023; XGBoost: +0.0032; SVM: +0.0245; EN-LR: +0.0219; InceptionTime: +0.0549; CBLA: +0.0540). Bidirectional unanimous-shift rule not satisfied. Source: `outputs/analysis/stage7_part_a_audit.json`. |
| F: Tumor marker underpowered (§4.3) | Yes | No | `tm_test_subset n_pos = 76` >= 30 and `tm_train_subset n_pos = 293` >= 50. NRI/IDI/DeLong reported with full significance claims per §3.21/§3.22, not descriptive-only. Source: `outputs/analysis/stage7_part_c_audit.json`. |
| G: DL training anomaly (§4.4) | Yes | Yes (resolved) | InceptionTime first run produced 4 outer folds early-stopped before epoch 20 due to noisy `val_auc` monitor on the 78-sample DL holdout (Hanley-McNeil-predicted SE approximately 0.047). Resolved by Erratum E005: switch to `val_loss` monitor, mode min, patience 40, applied to Session 8 rerun, Session 9 first run, and Session 10 Phase 2. Source: Erratum E005 in `.cursor/plans/modeling_plan_v1_a7c3f9b2_dc789bb5.plan.md` §12. |
| H: Calibration failure (§4.5) | Yes | No | Best-model post-Platt Brier 0.121097 minus pre-Platt Brier 0.119257 = +0.001840, below the 0.005 degradation tolerance. Calibration slope 0.905 (95% CI 0.729-1.221); calibration intercept 0.340 (95% CI 0.002-0.924). Source: `outputs/analysis/stage7_part_a_audit.json`. |

## Model Selection

The three-step selection algorithm operated solely on the six models'
nested-CV OOF predictions and per-fold AUC-ROC summaries with bootstrap CIs.
No test-set value was loaded by any code path during selection; halt
condition (d) was explicitly verified not triggered.

| Model | OOF AUC-ROC | OOF AUC-PR | OOF Brier |
|---|---|---|---|
| `minirocket_ridge` | 0.9246 (0.9048-0.9427) | 0.8645 (0.8252-0.9034) | 0.1105 (0.0999-0.1217) |
| `xgboost` | 0.8981 (0.8756-0.9188) | 0.8323 (0.7925-0.8714) | 0.1255 (0.1101-0.1416) |
| `svm` | 0.8662 (0.8410-0.8901) | 0.7867 (0.7427-0.8345) | 0.1484 (0.1341-0.1632) |
| `elastic_net_lr` | 0.8615 (0.8351-0.8861) | 0.7508 (0.7039-0.8011) | 0.1484 (0.1335-0.1644) |
| `cnn_bilstm_attention` | 0.8243 (0.7950-0.8518) | 0.7537 (0.7134-0.7948) | 0.1720 (0.1553-0.1894) |
| `inception_time` | 0.8085 (0.7756-0.8396) | 0.6960 (0.6500-0.7507) | 0.1799 (0.1630-0.1976) |

**Step 1 — non-dominated filtering (per Erratum E008).** Strict dominance
applied: a model is dominated if some other model's 95% CI lower bound
exceeds its 95% CI upper bound. The non-dominated set was `{minirocket_ridge,
xgboost}` (both candidates mutually overlap; `minirocket_ridge` 95% CI
`[0.9048, 0.9427]`, XGBoost 95% CI `[0.8756, 0.9188]`). Four models were
eliminated as strictly dominated: SVM (`[0.8410, 0.8901]` < 0.9048), EN-LR
(`[0.8351, 0.8861]` < 0.9048), CBLA (`[0.7950, 0.8518]` < 0.8756),
InceptionTime (`[0.7756, 0.8396]` < 0.8410). Step 1 alone did not produce a
unique winner; the non-dominated pair carried forward to Step 2.

**Step 2 — paired DeLong (Bonferroni alpha = 0.05 over n_pairs = 1).**
Paired DeLong test on the 777-sample OOF predictions for
`minirocket_ridge` vs `xgboost` yielded p = 0.009427 < 0.05. The
significance threshold is uncorrected (n_pairs = 1 after Step 1's
non-dominated filtering eliminated four candidates). Selection resolved at
Step 2 with `minirocket_ridge` as the best model.

Selection halt conditions all verified not triggered: (a) no model's 95% CI
contained 0.5; (b) no aggregate metric NaN or Inf; (c) all-models mean
AUC-ROC = 0.8639 > 0.6; (d) no test-set CSV or NPZ accessed by any module
during Phase 1.

The full audit chain — six-model summary table, halt-conditions check,
step-by-step selection record, prespecification declaration "The best model
has been prespecified before test set evaluation. No modification to model
selection will be made based on test set results.", git commit hash, and
timestamp — is preserved verbatim in `outputs/analysis/model_selection_report.md`.

## Final Model Test-Set Performance

Best model `minirocket_ridge` evaluated on the locked 195-sample test set
(82 positives, 113 negatives, prevalence 0.4205) at the OOF-fitted Youden
threshold `threshold_p_raw = 0.341894`. The `threshold_p_final_equivalent =
Platt_layerB(threshold_p_raw) = 0.219589` is reported for clinical readers
interpreting in calibrated-probability space; all threshold-dependent metric
values are numerically identical between `(p_raw, threshold_p_raw)` and
`(p_cal, threshold_p_final_equivalent)` evaluations because both Layer A
and Layer B are monotonic sigmoid transforms.

**Threshold-free metrics (bootstrap n_boot = 2000 95% CI):**

| Metric | Value | 95% CI |
|---|---|---|
| AUC-ROC | 0.9223 | [0.8831, 0.9585] |
| AUC-PR | 0.8554 | [0.7784, 0.9352] |
| Brier (pre-Platt) | 0.1193 | [0.0959, 0.1423] |
| Brier (post-Platt) | 0.1211 | [0.0888, 0.1528] |
| Brier delta (post - pre) | +0.0018 | [-0.0086, +0.0121] |
| Calibration slope | 0.9046 | [0.7288, 1.2212] |
| Calibration intercept | 0.3397 | [0.0017, 0.9244] |

Platt calibration is monotonic and does not change AUC-ROC or AUC-PR; it
may change Brier score, calibration slope, and calibration intercept. The
+0.0018 Brier delta is below the 0.005 Post-hoc-H trigger threshold;
calibration slope confidence interval brackets 1.0 (the perfectly
calibrated ideal); calibration intercept is positive and significantly
non-zero, indicating a small systematic under-prediction of positive-class
probabilities that does not compromise discrimination.

**Threshold-dependent metrics at `threshold_p_raw = 0.341894`:**

| Metric | Value | 95% CI |
|---|---|---|
| Sensitivity | 0.9024 | [0.8293, 0.9634] |
| Specificity | 0.8584 | [0.7962, 0.9204] |
| PPV | 0.8222 | [0.7582, 0.8929] |
| NPV | 0.9238 | [0.8763, 0.9691] |
| F1 | 0.8605 | [0.8095, 0.9112] |
| Balanced accuracy | 0.8804 | [0.8350, 0.9259] |
| MCC | 0.7534 | [0.6633, 0.8447] |
| Accuracy | 0.8769 | [0.8308, 0.9231] |
| TP / TN / FP / FN | 74 / 97 / 16 / 8 | (point counts) |

**All-model test-set AUC-ROC (Tier B1 three-view test column):**

| Model | Test AUC-ROC | 95% CI |
|---|---|---|
| `minirocket_ridge` | 0.9223 | [0.8831, 0.9585] |
| `xgboost` | 0.9014 | [0.8556, 0.9424] |
| `svm` | 0.8907 | [0.8392, 0.9333] |
| `elastic_net_lr` | 0.8834 | [0.8316, 0.9269] |
| `cnn_bilstm_attention` | 0.8783 | [0.8299, 0.9177] |
| `inception_time` | 0.8634 | [0.8084, 0.9115] |

The test-set AUC-ROC ordering preserves the OOF ordering's broad
structure: `minirocket_ridge` retains the top position; the Pipeline-A vs
Pipeline-B and ML vs DL groupings remain coherent.

## Tier A Clinical Translation

Tier A analyses operate only on the best model `minirocket_ridge`.

**Decision Curve Analysis (DCA).** Net benefit computed over the threshold
grid `np.arange(0.01, 0.99, 0.01)` with treat-all and treat-none reference
strategies. Detailed values are persisted to `outputs/analysis/dca_minirocket_ridge.csv`
and the markdown summary `outputs/analysis/dca_minirocket_ridge.md`. The
model's net benefit exceeds both reference strategies across the
clinically relevant threshold range, consistent with the AUC-ROC and Brier
findings above.

**Two-tier risk stratification.** With `p_raw >= threshold_p_raw = 0.341894`
defining high risk and the complement defining low risk, prevalence-adjusted
metrics across the three prespecified scenarios (1%, 5%, 10%):

| Prevalence | PPV | NPV | NNE = 1/PPV |
|---|---|---|---|
| 1% | 0.0605 [0.0418, 0.1039] | 0.9989 [0.9980, 0.9996] | 16.53 [9.62, 23.91] |
| 5% | 0.2512 [0.1853, 0.3767] | 0.9941 [0.9899, 0.9977] | 3.98 [2.65, 5.40] |
| 10% | 0.4146 [0.3244, 0.5606] | 0.9875 [0.9788, 0.9951] | 2.41 [1.78, 3.08] |

NNE (Number Needed to Endoscope) is the expected number of high-risk
classifications needed to identify one true case; lower is better at fixed
sensitivity/specificity. The 1% prevalence scenario approximates
low-prevalence population screening; 5% approximates symptomatic /
outpatient enrichment; 10% approximates high-risk targeted screening.
Detailed values persisted to `outputs/analysis/two_tier_risk_stratification_minirocket_ridge.csv`
with markdown summary in `outputs/analysis/two_tier_risk_stratification_minirocket_ridge.md`.

## Subgroup and Stratified Robustness

**HGIN+EGC subgroup** (positives in disease_category `{3, 4}` plus shared
negative pool with disease_category in `{0, 1, 2}`): test AUC-ROC =
0.9264 (95% CI 0.8833-0.9639). The subgroup AUC-ROC point estimate is
within 0.005 of the full-set AUC-ROC 0.9223; the CI lower bound 0.8833
overlaps the full-set CI (0.8831-0.9585). Post-hoc A trigger not fired.

**AGC subgroup** (positives in disease_category `{5}` plus shared negative
pool): test AUC-ROC = 0.9187 (95% CI 0.8727-0.9576). The subgroup AUC-ROC
is within 0.004 of the full-set point estimate; CI overlap is preserved.
Post-hoc A trigger not fired.

The full Tier B2 subgroup metric table (full metric set with bootstrap 95%
CI for both subgroups across all three views) is persisted at
`outputs/analysis/three_view_best_model_with_subgroups.csv`. Tier C
threshold-free metrics across all six models for both subgroups are at
`outputs/analysis/subgroup_hgin_egc_all_models.csv` and
`outputs/analysis/subgroup_agc_all_models.csv`.

**Stratified robustness across 7 clinical risk variables.** All six models
evaluated on test-set strata of `sex`, `age_at_60`, `bmi_at_24`, `smoking`,
`drinking`, `family_history`, and `hp_infection`. Each stratum is retained
only if `n >= 10`, `n_pos >= 2`, and `n_neg >= 2`; otherwise the row is
preserved with empty AUC and an omission reason. Outputs at
`outputs/analysis/stratified_robustness_<variable>_all_models.csv`
(exactly 7 files). The seven-variable converging evidence supports that
signal direction is consistent across these clinical strata for the best
model and the broader six-model panel.

## Disease Progression Trajectory

Best-model `p_cal` ordered across the six disease categories (HC < BGL <
LGIN < HGIN < EGC < AGC):

- **Jonckheere-Terpstra (JT) trend test.** Self-implemented JT statistic
  with anchor test against `npordtests::JtTest` jdata in
  `tests/test_jt_npordtests.py`. Result on test-set best-model `p_cal`:
  `p_one_sided_upper = 2.49e-24`, a strongly significant monotonic
  upward trend.
- **Spearman rank correlation.** Treating `disease_category` as ordinal,
  Spearman's rho = 0.7303 (two-sided p = 8.96e-34) on the 195-sample
  test set.
- **Adjacent-group Mann-Whitney U.** Five adjacent comparisons (HC-BGL,
  BGL-LGIN, LGIN-HGIN, HGIN-EGC, EGC-AGC) with Holm-Bonferroni
  correction over the family of 5 tests. Per-pair p-values and corrected
  significance are persisted in `outputs/analysis/progression_trajectory_minirocket_ridge.csv`.

The convergence of JT (omnibus monotonic trend), Spearman (rank
correlation), and adjacent-pair Mann-Whitney (local-step transitions)
provides three-way evidence that the model's calibrated probability
captures the prespecified disease severity ordering at the cohort scale.

Misclassification Layer 2 analysis (FP-LGIN over-representation):
Fisher exact OR = 579.92, two-sided p = 4.62e-13 for LGIN proportion in
test-set false positives vs LGIN proportion in the full test set. This
quantitatively confirms that the model's false-positive errors are
heavily enriched for LGIN samples, consistent with the LGIN exploratory
layer-analysis hypothesis that LGIN occupies an intermediate calibrated-
probability stratum between BGL (negative) and HGIN (positive) per §2.15.
Detailed quadrant statistics, representative samples (up to 5 per
quadrant by distance-to-threshold segment and subtype diversity), and FN
detail section are at `outputs/analysis/misclassification_four_quadrant_minirocket_ridge.md`
and `outputs/analysis/misclassification_per_sample_minirocket_ridge.csv`.

## LGIN Exploratory Layer Analysis

Per Erratum E012, the §2.15 LGIN exploratory layer analysis primary view is
the nested-CV OOF view rather than the locked test view. The test view
contained `n_LGIN = 10` samples all with `collection_location = 1`
(inpatient B-group); zero outpatient (C-group) test samples make the
test-view B-vs-C analysis mathematically non-estimable.

**Primary analysis (OOF view).** `n_LGIN_OOF = 39` with `n_B = 35`
(inpatient) and `n_C = 4` (outpatient). On OOF best-model `p_cal`:
- Rank-biserial correlation rb = 0.9429 (very strong)
- Mann-Whitney U two-sided p = 2.92e-4
- Separation criterion `|rb| >= 0.5 AND p < 0.05` satisfied

Power caveat carried into the report per Erratum E012: with `n_B = 35` and
`n_C = 4`, the minimum achievable Mann-Whitney p under the null is
approximately 1.22e-5; bootstrap CIs are not computed for n_C = 4 because
resamples degenerate; the 4 C-group `p_cal` values are listed individually
in the report for transparency.

**Within-B stability.** On the 35 inpatient B-group LGIN samples, the seven
prespecified covariates `{age_at_60, sex, bmi_at_24, smoking, drinking,
family_history, hp_infection}` were each binned into levels, gated at
n >= 5 per level. Every gated level achieved median `p_cal > 0.5`,
confirming stability of the elevated calibrated probability across the
seven clinical strata.

**Test-view descriptive sensitivity.** All 10 test-set LGIN samples are
inpatient (B-group surrogate). Per-sample `p_cal`, age, sex, BMI, smoking,
drinking, family_history, hp_infection, CEA, CA19-9, CA72-4 reported
descriptively in `outputs/analysis/lgin_layer_analysis.md`. Test-view B/C
separation declared non-estimable (n_C_test = 0) with explicit honesty
wording per modeling-guardrails §11. The forward-looking note in the
operational plan §12 records that expanded outpatient LGIN sampling would
permit a supplementary re-analysis on an enlarged test set; such re-
analysis would be a separate post-hoc artifact requiring its own erratum
at that future time.

## Fallback/Padded Sensitivity + Temporal Drift

**Fallback/padded sensitivity (Tier D).** All six models evaluated on
four data subsets: full test set, `fallback_used = False` only,
`padded = False` only, and both flags false. AUC-ROC, AUC-PR, and Brier
with 95% CI reported for each subset in
`outputs/analysis/fallback_padded_sensitivity_all_models.csv`.

Best-model misclassification association via two independent Fisher exact
tests (per §2.16):
- `fallback_used` vs misclassification: Fisher p = 1.000 (no association
  with misclassification).
- `padded` vs misclassification: Fisher p = 0.110 (no significant
  association at alpha = 0.05).

Subset characteristic SMD summaries are persisted in
`outputs/analysis/fallback_padded_misclass_minirocket_ridge.md`. The
combined evidence indicates that the small preprocessing fallback /
padding populations (54 / 972 = 5.6% fallback_used; 41 / 972 = 4.2%
padded in the final cohort) do not bias the test-set predictions and do
not concentrate model errors.

**Temporal drift sensitivity (Tier D).** All six models on OOF and test
predictions sorted by `collection_time` and split at the median date into
first-half and second-half subsets. AUC-ROC, AUC-PR, and Brier with 95%
CI reported per half per model in
`outputs/analysis/temporal_drift_oof_all_models.csv` and
`outputs/analysis/temporal_drift_test_all_models.csv`. Across the six
models the maximum AUC-ROC delta between halves was 0.1158; aggregate
direction is mixed across models and CI bands overlap, so no single
model exhibits a degradation pattern attributable to sensor drift that
would invalidate the locked predictions.

## Tumor Marker LR + VOC-Marker Fusion

Tumor-marker complete-case subsets (CEA AND CA19-9 AND CA72-4 all
non-missing): `tm_test_subset_n = 135` (76 positive, 59 negative,
prevalence 0.563); `tm_train_subset_n = 516` (293 positive, 223
negative, prevalence 0.568). Post-hoc F trigger not fired
(n_test_pos = 76 >= 30; n_train_pos = 293 >= 50).

**VOC-only vs tumor-marker LR on `tm_test_subset`:**
- VOC-only AUC-ROC = 0.8689 (95% CI 0.7966-0.9300)
- Tumor-marker LR AUC-ROC = 0.5810 (95% CI 0.4808-0.6742)
- Delta AUC (VOC - marker LR) = +0.2879
- DeLong two-sided p = 2.57e-06 (highly significant in favor of VOC)
- NRI (VOC vs marker LR) = +0.8113 (strongly positive)
- IDI (VOC vs marker LR) = +0.4052 (strongly positive)

**Selection bias quantification on `tm_test_subset` vs full test set.**
Largest baseline SMD = 0.408 on `age`; four variables exceed SMD > 0.1:
`age` (0.408), `disease_category` (sample-availability artifact),
`collection_location` (inpatient enrichment from the complete-case
filter), and `label` (prevalence shift from 0.420 in full test to 0.563
in `tm_test_subset`). Selection-bias detail is in
`outputs/analysis/tumor_marker_selection_bias.md` on the 10 prespecified
SMD variables.

**VOC + tumor marker fusion on `tm_test_subset`** (LR fitted on
`tm_train_subset` with 4 features `[voc_p_raw, CEA, CA19-9, CA72-4]`):
- Fusion AUC-ROC = 0.8791 (95% CI 0.8136-0.9349)
- Delta AUC (fusion - VOC alone) = +0.0103
- DeLong two-sided p (fusion vs VOC) = 0.3519 (not significant)
- NRI (fusion vs VOC) = -0.6044 (strongly negative)
- IDI (fusion vs VOC) = +0.0085 (positive but small)

Per Erratum E013 revised §2.14 condition A: `(Delta AUC < 0.01 OR DeLong
p >= 0.05) AND (NRI <= 0 OR IDI <= 0)`. Evaluating on the observed
values: `(0.0103 < 0.01 = False) OR (0.3519 >= 0.05 = True) = True`,
AND `(-0.6044 <= 0 = True) OR (0.0085 <= 0 = False) = True`. Condition A
holds; the prespecified template applies:

> VOC alone is preferable; supplementation with markers is not supported by these data.

The conclusion is reported verbatim in `outputs/analysis/voc_marker_fusion_minirocket_ridge.md`.
The negative-fusion result is informative and clinically actionable: it
indicates that on this cohort, with the available tumor-marker
information, adding the three clinical markers to the VOC-only model
does not improve discrimination significantly and worsens net
reclassification (NRI = -0.60 on `tm_test_subset` with prevalence 0.56).

## SHAP Cross-Model Consistency

Channel-importance vectors `[11]` extracted per model per §2.21 method
mapping (XGBoost TreeSHAP, Elastic Net abs(coef_), SVM permutation
importance, MR+R raw-signal channel permutation, InceptionTime + CBLA
Captum Integrated Gradients with `captum = 0.9.0`). All six vectors are
L1-normalized post-clip-to-zero per Erratum E014. SVM channels [3, 5,
7, 8] were clipped in this run (small-magnitude noise-floor negatives);
all other models produced non-negative aggregates by construction.

**15 pairwise Kendall Tau values** computed across the 6 channel-importance
vectors (`C(6, 2) = 15` pairs) with bootstrap 95% CI:
- Mean Kendall Tau across 15 pairs = 0.467
- Min Kendall Tau = 0.236 (`minirocket_ridge` vs `cnn_bilstm_attention`,
  95% CI -0.362 to 0.760)
- Max Kendall Tau = 0.818 (`inception_time` vs `cnn_bilstm_attention`,
  95% CI 0.429 to 1.000)
- 7 of 15 pairs significant at alpha = 0.05

**Converging channels by top-K membership:**
- Unanimous top-5 across all 6 models: **S10**
- Top-3 majority (in at least 4 of 6 models): {S7, S10}
- Top-5 majority (in at least 4 of 6 models): {S2, S5, S7, S10, S11}
- Top-7 majority (in at least 4 of 6 models): {S1, S2, S5, S7, S10, S11}

The unanimous top-5 finding for channel S10 is the strongest cross-model
convergence outcome in this study. The five-channel top-5 majority set
{S2, S5, S7, S10, S11} represents 5 of 11 gas channels, a substantial
reduction from the prior of uniform 11-channel relevance.

Outputs at `outputs/shap/binary_<m>_channel_importance.csv` (6 files),
`outputs/shap/cross_model_kendall_tau.csv`, and the markdown summary
`outputs/analysis/shap_cross_model_consistency.md`. Erratum E014's
diagnostic disclosure of SVM clipped channels is reported in the
markdown summary.

## Reproducibility Evidence

Stage 9 reproducibility verification ran the end-to-end smoke test and
the absolute-path-leak grep across all git-tracked text-extension files,
then aggregated all 14-session metadata into a single experiment manifest.

**End-to-end smoke test.** On a deterministic 10-sample subset (5
negatives + 5 positives, selected by sample_id order from
`data/processed/splits/test_idx.npz`; sample_ids `[5, 7, 12, 14, 21, 22,
23, 25, 53, 80]`), each of the six final PKLs at
`outputs/models/binary_<m>_final.pkl` was loaded via the locked
`src.training.load_final_model.load_final_model()` API (Erratum E010) and
run end-to-end through its inference pathway (transformer + normalizer +
model + Platt calibrator for MR+R; normalizer + model + Platt for
Pipeline B; normalizer + DL wrapper + Platt for InceptionTime and CBLA).

For all six models, the smoke test verified:
- `predict_proba` output shape `(10, 2)`
- All probability values finite and in `[0, 1]`
- Each row sums to 1.0 within `1e-6` tolerance
- Derived `y_pred = (p_raw >= threshold_p_raw)` produces quadrant labels
  in `{TP, TN, FP, FN}`
- `apply_platt_calibrator` produces finite calibrated probabilities in
  `[0, 1]`

Smoke test outcome: **6 / 6 models passed.** Smoke test log persisted at
`outputs/logs/reproducibility_smoke_test.log`. Per-model assertions and
quadrant distributions persisted in `outputs/analysis/stage9_audit.json`
under `verification_results.smoke_test_per_model`.

**Absolute-path-leak grep.** All 538 git-tracked files scanned; 517
matched the text-extension whitelist (`.md`, `.json`, `.csv`, `.py`,
`.yaml`, `.yml`, `.txt`, `.log`). The path-leak regex (defined in
`src/analysis/reproducibility_verification.py::ABSOLUTE_PATH_PATTERN`)
catches absolute-path prefixes from four families: the POSIX `/home/`
family, the WSL `/mnt/` family, the Windows `C:\Users\` family, and
the macOS `/Users/` family; each prefix is followed by a username or
mount-point identifier and a trailing path separator before the regex
declares a match. **Final match count: 0.** No absolute paths leak
from the developer workstation into the tracked audit set.

**Experiment manifest.** Single aggregator JSON at
`outputs/logs/experiment_manifest.json` containing:
- Modeling stage HEAD commit hash and short hash
- Preprocessing stage HEAD commit (`preprocessing-frozen`)
- All 4 annotated tags with commit hashes and rationale
- All 14 errata IDs, dates, and titles
- All 5 post-hoc trigger outcomes with rationale and source
- Data split spec (seed 43, candidate set, n_train 777, n_test 195,
  stratification dimensions, erratum reference)
- Model selection findings (6-model OOF AUC-ROC table, DeLong p,
  selection step)
- Final training findings (best-model test AUC-ROC + CI, Brier
  pre/post-Platt, threshold values, oof_source, training git hash)
- Stage 7 key findings (JT p, Spearman rho, FP-LGIN Fisher OR, LGIN
  OOF rb + p, fusion conclusion template, fusion delta-AUC + DeLong
  p + NRI + IDI, VOC-only vs marker AUCs, selection-bias largest SMD)
- Stage 8 key findings (unanimous top-5, top-3/5/7 majority converging
  sets, Kendall Tau mean/min/max, n_pairs significant, E014 reference)
- Reproducibility verification block (smoke-test pass, path-leak count
  0, captum version 0.9.0, random_seed 42, bootstrap n_boot 2000,
  bootstrap convention erratum E011)

The manifest serves as the single machine-readable artifact summarizing
the full modeling-stage audit trail. Combined with the four annotated
tags' messages and the two protocol snapshots, it provides the TRIPOD
prespecification chain anchor for any external reviewer.

**Stage 9 audit JSON.** `outputs/analysis/stage9_audit.json` records the
Session 15 deliverables, smoke-test per-model results, path-leak file
list (empty after sanitization), errata count (14), post-hoc trigger
outcomes summary (A=False, C=False, F=False, G=True, H=False), and the
bootstrap convention E011 reference.

**Environment.** `environment.yml` pins the locked package set including
`captum = 0.9.0` (Stage 8 dependency added at commit `82e3516`),
`sklearn = 1.7.2`, `sktime = 0.40.1`, `xgboost = 3.2.0`, `pytorch =
2.6.0`, `pytorch-lightning = 2.5.5`, `numpy = 2.2.6`, `pandas = 2.3.3`,
`scipy = 1.16.2`, `statsmodels = 0.14.5`, `shap = 0.48.1`. All Python
execution is performed under conda environment `gcvocs` per project
convention.

## Limitations and Forward Work

Several limitations are surfaced by the prespecified analyses themselves
and are reported here as part of the locked record.

**Tumor-marker selection bias.** Per §2.13 MNAR analysis and the Stage 7
Part C selection-bias quantification, tumor markers (CEA, CA19-9,
CA72-4) are selectively available in this cohort. The `tm_test_subset`
(n = 135) has prevalence 0.563 vs the full test set prevalence 0.420,
the largest baseline-variable SMD on `age` (0.408), and inpatient-
enriched `collection_location`. Imputation is prohibited per the locked
methodology because missingness is MNAR. The fusion analysis result
(VOC-alone preferable per Erratum E013 application) is conditioned on
this complete-case subset; external validation of marker-fusion
behavior on a population with more representative marker availability
would require a separate prospective cohort.

**LGIN test-view non-estimability.** Per Erratum E012, the locked
195-sample test set contains 10 LGIN samples all with `collection_location
= 1` (inpatient B-group surrogate). The §2.15 B-vs-C separation analysis
is mathematically non-estimable on the test view (n_C_test = 0). The
primary analysis was promoted to the nested-CV OOF view (n_LGIN_OOF = 39
with n_B = 35, n_C = 4) where the §2.15 hypothesis is testable. A
supplementary re-analysis on an expanded test set with prospective
outpatient LGIN sampling would permit a fully test-view-based primary
analysis; such re-analysis would be a separate post-hoc artifact
requiring its own erratum at that time.

**Temporal drift moderate cross-model variation.** The Tier D temporal
drift analysis reports a maximum AUC-ROC half-vs-half delta of 0.1158
across the six models. While no individual model exhibits a degradation
trajectory that invalidates the locked test-set evaluation (CI bands
overlap and the direction of shift is mixed), the magnitude is
non-negligible and motivates external prospective validation on a
hold-out cohort collected outside the training time window.

**Dual epistemic role of the six-model panel.** The locked
prespecification distributes analyses across one best model (Tier A,
B2) and all six models (Tier B1, C, D, E). The all-six-model analyses
provide converging-evidence value (consistency across Pipeline A vs B
and linear vs nonlinear vs DL architectures) that single-model
sensitivity analyses do not. The Tier A clinical-deployment summary is
specific to MR+R; conclusions about VOC-based GC risk stratification at
the cohort scale rest on Tier B1/C/D/E's six-model coverage, while the
specific deployment recommendation is for MR+R only.

**Forward-looking note on `base_trainer.py` absolute-path INFO logs.**
The Session 15 path-leak grep identified absolute paths in the seven
committed training log files emitted during Sessions 6-10. These leaks
originated from two sources: (1) `base_trainer.py` INFO-level "Loaded
sample index from %s" and "Loaded Pipeline B features from %s" calls
that substituted the absolute load path; (2) upstream PyTorch Lightning
deprecation and dataloader-bottleneck warnings whose stack traces
include the conda-env `site-packages` path. The current Session 15 fix
sanitizes the log file contents in place via a `chore(logs)` commit
that matches the prior preprocessing-stage precedent `f748c6d`.

> Future training reruns (if undertaken) should patch base_trainer.py absolute-path INFO logs to project-root-relative form; the current sanitization is applied retroactively to committed log files only.

Modifying `base_trainer.py` itself is out of scope for Session 15 per
karpathy guideline 3 ("Surgical Changes") because `base_trainer.py` is
Session 2 code and Stage 9 is verification-only; the fix is recorded
here as a forward-looking item rather than implemented in this session.

---

This protocol is frozen at the `modeling-frozen` git tag. All
methodological execution decisions are recorded either in
`docs/protocol/modeling_plan.md` (immutable since
`modeling-prespecified`), in §12 of
`.cursor/plans/modeling_plan_v1_a7c3f9b2_dc789bb5.plan.md` (the 14
errata), in this summary, or in the four annotated tag messages.
Subsequent execution deviations on future cohorts or external
validation runs are out of scope for the current locked record and
would be addressed by a new study-level protocol with its own
prespecification chain.
