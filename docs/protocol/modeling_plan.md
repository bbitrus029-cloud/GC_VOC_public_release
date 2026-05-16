# Modeling Protocol Snapshot

This document is the TRIPOD prespecification snapshot for the modeling stage of
the gastric cancer VOC study. It is authored independently from the operational
working document, `.cursor/plans/modeling_plan_v1_a7c3f9b2_dc789bb5.plan.md`,
which remains the authoritative execution guide for implementation details,
session sequencing, and any future errata.

This snapshot is frozen at the `modeling-prespecified` git tag before any
modeling-stage implementation commit. It records the locked cohort facts,
prespecified configuration values, binding modeling decisions, prespecified
analyses, and sanctioned post-hoc trigger conditions that define the study's
methodological scope.

## 1. Provenance Discipline

Every methodological numeric value in this snapshot is assigned to one L17
provenance category at its first introduction. Later restatements of the same
value may omit a duplicate label when the provenance anchor has already been
established earlier in this snapshot.

The L17 categories used in the modeling stage are:

- `[domain-physical-constant]`: values inherited from Stage 1 artifacts,
  acquisition structure, or fixed repository contracts.
- `[clinical-literature]`: values anchored in clinical epidemiology,
  guideline-based cutpoints, or prespecified clinical interpretation rules.
- `[ml-methodology-literature]`: values anchored in machine-learning,
  statistics, calibration, or evaluation methodology.
- `[hyperparameter-range-from-domain-practice]`: search-space sizes or training
  settings grounded in model-family practice for comparable problems.
- `[runtime-determined-by-prespecified-algorithm]`: quantities computed by a
  deterministic algorithm specified before cohort-wide modeling.
- `[project-reproducibility-convention]`: repository-level conventions fixed
  before implementation.
- `[runtime-determined; not preregistered]`: operational estimates allowed only
  when they do not alter inferential claims.

## 2. Cohort And Stage 1 Facts

### 2.1 Final Modeling Cohort

The modeling stage starts from the Stage 1 outputs and does not revisit
preprocessing decisions. The final cohort contains `1040` rows
`[domain-physical-constant: Stage 1 artifact verified]` in
`data/processed/sample_index.csv`, with `12` columns
`[domain-physical-constant: Stage 1 artifact schema verified]`. The final
modeling subset contains `972` samples
`[domain-physical-constant: Stage 1 artifact verified]`.

The final disease-category distribution is HC=`158`, BGL=`352`, LGIN=`49`,
HGIN=`71`, EGC=`122`, and AGC=`220`
`[domain-physical-constant: Stage 1 artifact verified]`. The binary endpoint
distribution is label `0` = `559` and label `1` = `413`
`[domain-physical-constant: Stage 1 artifact verified]`. Class imbalance is
therefore `413:559`, approximately `1:1.35`
`[domain-physical-constant: computed from final label counts]`.

Preprocessing-pathway flags are retained for downstream sensitivity analyses.
`fallback_used=True` occurs in `60` samples overall and `54` samples in the
final cohort `[domain-physical-constant: Stage 1 artifact verified]`,
corresponding to `54 / 972 = 5.6%`
`[runtime-determined-by-prespecified-algorithm: reported as one-decimal percentage]`.
`padded=True` occurs in `41` samples overall and `41` samples in the final
cohort `[domain-physical-constant: Stage 1 artifact verified]`, corresponding
to `41 / 972 = 4.2%`
`[runtime-determined-by-prespecified-algorithm: reported as one-decimal percentage]`.

Collection-time reporting is locked to the Stage 0 sanity audit. The
modeling-stage source of truth is `sample_index.csv`, and the binding final
cohort range for temporal-drift analysis is taken from
`sample_index.csv[in_final_output==True]` rather than from raw metadata.

### 2.2 Pipeline A And Pipeline B Products

Pipeline A is stored at `data/processed/sequences/pipeline_a_raw.npz`. It must
contain the keys `x`, `sample_id`, `label`, `channel_names`, and
`clip_length_L` `[domain-physical-constant: Stage 1 artifact schema verified]`.
The tensor shape is `[972, 960, 11]`
`[domain-physical-constant: Stage 1 artifact verified]`. The clip length is
`L = 960` `[domain-physical-constant: finalized by Stage 1 preprocessing output]`,
and the retained gas-channel count is `11`
`[domain-physical-constant: hardware channel layout]`.

Pipeline B is stored at `data/processed/features/pipeline_b_raw.csv`. Its shape
is `[972, 101]`, interpreted as `99` engineered features plus `sample_id` and
`label` `[domain-physical-constant: Stage 1 artifact verified]`.

The modeling contract inherits the Stage 1 decision to exclude S0 from
modeling inputs and to treat S12 and S13 as preprocessing-only environmental
channels `[domain-physical-constant: Stage 1 protocol contract]`.

### 2.3 Data-Alignment Contract

Every modeling entry point that loads more than one of `sample_index.csv`,
`pipeline_a_raw.npz`, or `pipeline_b_raw.csv` must:

- filter `sample_index.csv` to `in_final_output == True`,
- sort the retained rows by `sample_id`,
- assert exact equality of ordered sample IDs across all loaded sources, and
- assert equality of aligned labels across the same ordered rows.

Any violation is treated as a programmer error and aborts execution.

## 3. Locked Methodological Configuration Values

This section records the methodological values that the operational plan anchors
through the modeling configuration schema. Labels are reproduced at first
introduction here, and later references follow the same first-introduction rule.

- `provenance_schema_version: 1` `[project-reproducibility-convention]`.
- `random_seed: 42` `[project-reproducibility-convention]`.
- `data_split_seed` is selected from the candidate set `{42, 43, 44, 45, 46}`
  `[project-reproducibility-convention]`.
- `split.train_fraction: 0.80` `[project-reproducibility-convention]`.
- `cv.outer_folds: 5`, `cv.inner_folds: 3`
  `[project-reproducibility-convention]`.
- `dl.holdout_fraction: 0.125`
  `[project-reproducibility-convention]`.
- `bootstrap.n_boot: 2000`
  `[ml-methodology-literature: percentile bootstrap precision convention]`.
- `calibration.method: sigmoid`; this primary calibration choice follows the
  locked rule that isotonic is not used when OOF calibration sample size is
  `n < 1000`
  `[ml-methodology-literature: Platt preferred for OOF n<1000]`.
- `age_cutpoint: 60`
  `[clinical-literature: gastric cancer incidence in China peaks in the 60-70 age range per GLOBOCAN and IARC reports; multiple Chinese gastric cancer epidemiology studies use age 60 as a risk-elevation stratification threshold; web-search-verify and cite primary sources during plan execution]`.
- `bmi_cutpoint: 24`
  `[clinical-literature: WS/T 428-2013 Chinese adult weight criterion; supported by WGOC Chinese cohort meta-analysis recommending BMI 24 for overweight and 28 for obesity; Zhou, Asia Pacific Journal of Clinical Nutrition, 2002]`.
- `prevalence_scenarios: 0.01`
  `[clinical-literature: scenario approximating low-prevalence population screening; Chen et al. Gut 2021, DOI 10.1136/gutjnl-2019-320200 supports large Chinese endoscopic screening context but does not itself define exactly 1%; REQUIRES-USER-VERIFICATION for exact scenario wording]`.
- `prevalence_scenarios: 0.05`
  `[clinical-literature: scenario approximating symptomatic/outpatient enrichment; BMC Gastroenterology 2021 uncomplicated dyspepsia cohort reported upper GI malignancy 1.5% and an optimal age cutoff of 56 years, DOI 10.1186/s12876-021-01951-x; exact 5% scenario remains REQUIRES-USER-VERIFICATION]`.
- `prevalence_scenarios: 0.10`
  `[clinical-literature: high-risk screening scenario; JMIR Public Health and Surveillance 2024 risk-adapted Chinese UGC screening supports high-risk enrichment and reports high-risk program structure, DOI 10.2196/62864; exact 10% scenario remains REQUIRES-USER-VERIFICATION]`.

## 4. Binding Modeling Architecture Decisions

### 4.1 Two-Document Prespecification Architecture

The modeling stage uses two documents with distinct roles. The operational plan
under `.cursor/plans/` governs execution details, while this protocol snapshot
governs methodology and sanctioned post-hoc conditions only. This snapshot is
authored independently rather than copied verbatim from the operational plan.

### 4.2 Model Lineup

The locked comparison set contains the following prespecified models:

- `minirocket_ridge`: Pipeline A, MiniRocket plus calibrated Ridge, with `1`
  fixed configuration
  `[hyperparameter-range-from-domain-practice: Dempster et al. MiniRocket; sktime/aeon practice uses fixed kernels plus ridge classifier]`.
- `inception_time`: Pipeline A deep learning model with `3` configurations
  `[hyperparameter-range-from-domain-practice: Fawaz et al. InceptionTime defaults and small search around depth/filters/dropout]`.
- `cnn_bilstm_attention`: Pipeline A deep learning model with `4`
  configurations
  `[hyperparameter-range-from-domain-practice: gas-sensor/e-nose deep architectures with CNN/LSTM/attention; exact ranges conservative due cohort size]`.
- `xgboost`: Pipeline B gradient boosting with `n_iter = 20`
  `[hyperparameter-range-from-domain-practice: e-nose gradient boosting literature plus computational budget]`.
- `svm`: Pipeline B RBF support-vector machine with `9` configurations
  `[hyperparameter-range-from-domain-practice: e-nose SVM grid search over C and gamma]`.
- `elastic_net_lr`: Pipeline B elastic-net logistic regression with `45`
  configurations
  `[hyperparameter-range-from-domain-practice: scikit-learn LogisticRegressionCV-style C and l1_ratio grids]`.

No additions, removals, or substitutions are allowed.

### 4.3 Seed Separation And Split Locking

`RANDOM_SEED` controls model initialization, resampling, bootstrapping, and any
other modeling randomness. `data_split_seed` is a separate quantity chosen from
the locked candidate set and then written into the modeling configuration before
split generation. These two seeds are never conflated, reassigned, or silently
derived from one another.

### 4.4 Fold-Level Normalization

Pipeline A normalization uses per-channel statistics with shape `[11]`
`[domain-physical-constant: gas-channel count]`. Pipeline B normalization uses
per-feature statistics with shape `[99]`
`[domain-physical-constant: feature count]`.

Normalization is fit on the current inner-train subset and then applied to the
matching validation subset. During final model training, normalization is fit
on the full `778`-sample training set
`[runtime-determined-by-prespecified-algorithm: 80% split of 972 with stratification]`
and then applied to the held-out test set.

If any standard deviation satisfies `sigma < 1e-8`, it is clamped to `1.0`
`[ml-methodology-literature: numerical stability convention for degenerate variance]`
and logged as a warning.

### 4.5 OOF Youden Thresholding And Platt Calibration

Final hyperparameters are aggregated from the outer-fold selections. Continuous
values use the median, categorical values use the mode, and integer-valued
quantities use the rounded median.

The OOF reuse rule is as follows. If every hyperparameter matches the
aggregated final setting across all outer folds, the existing nested-CV OOF
predictions are reused. Otherwise the model is rerun once over the locked outer
folds using the aggregated final hyperparameters to produce final-hyperparameter
OOF predictions. In both paths, the threshold and calibrator are fit only from
OOF predictions, never from final-model in-sample predictions.

Youden thresholding is fit in raw-probability space by maximizing `TPR - FPR`
`[ml-methodology-literature: Youden index]`.

Platt calibration is fit as logistic regression on `(p_raw, y_true)`
`[ml-methodology-literature: Platt/scikit-learn calibration; isotonic avoided when calibration n << 1000]`.
The design explicitly prohibits isotonic regression as the primary prespecified
calibration method.

The OOF provenance clarification is binding: OOF predictions come from the
outer-fold models, each trained on approximately `622` samples in the outer
training split, not from the final model refit on the full `778` training
samples. The in-sample training predictions saved later are diagnostic only and
are never used to fit thresholds, calibrators, or selectors.

### 4.6 MiniRocket Plus Ridge Double-Layer Calibration

MiniRocket plus Ridge uses an intrinsic Layer A probability mechanism through
`CalibratedClassifierCV(..., method='sigmoid', cv=5, ensemble=True)`
`[ml-methodology-literature: calibrated linear classifier probability mechanism]`.

Layer B is the shared OOF Platt post-processing step applied to all `6` models
`[project-reproducibility-convention: uniform calibration reporting]`. Youden
thresholding remains defined in raw-probability space, and
`threshold_p_final_equivalent` is the Platt-transformed counterpart saved for
clinical interpretation.

Both layers are monotonic sigmoid transforms. Therefore Platt calibration does
not change AUC-ROC or AUC-PR; it may change Brier score, calibration slope, and
calibration intercept. Threshold-dependent metrics are computed once in raw
space using `threshold_p_raw`.

### 4.7 Class-Imbalance Handling

Class weights are always computed on the current inner-train subset:

- XGBoost uses `scale_pos_weight = n_neg / n_pos`
  `[ml-methodology-literature: class-imbalance weighting for gradient boosting]`.
- Linear models and SVMs use `class_weight='balanced'`
  `[ml-methodology-literature: inverse-frequency class weighting]`.
- Deep learning uses
  `cw = [n_total / (2 * n_neg), n_total / (2 * n_pos)]`
  `[ml-methodology-literature: inverse-frequency mean-normalized class weights]`.

Global-cohort weighting is prohibited.

### 4.8 Prespecified Model Selection

Model selection is locked before any test-set prediction is computed.

Step 1 uses nested-CV AUC-ROC `95%` confidence-interval separation
`[ml-methodology-literature: interval-based uncertainty screening]`. If a
single candidate's lower bound exceeds every other candidate's upper bound, the
model is selected immediately.

Step 2 uses paired DeLong testing with
`0.05 / number_of_pairs`
`[ml-methodology-literature: DeLong 1988; Bonferroni family-wise control]`.

Step 3 uses paired AUC-PR bootstrap with `n_boot = 2000`
`[ml-methodology-literature: percentile bootstrap for paired performance differences]`.

Step 4 applies the prespecified parsimony order if earlier steps still tie.

The CV-only halt conditions are binding:

- Any model whose nested-CV AUC-ROC confidence interval contains `0.5`
  `[ml-methodology-literature: AUC-ROC = 0.5 is the no-skill random-classifier baseline by definition; this halt condition flags models whose 95% CI fails to exclude no-skill performance]`
  triggers a halt.
- Any OOF metric that is NaN or Inf triggers a halt.
- If all models have mean nested-CV AUC-ROC `< 0.6`
  `[project-reproducibility-convention: conservative cohort-wide signal-viability threshold; AUC-ROC 0.5 represents random performance and 0.6 is the project-level minimum below which modeling decisions are deferred for user inspection]`,
  modeling halts for user review.
- A p-value alone is never a halt condition.

Test-vs-CV shift is explicitly excluded from model-selection halting and is
handled only through Post-hoc C after best-model locking and test evaluation.

### 4.9 Five-Tier Analysis Hierarchy

The hierarchy of downstream claims is fixed:

- Tier A: best-model-only clinical translation, including DCA, two-tier risk
  stratification, four-quadrant descriptive analysis, representative samples,
  tumor-marker comparison, fusion, and the LGIN exploratory layer.
- Tier B1: all-model full-set reporting across the in-sample, OOF, and test
  views, with bootstrap confidence intervals.
- Tier B2: best-model subgroup reporting for the full set, HGIN+EGC subgroup,
  and AGC subgroup.
- Tier C: all-model subgroup reporting limited to threshold-free metrics.
- Tier D: all-model sensitivity analyses covering preprocessing-pathway flags,
  temporal drift, and robustness by the locked clinical covariates.
- Tier E: all-model channel-importance comparison and cross-model consistency.

This hierarchy separates clinical-deployment interpretation from
all-model converging-evidence analyses and from robustness checks.

### 4.10 Subgroup Immutability

The shared negative pool is fixed to disease categories `{0, 1, 2}`
`[domain-physical-constant: HC+BGL+LGIN category codes]`. The HGIN+EGC subgroup
adds positive categories `{3, 4}`
`[domain-physical-constant: category codes]`, and the AGC subgroup adds
category `{5}` `[domain-physical-constant: category code]`.

Subgroup analyses filter samples and recompute metrics only. They never refit
the final model, normalizer, transformer, calibrator, `threshold_p_raw`, or
`threshold_p_final_equivalent`.

### 4.11 Subgroup Metric Coverage

Tier B2 reports the full subgroup metric set with bootstrap `2000` `95%`
confidence intervals
`[ml-methodology-literature: percentile bootstrap]`. The metric set includes
threshold-free discrimination, calibration, threshold-dependent performance,
and confusion-matrix counts. NNE is excluded from subgroup analysis.

### 4.12 Three-View Comparison

Three-view reporting compares the final-model in-sample training predictions,
outer-fold OOF predictions, and held-out test predictions. The one-out-of-five
training subset of approximately `622` samples is not itself a reporting view
`[runtime-determined-by-prespecified-algorithm: not a stable TRIPOD view]`.

Each model uses its own raw-space threshold. In-sample predictions carry an
explicit warning that they are an overfitting upper bound and are not used for
claims.

### 4.13 Tumor-Marker Complete-Case Rule

Tumor-marker analyses are evaluation-time complete-case analyses only.
`tm_test_subset` and `tm_train_subset` are defined by requiring CEA, CA19-9,
and CA72-4 to be jointly non-missing
`[runtime-determined-by-prespecified-algorithm]`.

Selection-bias quantification uses exactly `10` baseline variables
`[project-reproducibility-convention: prespecified selection-bias table]`.
Imputation is prohibited because tumor-marker missingness is treated as MNAR
`[ml-methodology-literature: Little & Rubin 2019 missing-data framework]`.

### 4.14 Fusion Conclusion Rule

VOC-plus-marker fusion uses logistic regression over the `4`-feature vector
`[domain-physical-constant: fusion feature contract]`
`[voc_p_raw, CEA, CA19-9, CA72-4]`.

If `Delta AUC <= 0.01` and either `NRI < 0` or `IDI < 0`
`[0.01: project-reproducibility-convention — minimal clinically meaningful AUC-ROC improvement threshold below which fusion is not declared beneficial regardless of statistical significance; NRI<0 / IDI<0: ml-methodology-literature — Pencina sign convention where negative values indicate net reclassification or discrimination worsening relative to the baseline model]`,
the required conclusion is: "VOC alone is preferable; supplementation with
markers is not supported by these data." Otherwise the report uses the positive
fusion template with Delta AUC, p-value, NRI, and IDI.

### 4.15 LGIN Exploratory Layer Analysis

The LGIN layer is prespecified rather than post-hoc. It asks whether best-model
calibrated probabilities stratify LGIN samples beyond biopsy resolution.

The within-layer baseline comparison uses exactly `10` variables
`[project-reproducibility-convention]`. Within-B stability uses age binned at
`60` and BMI binned at `24`, together with the other locked covariates, for
exactly `7` variables
`[clinical-literature and project-reproducibility-convention; age 60 per §0 web verification ledger]`.

Each B-group level requires at least `5` samples
`[project-reproducibility-convention: small-cell descriptive threshold]`.
Every gating-passed level must have median `p_cal > 0.5`
`[runtime-determined-by-prespecified-algorithm: positive average classification criterion]`.
The primary alpha is `0.05`
`[ml-methodology-literature: conventional single-hypothesis alpha]`. Separation
is declared only if absolute rank-biserial correlation is `>= 0.5` and
Mann-Whitney `p < 0.05`
`[ml-methodology-literature: Cohen large-effect convention plus single-test alpha]`.

### 4.16 Fallback/Padded Sensitivity

Sensitivity analysis compares the full test set against the `fallback_used=False`,
`padded=False`, and doubly restricted subsets. Best-model misclassification
association uses `2` independent `2 x 2` Fisher exact tests
`[ml-methodology-literature: exact test for small contingency tables]`.

Interpretation thresholds are prespecified as follows:
`OR > 2 / OR < 0.5`
`[ml-methodology-literature: conventional doubling/halving odds-ratio convention for substantively meaningful binary association in 2×2 contingency analysis; p thresholds inherit the conventional 0.05 alpha labeled at §2.15]`.
These rules define associated misclassification, counterintuitive better
classification, or no significant association.

### 4.17 Temporal Drift Sensitivity

Temporal drift sorts predictions by collection time and splits each set into the
first `50%` and second `50%`
`[runtime-determined-by-prespecified-algorithm]`. No refitting is allowed.

### 4.18 Stratified Robustness

Robustness is evaluated across the locked clinical strata, including age binned
at `60`, BMI binned at `24`, and the remaining predefined covariates. A stratum
is reported only if `n >= 10`, `n_pos >= 2`, and `n_neg >= 2`
`[project-reproducibility-convention: minimum event/non-event support]`.
Failed strata remain in the output with omission reasons.

### 4.19 Disease Progression Trajectory

The best-model trajectory analysis preserves all `6` disease categories
`[domain-physical-constant: category count]`. The Jonckheere-Terpstra anchor is
`J=71`, `z=2.148`, `p=0.0159`
`[ml-methodology-literature: Altunkaynak & Gamgam R Journal 2020; anchor requires local fixture]`.

Adjacent-group testing is fixed to `5` ordered pairs
`[domain-physical-constant: ordered six-category trajectory]`, with
Holm-Bonferroni correction over the same `5` tests
`[ml-methodology-literature: Holm 1979]`.

### 4.20 Misclassification Analysis

Layer 1 augments every model's test predictions with `y_pred` and quadrant
labels. Layer 2 is best-model-only and includes descriptive quadrant summaries,
per-sample records, and representative-sample selection with up to `5` samples
per quadrant `[project-reproducibility-convention]`. False-negative detail is
always reported.

### 4.21 SHAP And Cross-Model Consistency

Model-specific explainability methods are fixed by model family:

- TreeSHAP for XGBoost, aggregated from feature space to channels.
- Absolute coefficients for elastic-net logistic regression.
- Permutation importance for SVM and MiniRocket plus Ridge.
- Integrated Gradients for the deep-learning models, using input geometry
  `[N, 11, 960]`.

The expected InceptionTime IG runtime is `20-30` minutes
`[runtime-determined; not preregistered: wall-clock estimate only]`.
Cross-model consistency computes `15` pairwise Kendall Tau values with bootstrap
`95%` confidence intervals
`[runtime-determined-by-prespecified-algorithm]`, and core converging channels
are summarized for `K in {3, 5, 7}`
`[project-reproducibility-convention]`.

### 4.22 Final Artifact Schema Contract

Every `outputs/models/binary_<m>_final.pkl` must contain the following top-level
keys:

- `transformer`
- `normalizer`
- `model`
- `calibrator`
- `threshold_p_raw`
- `threshold_p_final_equivalent`
- `feature_names`
- `training_sample_ids`
- `metadata`

The `metadata` dictionary must contain `model_name`, `final_hparams`,
`oof_source`, `git_commit_hash`, and `training_timestamp_utc`. Pipeline A
artifacts must retain the fitted transformer object rather than `None`. An
end-to-end smoke test blocks any checkpoint that touches final model artifacts.

### 4.23 Deep-Learning Training Discipline

Deep-learning runs are deterministic under the locked seed and backend settings.
Mixed precision is `'16-mixed'`
`[hyperparameter-range-from-domain-practice: GPU efficiency setting]`. A
CSVLogger is mandatory, and per-epoch train loss, validation loss, validation
AUC, and validation Brier score must be persisted.

The NaN-loss rule is fixed to one halving of the learning rate and one retry
`[project-reproducibility-convention: NaN-loss recovery budget — single retry with halved learning rate prevents silent retry loops while permitting one numerical-stability recovery before fold failure is recorded]`.
If the retry still fails, the fold is marked failed, logged at ERROR, and
excluded from aggregates. The DL early-stopping holdout fraction is `12.5%`
`[project-reproducibility-convention: early-stopping holdout size]`.

### 4.24 Hyperparameter Budget Guard

Any hyperparameter combination exceeding `5` minutes per fold triggers a
warning
`[runtime-determined; not preregistered: operational stall diagnostic]`.
XGBoost is locked to `tree_method='exact'` and `n_jobs=1`
`[project-reproducibility-convention: locked anti-stall setting]`.

### 4.25 Ground-Truth Anchor Tests

The DeLong implementation is anchored to the aSAH fixture with `n=113`,
s100b AUC `0.7314`, ndka AUC `0.612`, DeLong `p=0.1643`, and `Z=-1.3908`
`[ml-methodology-literature: Robin et al. pROC 2011; pROC package anchor values require fixture verification]`.

The Jonckheere-Terpstra implementation is anchored to the npordtests fixture
with `J=71`, `z=2.148`, and `p=0.0159`
`[ml-methodology-literature: Altunkaynak & Gamgam 2020; REQUIRES-USER-VERIFICATION of exact jdata values if not present in package docs]`.

Both anchors use local fixtures only.

## 5. Prespecified Analyses

This section records methodology only: inputs, algorithmic intent, and
persisted outputs. Operational sequencing is intentionally omitted.

### 5.1 Repository Configuration Update

Inputs are `.gitignore`, `.cursorignore`, `.gitattributes`, `environment.yml`,
`README.md`, and the project rules. The method is to apply the modeling audit
allowlist, verify indexing and dependency state, and mark the repository as
being in the modeling stage. Outputs are the repository-configuration diff and
the Stage 0 review record.

### 5.2 Stage 1 Product Completeness Verification

Inputs are the Stage 1 artifacts, Stage 1 logs, and preprocessing protocol
anchors. The method is the locked Stage 1 sanity audit plus package
importability/version verification. Output is `outputs/logs/env_sanity.log`.

### 5.3 Common Utility Modules

Inputs are the locked Stage 1 artifacts and future split files. The method is
to implement common fold-generation, normalization, metric, bootstrap, trainer,
and early-stopping helpers. Outputs are the training-common modules and their
tests.

### 5.4 Evaluation Modules

Inputs are prediction tables and binary labels. The method is to implement
metrics, calibration, DeLong testing, Jonckheere-Terpstra testing, and the
associated anchor tests. Outputs are the evaluation modules and the anchor-test
suite.

### 5.5 Deep-Learning Architecture Modules

Inputs are synthetic Pipeline A tensors shaped `[B, 11, 960]`
`[domain-physical-constant: model input geometry]`. The method is to implement
the InceptionTime model, the CNN-BiLSTM-Attention model, and the Lightning
data module. Outputs are the DL model modules and the synthetic-shape tests.

### 5.6 Candidate Seed Evaluation

Inputs are final-cohort labels, disease categories, clinical covariates, and
preprocessing-pathway flags. The method is to compare the locked candidate
split seeds using the prespecified imbalance criteria and to select one seed
before split generation. Output is `outputs/analysis/data_split_seed_selection.md`.

### 5.7 Split And Fold Generation

Inputs are the locked final cohort and the user-confirmed `data_split_seed`.
The method is to create the stratified train/test split, outer CV folds, inner
CV folds, and DL holdout files. Outputs are the split `.npz` files under
`data/processed/splits/`.

### 5.8 Table 1

Inputs are the sample index joined to aggregate metadata columns. The method is
to summarize training-vs-test and negative-vs-positive baseline characteristics
with standardized mean differences. Outputs are the markdown and CSV Table 1
artifacts.

### 5.9 Classical-Model Nested CV

Inputs are Pipeline A for MiniRocket plus Ridge, Pipeline B for the classical
tabular models, and the locked split files. The method is nested CV with
fold-level normalization, inner-train class weighting, hyperparameter scoring,
and OOF persistence. Outputs are per-model OOF, per-fold, summary, and JSON
result files.

### 5.10 Deep-Learning Nested CV

Inputs are the Pipeline A tensor and locked split files. The method is DL
nested CV with mixed precision, deterministic training, locked early-stopping
discipline, CSV logging, and NaN-loss retry handling. Outputs are the nested-CV
prediction files, checkpoints, and training logs.

### 5.11 Model Selection

Inputs are the nested-CV OOF predictions and summaries for all candidate
models. The method is the three-step locked selection algorithm plus its CV-only
halt checks. Output is `outputs/analysis/model_selection_report.md`.

### 5.12 Final Training And Test Evaluation

Inputs are the locked train/test split, aggregated final hyperparameters, and
the OOF reuse-or-rerun decision. The method is to refit final models, fit raw
Youden thresholds and Platt calibrators from OOF data, generate in-sample and
test predictions, compute metrics, and save final artifacts. Outputs include
final models, prediction CSVs, metrics CSVs, threshold/calibrator artifacts,
and decision JSONs.

### 5.13 Decision Curve Analysis

Inputs are the best-model test predictions and the locked prevalence scenarios.
The method is DCA over the prespecified threshold grid with treat-all and
treat-none comparators. Outputs are the DCA CSV and markdown files.

### 5.14 Two-Tier Risk Stratification

Inputs are best-model test predictions, the fixed raw threshold, and the locked
prevalence scenarios. The method is to classify low vs high risk and compute
prevalence-adjusted PPV, NPV, and NNE. Outputs are the risk-stratification CSV
and markdown files.

### 5.15 Tier B1 Three-View Comparison

Inputs are in-sample, OOF, and test predictions for every model. The method is
to compute the full metric set for each model-view combination with bootstrap
confidence intervals. Output is
`outputs/analysis/three_view_all_models_full_set.csv`.

### 5.16 HGIN+EGC Subgroup Analysis

Inputs are predictions and disease-category labels. The method is to run the
best-model full-metric subgroup report and the all-model threshold-free subgroup
report for the HGIN+EGC subgroup. Outputs are the subgroup CSV artifacts.

### 5.17 AGC Subgroup Analysis

Inputs are predictions and disease-category labels. The method is the same as
for the HGIN+EGC subgroup, but applied to AGC. Outputs are the AGC subgroup
artifacts.

### 5.18 Stratified Robustness

Inputs are test predictions plus the locked clinical covariates. The method is
to compute gated stratum-specific metrics without refitting any model component.
Outputs are the stratified-robustness CSV files.

### 5.19 Disease Progression Trajectory

Inputs are best-model calibrated test predictions and disease-category order.
The method is Jonckheere-Terpstra testing, Spearman correlation, and adjacent
Mann-Whitney testing with Holm correction. Outputs are the progression-trajectory
markdown and CSV files.

### 5.20 Misclassification Analysis

Inputs are best-model test predictions, the raw signal tensor, and SHAP-related
artifacts. The method is to derive quadrant labels, summarize quadrant-specific
clinical patterns, persist per-sample records, and select representative
examples. Outputs are the quadrant markdown report, per-sample CSV, and
representative-sample CSV/NPZ artifacts.

### 5.21 Tumor-Marker Logistic-Regression Comparison

Inputs are the complete-case train/test subsets and VOC predictions. The method
is to fit tumor-marker-only logistic regression, evaluate it on the same
complete-case test subset used by VOC comparisons, compare models with DeLong,
NRI, IDI, and DCA overlay, and quantify selection bias. Outputs are the tumor
marker comparison and selection-bias reports.

### 5.22 VOC Plus Marker Fusion

Inputs are VOC raw probabilities and complete-case tumor-marker values. The
method is to fit the locked four-feature logistic-regression fusion model and to
interpret results using the locked positive-vs-negative conclusion template.
Outputs are the fusion markdown and CSV files.

### 5.23 LGIN Exploratory Layer

Inputs are LGIN samples, collection location, best-model calibrated
probabilities, and baseline covariates. The method is the locked B-vs-C
comparison, baseline SMD description, confounding-by-indication rebuttal, and
within-B stability analysis. Output is `outputs/analysis/lgin_layer_analysis.md`.

### 5.24 Fallback/Padded Sensitivity

Inputs are all-model test predictions and the Stage 1 pathway flags. The method
is subset-specific metric recomputation without refitting, plus best-model Fisher
testing for misclassification association. Outputs are the all-model sensitivity
CSV and best-model markdown summary.

### 5.25 Temporal Drift Sensitivity

Inputs are OOF and test predictions plus collection times. The method is
first-half vs second-half metric comparison by collection date order, with no
refitting. Outputs are the temporal-drift CSV reports for OOF and test data.

### 5.26 SHAP Cross-Model Consistency

Inputs are final models, test data, feature names, and raw signals. The method
is to derive channel-level importance vectors by model family, compare them with
pairwise Kendall Tau, and summarize converging channels. Outputs are the
channel-importance CSVs, the Kendall Tau matrix, and the markdown summary.

### 5.27 End-To-End Smoke Test

Inputs are final model artifacts and a small representative subset of Stage 1
inputs. The method is deterministic end-to-end inference with probability range
checks and artifact-schema checks. Output is the smoke-test pass record.

### 5.28 Experiment Manifest

Inputs are the full set of modeling outputs and logs. The method is to
assemble a relative-path-only manifest containing seeds, timestamps, selected
model identity, key metrics, post-hoc trigger states, and traceability
metadata. Output is `outputs/logs/experiment_manifest.json`.

## 6. Sanctioned Post-Hoc Trigger Conditions

No post-hoc analyses other than A, C, F, G, and H are permitted.

### 6.1 Post-Hoc A: Subgroup Retrain

This trigger fires only when both subgroup degradation conditions are met:

- subgroup AUC-ROC point estimate is more than `0.05` below the full-set point
  estimate
  `[project-reproducibility-convention: substantive AUC degradation threshold]`,
  and
- subgroup AUC-ROC `95%` upper confidence bound remains below the full-set
  point estimate
  `[ml-methodology-literature: interval-based evidence of subgroup degradation that distinguishes a statistically real drop from sampling fluctuation]`.

If triggered, the full locked model-comparison pipeline is rerun on the subgroup
positives plus the shared negative pool. The result is supplementary only and
never replaces the main full-cohort analysis.

### 6.2 Post-Hoc C: Test-Set Shift Attribution

This trigger inspects systematic divergence between nested-CV and test AUC-ROC.
It fires when all models shift in the same direction and the average shift is at
least `+0.02` or at most `-0.02`
`[project-reproducibility-convention: bidirectional test-vs-CV shift-inspection threshold; ±0.02 average AUC across all six models distinguishes systematic test-set easiness/hardness from sampling variation]`.

If the absolute average shift reaches `0.05`
`[project-reproducibility-convention: magnitude threshold above which Post-hoc C adds per-fold granular attribution; chosen to match the substantive-AUC-shift convention used in Post-hoc A condition 1]`,
the attribution expands from global explanation to per-fold detail.

### 6.3 Post-Hoc F: Tumor-Marker Underpowering

This trigger fires if complete-case tumor-marker positives fall below `30` in
test or below `50` in train
`[ml-methodology-literature: small event-count caution; REQUIRES-USER-VERIFICATION as claim threshold]`.
If triggered, DeLong, NRI, and IDI remain descriptive only.

### 6.4 Post-Hoc G: Deep-Learning Training Anomaly

This trigger fires if the same DL model has at least `3` outer folds with early
stopping epoch `< 20`, or if any fold still has NaN loss after the locked retry
rule
`[project-reproducibility-convention: anomaly flag]`.
If triggered, the report may recommend future search-space adjustment but does
not authorize one inside the primary study.

### 6.5 Post-Hoc H: Calibration Failure

This trigger fires if post-Platt Brier score exceeds pre-Platt Brier score by
more than `0.005`
`[project-reproducibility-convention: calibration degradation tolerance]`.
If triggered, isotonic regression may be explored only as a supplementary
analysis.

## 7. References

- Altunkaynak, B., and Gamgam, H. "npordtests: An R Package of Nonparametric
  Tests for Equality of Location Against Ordered Alternatives." The R Journal,
  2020.
- Chen, R., Liu, Y., Song, G., et al. "Effectiveness of one-time endoscopic
  screening programme in prevention of upper gastrointestinal cancer in China."
  Gut, 2021;70:251-260. DOI: 10.1136/gutjnl-2019-320200.
- Cohen, J. Statistical Power Analysis for the Behavioral Sciences. 2nd ed.
  Lawrence Erlbaum Associates, 1988. ISBN 9780805802832.
- DeLong, E. R., DeLong, D. M., and Clarke-Pearson, D. L. "Comparing the Areas
  under Two or More Correlated Receiver Operating Characteristic Curves."
  Biometrics, 1988;44:837-845. DOI: 10.2307/2531595. PMID: 3203132.
- Dempster, A., Schmidt, D. F., and Webb, G. I. "MiniRocket: A Very Fast
  (Almost) Deterministic Transform for Time Series Classification." KDD, 2021.
  DOI: 10.1145/3447548.3467231.
- Fawaz, H. I., Lucas, B., Forestier, G., et al. "InceptionTime: Finding
  AlexNet for Time Series Classification." Data Mining and Knowledge Discovery,
  2020. DOI: 10.1007/s10618-020-00710-y.
- Hanley, J. A., and McNeil, B. J. "The Meaning and Use of the Area under a
  Receiver Operating Characteristic Curve." Radiology, 1982;143:29-36.
  DOI: 10.1148/radiology.143.1.7063747. PMID: 7063747.
- Holm, S. "A Simple Sequentially Rejective Multiple Test Procedure."
  Scandinavian Journal of Statistics, 1979;6:65-70. DOI: 10.2307/4615733.
- Jain, S., and Wallace, B. C. "Attention is not Explanation." NAACL, 2019.
  DOI: 10.18653/v1/N19-1357.
- Li, Z.-S., et al. "Chinese National Clinical Practice Guidelines on the
  Prevention, Diagnosis, and Treatment of Early Gastric Cancer." Chinese
  Medical Journal, 2024;137:887-908. DOI: 10.1097/CM9.0000000000003101.
- Little, R. J. A., and Rubin, D. B. Statistical Analysis with Missing Data.
  3rd ed. Wiley, 2019.
- National Health Commission of the People's Republic of China. "National
  Guidelines for Diagnosis and Treatment of Gastric Cancer 2022 in China."
  Chinese Journal of Cancer Research, 2022;34:207-237.
  DOI: 10.21147/j.issn.1000-9604.2022.03.04. PMID: 35873885.
- O'Brien, R. M. "A Caution Regarding Rules of Thumb for Variance Inflation
  Factors." Quality & Quantity, 2007;41:673-690.
  DOI: 10.1007/s11135-006-9018-6.
- Otsu, N. "A Threshold Selection Method from Gray-Level Histograms." IEEE
  Transactions on Systems, Man, and Cybernetics, 1979;9:62-66.
  DOI: 10.1109/TSMC.1979.4310076.
- Pardo, M., and Sberveglieri, G. "Classification of Electronic Nose Data with
  Support Vector Machines." Sensors and Actuators B: Chemical, 2005;107:730-737.
  DOI: 10.1016/j.snb.2004.12.005.
- Pencina, M. J., D'Agostino, R. B. Sr., D'Agostino, R. B. Jr., and
  Vasan, R. S. "Evaluating the Added Predictive Ability of a New Marker."
  Statistics in Medicine, 2008;27:157-172. DOI: 10.1002/sim.2929.
- Pencina, M. J., D'Agostino, R. B. Sr., and Steyerberg, E. W. "Extensions of
  Net Reclassification Improvement Calculations." Statistics in Medicine,
  2011;30:11-21. DOI: 10.1002/sim.4085. PMID: 21204120.
- Robin, X., Turck, N., Hainard, A., et al. "pROC: An Open-Source Package for
  R and S+ to Analyze and Compare ROC Curves." BMC Bioinformatics, 2011;12:77.
  DOI: 10.1186/1471-2105-12-77. PMID: 21414208.
- Sundararajan, M., Taly, A., and Yan, Q. "Axiomatic Attribution for Deep
  Networks." ICML, 2017;PMLR 70:3319-3328.
  arXiv DOI: 10.48550/arXiv.1703.01365.
- Tukey, J. W. Exploratory Data Analysis. Addison-Wesley, 1977.
  ISBN 9780201076165.
- Vickers, A. J., and Elkin, E. B. "Decision Curve Analysis: A Novel Method
  for Evaluating Prediction Models." Medical Decision Making, 2006;26:565-574.
  DOI: 10.1177/0272989X06295361. PMID: 17099194.
- Wiegreffe, S., and Pinter, Y. "Attention is not not Explanation."
  EMNLP-IJCNLP, 2019. DOI: 10.18653/v1/D19-1002.
- Yan, J., Guo, X., Duan, S., Jia, P., Wang, L., Peng, C., and Zhang, S.
  "Electronic Nose Feature Extraction Methods: A Review." Sensors,
  2015;15:27804-27831. DOI: 10.3390/s151127804.
- Zhao, G., Xue, M., Hu, Y., et al. "How Commonly Is the Diagnosis of Gastric
  Low Grade Dysplasia Upgraded following Endoscopic Resection? A Meta-Analysis."
  PLOS ONE, 2015;10:e0132699. DOI: 10.1371/journal.pone.0132699.
- Zhou, B.-F. "Predictive Values of Body Mass Index and Waist Circumference for
  Risk Factors of Certain Related Diseases in Chinese Adults." Asia Pacific
  Journal of Clinical Nutrition, 2002;11:S685-S693.
- WS/T 428-2013. Criteria of Weight for Adults. National Health and Family
  Planning Commission of China, released 2013-04-18, implemented 2013-10-01.

This protocol is frozen at the `modeling-prespecified` git tag. Subsequent execution deviations are recorded as errata in `.cursor/plans/modeling_plan_v1_a7c3f9b2_dc789bb5.plan.md` §12, never by editing this snapshot.
