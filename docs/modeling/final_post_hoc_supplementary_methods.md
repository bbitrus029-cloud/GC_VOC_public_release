# Final Post-Hoc Robustness and Reporting-Closure Methods Snapshot

**Package status:** post-hoc supplementary  
**Package ID:** `posthoc_final_supplementary`  
**Current operational authority:** `docs/plans/acquisition_context_robustness_audit_97707a26.plan.md`
**Current operational-plan SHA-256:** `ea38735dce9383adeff839ca5c8bdcd084b5f065f5fd68eeda924cdaa522a452`
**Current agent-guidance authority:** `AGENTS.md` (`1cb134c06871ba1837e0ff235dca9ca6ee0c10c1d08a123ff3912354299f9c9a`)
**Baseline HEAD:** `cbb87b0380d7129798002ff4ea8b869c4b786582`  
**Machine-readable contract:** `configs/post_hoc_final_supplementary.yaml`  
**Record status:** current consolidated methods/provenance authority; the original Session 0 snapshot was entirely results-free.

## 1. Purpose and analysis category

This package freezes and later executes a bounded, aggregate-only final post-hoc robustness and reporting-closure exercise. It does not alter or replace the frozen primary endpoint, cohort, split, preprocessing, model selection, MiniRocket configuration, calibrators, thresholds, predictions, models, or annotated tags. It is conceived after the `modeling-frozen` temporal anchor and is therefore post-hoc supplementary, not prespecified primary analysis.

The package closes four reviewer-facing robustness questions and freezes aggregate reporting-closure sources needed for later manuscript and visualization work. Visualization v2 remains unauthorized through Session 7 and requires a later additive `outputs/analysis/posthoc_final_visualization_release.json` after Session 7 is committed and independently approved.

### 1.1 Administrative/runtime consolidation — 2026-07-20

Original result-free Session 0 was committed at `3c0ff6b3b40460a2c340ad3193222ee54979109d` with an exact ISO `%Y-%m-%d` parser assumption. Its historical approval provenance is the legacy path `.cursor/plans/acquisition_context_robustness_audit_97707a26.plan.md` at SHA-256 `c1fac4e6ec408f06a9be2efb9748c5ec48d43d981af0faf36bf305c5386b7ef6`; that path is not a current runtime dependency. The first score-blind preflight subsequently failed closed at modeling-cohort date parsing with the sanitized message `Date parsing failed: missing_or_nonstring_count=0, invalid_date_format_count=972, invalid_date_count=0.` That attempt did not access scores, compute score-derived results, fit a model, promote a preflight artifact, or execute Session 4 or any later session.

Pre-execution Amendment 01, committed at `2e08e8fb89d99bb62cca2e5204a13ff09e47d432`, approved only the complete replacement of `loaders.date_parsing` by the exact ASCII slash-date contract and recorded the failed attempt, aggregate format audit, access-state ledgers, retry gates, and required C01–C16 corrections. The amendment did not rewrite or backdate the original Session 0 blob. Session 1 implemented exactly C01–C16 and committed the successful score-blind preflight at `d19dcbdb7bd9ef20c8534cca0a87384ed1c6fa9f`; `outputs/analysis/posthoc_final_score_blind_preflight.json` remains preserved byte-for-byte at SHA-256 `2014bc72ba2c08952be1ad7d7327f07273605c98b3814c2f0ca1b9de4f10e3bf`.

This Commit B candidate consolidates the currently effective Session 0 plus Amendment 01 record into this methods file, the main configuration, and the main manifest, and relocates current local workspace authority to `docs/plans/` and root `AGENTS.md`. The original Session 0 and Amendment 01 commits, changed-path inventories, blobs, hashes, and approval records remain independently verifiable from Git history; the full historical Amendment documents are recoverable from commit `2e08e8fb89d99bb62cca2e5204a13ff09e47d432`. This is administrative/runtime consolidation only: it creates no new approval, freeze, Session 0, amendment, scientific execution, statistical change, or Session 2 authorization.

## 2. Interpretation boundaries

- The 195-participant test set is an internal same-center, same-device, same-period random holdout. It is not external or temporal validation.
- Breath samples were collected after diagnostic gastroscopy and indicated biopsy but before treatment. This study does not directly validate future pre-endoscopy triage.
- The endpoint is high-grade gastric lesion (Yes = HGIN + EGC + AGC; No = HC + BGL + LGIN). A model score is not a gastric-cancer probability.
- The package does not support population-screening performance claims, deployment readiness, causal effect, clinical benefit, or pathological progression claims.
- LGIN findings remain exploratory because biopsy can undergrade disease and complete resection pathology was unavailable. Differential reference standards must be disclosed.
- Every planned favorable, unfavorable, inconclusive, and non-estimable result must be retained. Result direction cannot trigger a new analysis, model, split, threshold, or reporting omission.

## 3. Four complementary analyses

### 3.1 Analysis 1 — difficult-spectrum locked-score sensitivity

Removes the easiest spectrum extremes (HC and AGC) while retaining the frozen MiniRocket Ridge `p_raw` score. Include HGIN (`disease_category=3`) and EGC (`4`) as positive and BGL (`1`) and LGIN (`2`) as negative. Internal test is primary; training out-of-fold (OOF) is supporting and is never pooled with test. Report sample size, class counts, six-category composition, AUROC with DeLong 95% CI, fixed-threshold confusion counts, Wilson sensitivity/specificity intervals, and likelihood ratios. No AUPRC, PPV, NPV, F1, calibration, DCA, threshold change, or formal delta from the full cohort is computed. This is a restricted spectrum analysis, not a new endpoint or progression model.

### 3.2 Analysis 2 — within-date conditional concordance

Conditions concordance on shared calendar date without changing scores. For each date with both endpoint classes, pair-weighted within-date concordance uses 0.5 credit for score ties. Training OOF is primary; internal test is supporting. Uncertainty uses a nonparametric comparable-date cluster bootstrap with independent `np.random.default_rng(42)` streams per view, exactly 2,000 draws when at least ten comparable dates exist, and linear quantiles. Sparse-group and zero-comparable-date rules mark CI or point non-estimability as specified in the configuration. No actual, hashed, chronological, or pseudonymous date labels are persisted.

### 3.3 Analysis 3 — fixed unshuffled date-grouped auxiliary MiniRocket validation

The sole training-set-only auxiliary fit. Population is the locked 777-member training set. Internal-test predictions are not an input; test indices are used only for a negative exclusion assertion. Split exactly once with installed scikit-learn 1.7.2 `StratifiedGroupKFold(n_splits=5, shuffle=False)`, `random_state=None`, missing `split_seed`, and `split_seed_status=not_applicable`. This avoids the installed `shuffle=True` group-index mapping defect. Pre-fit feasibility must pass before any fit. When feasible, call `MiniRocketRidgeTrainer.fit_fold()` exactly five outer times, leave Layer-A five-fold sigmoid calibration unchanged, use a minimal adapter, prohibit broad trainer loaders, discard fitted objects, and persist only aggregate diagnostics plus a canonical little-endian `<f8` OOF digest. Paired date-cluster bootstrap uses seed 42 with a 2,000-valid / 20,000-attempt contract. This is not prospective temporal or external validation.

### 3.4 Analysis 4 — strict historical QC5 locked-prediction sensitivity

Reconstruct historical any-channel QC5 status from the eleven lower-case `s1_saturation_fraction` through `s11_saturation_fraction` fields. A channel is saturated only when `saturation_fraction > 0.003` (equality passes). Strict pass is `n_saturated_channels == 0`; final E005 pass is `n_saturated_channels < 7`. Validate reconstruction against E005 for all 974 QC5-evaluable source rows and require the 298 strict-fail / 676 strict-pass aggregate reconciliation before joining the 972-member modeling cohort. Evaluate locked `p_raw` on strict-pass subsets (test primary; OOF supporting). Also report an unadjusted pass-minus-fail Newcombe accuracy contrast with outcome and disease composition. Empty strict-status groups retain all planned rows and count fields, including zeros, but leave estimates/proportions and applicable CI fields missing with `non_estimable` / `empty_strict_status_group`; no fallback values are invented. Exact historical 675-member identity cannot be proven even if aggregate counts reconcile; do not claim otherwise. Causal or channel-level follow-up is forbidden.

## 4. Locked scores, threshold, and single auxiliary fit

Analyses 1, 2, and 4 reuse persisted locked predictions. They do not refit the model, transformer, normalizer, calibrator, or threshold. Threshold-dependent metrics use full-precision raw threshold `0.3418938786147648` with comparison `p_raw >= threshold`. The calibrated-equivalent threshold `0.2195889732130093` is frozen lineage only and is not a second decision rule. Required wording: “Platt calibration is monotonic and does not change AUC-ROC or AUC-PR; it may change Brier score, calibration slope, and calibration intercept.”

Analysis 3 is the only authorized auxiliary fit and cannot use the internal test for design or fitting, retry a split for better results, tune a model, save a replacement model, or change the primary analysis.

## 5. Common statistical contracts

### 5.1 Single-AUROC DeLong confidence interval

Session 1 implements a result-independent single-AUROC wrapper that reuses `_fast_delong()` and `DEGENERATE_VARIANCE_EPSILON` from `src/evaluation/statistical_tests.py`. No standalone `src/evaluation/delong.py` is used. Labels must be exactly 0/1. Absent class makes point and CI non-estimable. When both classes exist, AUROC uses 0.5 tie credit and cross-checks the core within `1e-12`. CI requires at least two positives and two negatives; variance uses the scalar covariance diagonal; limits are clipped to `[0,1]`; degenerate variance marks only the CI non-estimable. No bootstrap or other fallback replaces a non-estimable DeLong CI.

### 5.2 Sensitivity, specificity, and likelihood ratios

Sensitivity and specificity use Wilson score intervals from installed statsmodels. Likelihood ratios preserve uncorrected confusion counts, sensitivity, and specificity. If either outcome class is absent, LRs are non-estimable and the 0.5 correction must not invent a class. If both classes exist and any confusion cell is zero, apply the all-four-cell Haldane–Anscombe correction for LR point estimates and CIs only. Exact LR+, LR−, log-SE, and exponential CI formulas are frozen in configuration. LR rows cover the frozen primary internal test and the new difficult-spectrum and strict-QC5 views only.

## 6. Privacy, loaders, dates, and access-state semantics

The package is aggregate-only. Participant identifiers, participant scores, date labels, fold assignments, and participant-level QC flags are not output artifacts. Exact collection dates may be processed only in memory for approved date analyses; persist aggregate counts and estimates only.

Per-purpose ordered `usecols` loaders are frozen in configuration. Loading a broader table is a hard failure. No loader requests `collection_location`. Analysis 2 and Analysis 3 accept only ASCII date-only strings matching `^([0-9]{4})/([0-9]{1,2})/([0-9]{1,2})$`; they parse the three integer tokens explicitly, validate the calendar exactly, normalize only in memory, and reject ISO text, whitespace, time content, Unicode digits, and normalization collisions. Errors report sanitized aggregate invalid/missing counts only. This effective parser is the exact Amendment 01 correction and is not attributed to the original Session 0 blob.

Every Session 0 or later manifest and handoff records `session_activity` and `cumulative_through_session` with the nine required boolean keys. Session 0 may hash opaque prediction containers and inspect static headers, but must not read participant-level label, score, date, or QC values; must not compute new score-derived results; must not fit the auxiliary model; and must not execute the privacy-reference loader. Legacy access flags are not emitted.

Privacy validation is schema-aware: policy mentions of field names such as `sample_id`, `filename`, `collection_time`, or `collection_location`, and allowlisted provenance timestamps, are not participant-output leaks. The isolated privacy-reference loader, when later authorized, never echoes matched values.

## 7. Deterministic execution and promotion controls

From Session 1 onward, real-data CLIs require an explicit full freeze-commit argument and validate parent, path set, and blob identity before participant-value reads or fitting. Candidates are generated in destination-local temporary locations, validated, and promoted individually by atomic replacement into absent targets. The Session 1 score-blind preflight creates no persisted completion log. Sessions 4–7 each promote a sanitized completion log last as a successful completion receipt only; a failed-closed execution produces no completion log. Each completion log hashes exactly its session output inventory except itself. The Session 4 locked-scores log inventories exactly four outputs—locked-score metrics, locked-score composition, QC5 correctness/composition, and within-date concordance—and must neither hash nor claim the Session 1 preflight as a Session 4 output. Unexpected or stale targets fail closed. Outputs present without the expected completion log constitute a partial interrupted promotion requiring user provenance review. Deterministic scientific-payload comparison rules and runtime/thread/job-control contracts are frozen in configuration before execution; a mismatch never authorizes a post-hoc scientific or runtime-contract change.

## 8. Reporting closure

### 8.1 Cohort flow and inclusion/exclusion

Create an aggregate 11-node cohort-flow hierarchy from the frozen sample index and split: assessed 1040; seven closed-set QC/clip exclusion nodes (`qc1_env_sensor_fault`, `qc2_acquisition_truncated`, `qc3_unstable_baseline`, `qc4_insufficient_response`, `qc5_sensor_saturation`, `qc6_excessive_spikes`, `clip_length_insufficient`), including zero-count nodes; modeling cohort 972; training set 777; and internal test 195. Required reconciliations are assessed = exclusions + modeling cohort, exclusions = 68, modeling cohort = 972, and modeling cohort = 777 + 195.

Every excluded participant must yield exactly one valid closed-set exclusion-reason key before cohort-flow or exclusion-reason summary counts are generated. Unknown, missing, malformed, or multiple/overlapping frozen failure reasons are a hard integrity failure: halt without producing those outputs, do not choose a priority, do not double-count, and do not invent a mutually exclusive assignment. Overlap requires a separately reviewed, result-free mutually exclusive rule amendment before execution can continue. The same validated-single-reason contract governs the seven cohort-flow exclusion nodes and the seven excluded-group QC-reason rows in the included/excluded summary.

Compare included versus excluded participants with signed included-minus-excluded standardized mean differences and linear quantiles. Freeze exactly nine primary SMD rows in order: age (level null); sex (female); endpoint (yes); and disease_category one-versus-rest levels HC, BGL, LGIN, HGIN, EGC, and AGC. Age uses `(mean_included - mean_excluded) / sqrt((var_included + var_excluded) / 2)` with sample variances `ddof=1` and requires at least two nonmissing observations in each group; if age support is insufficient, retain simpler estimable age descriptions and mark only the main age SMD non-estimable. Sex, endpoint, and the six disease-category rows use the signed binary pooled-proportion formula with fixed focal coding (female=1, Yes=1, named category=1) and do not use the age two-nonmissing support rule. After missing-value handling, binary values must be exact `{0,1}`; unexpected values are fixed-schema errors. Zero denominators return 0 only when group values are identical; otherwise the main SMD is non-estimable. Do not reuse legacy absolute-SMD helpers.

Missingness uses the existing `missingness_smd_included_minus_excluded` field rather than a duplicate primary SMD row. Age, sex, and endpoint rows carry their variable missingness SMDs; the HC disease-category row is the sole carrier of the variable-wide disease-category missingness SMD and is not an HC-specific missingness statistic; BGL–AGC rows leave that field missing. Every missingness SMD uses missing=1 and the signed binary pooled-proportion formula. Row-level `estimability_status` / `nonestimable_reason` govern only the primary `smd_included_minus_excluded` field. Fields `collection_time` and `collection_location` are prohibited in this comparison.

### 8.2 Calibration points and descriptive LOWESS

Use only the 195 frozen internal-test `p_cal` values and aligned labels. Deterministic quantile-bin calibration points use `pd.qcut(q=10, duplicates="drop")` with Wilson intervals; bin edges are not persisted. One-bin all-ties behavior and visualization-readiness rules are frozen. Separately freeze one descriptive LOWESS curve on the fixed 101-point grid using statsmodels 0.14.6 with specified parameters, tie stabilization, observed-support masking, clipping flags, and global non-estimability rules. This LOWESS fit is graphical descriptive evidence, not a probability calibrator, and must not overwrite or replace `p_cal`. Existing mixed calibration evidence remains visible.

### 8.3 Reference-standard and timing fact hierarchy

Create a structured non-statistical audit distinguishing `repository_supported`, `investigator_attested`, `not_systematically_recorded`, and `prohibited_claim`. Exact fact-record identities, evidence statuses, permitted manuscript implications, and prohibited stronger wording are frozen in `configs/post_hoc_final_supplementary.yaml`. The approved conservative facts and non-claims are:

- HC reference standard is based on endoscopic morphology; biopsy is generally not clinically indicated for this category.
- BGL was confirmed by a mixture of endoscopic morphology and biopsy; exact proportions are unavailable.
- LGIN had pre-breath biopsy pathology, possible biopsy undergrading, and no complete ESD resection specimen during the study period.
- Inpatient/outpatient disposition is not a pathological gold standard.
- HGIN, EGC, and AGC final categories were based on complete endoscopic or surgical resection pathology, which may have occurred after breath collection.
- Clinicians responsible for routine endoscopy, pathology reporting, and treatment were unaware of future participation and breath-test results; wording is conservative result-unawareness, not a research-grade adjudication-blinding claim.
- Collector blinding was not documented.
- Breath collection occurred after diagnostic gastroscopy and indicated biopsy but before treatment.
- The exact gastroscopy/biopsy-to-breath interval distribution was not systematically recorded.
- No “generally under three months” interval claim is supported.
- No confirmed central pathology review claim is supported.
- Differential reference standards across disease categories must be disclosed.
- The principal limitation is lack of direct pre-endoscopy-triage validation due to post-gastroscopy/indicated-biopsy sampling.

Do not invent BGL proportions, central review, interval statistics, or “generally under three months” wording. Do not upgrade investigator-attested facts to repository-supported evidence without direct repository evidence.

### 8.4 QC5 historical reconstruction limitation

Git history retains only the final E005 participant-level artifacts, not the overwritten historical 675-member identity list. Aggregate reconciliation (974 evaluable; 298 strict-fail; 676 strict-pass) is required, but exact historical membership identity remains unrecoverable.

### 8.5 QC6 reporting-lineage correction

Session 6 creates `docs/preprocessing/qc6_lineage_correction.md` as a new additive note and never edits the immutable protocol or run summary. The note must cite the operational any-channel rule; executed `any(spike_counts > configured_threshold)` semantics and code commit; frozen threshold `211.0` with equality passing; protocol omission of sample-level aggregation; the run-summary six-channel wording error; zero QC6 failures; and absence of cohort/model impact. Future Methods and cohort-flow text must use the actually executed any-channel rule.

## 8.6 Machine-readable schema freeze

Exact machine-readable record inventories, dtypes, field-population rules, units/scales, missing representations, complete nested JSON leaf contracts, completion-log record grammar, exact hash-map key inventories, and the central row-order/schema validation registry are frozen in `configs/post_hoc_final_supplementary.yaml` before any Session 1 value access. Nested JSON artifacts (score-blind preflight, reference-standard/timing audit, Session 6 execution manifest, Session 7 integrity audit, and future visualization-release authorization) and all four completion logs freeze top-level and nested key/record order, required versus nullable leaves, logical types, closed enums, machine-enforceable status/reason dependencies, exact required map-key sets with omission and undeclared-key rejection, self-hash exclusions, allowlisted provenance timestamps, and rejection of wrong dtypes. A top-level key list, generic type, path pattern, or incomplete schema reference is not sufficient. Session 1–7 implementations may not infer or add undeclared rows, fields, dtypes, status codes, or reason codes. Any future schema change before score access requires review and new hashes. After score access, an unplanned schema or scientific extension requires a separate results-free plan and freeze.

## 9. Known pre-analysis facts and evidence-hierarchy discrepancies

Permitted known facts include the 1040/972/777/195 cohort structure, fixed endpoint, fixed best model, full-precision thresholds, 40/34 date feasibility anchors, and QC5 974/298/676 anchors. Difficult-spectrum counts are first derived in the score-blind Session 1 preflight as feasibility metadata and cannot change the population definition.

Discrepancies that must remain visible and must not be silently repaired:

1. Protocol planning-time 778/194 versus binding operational Erratum E001 and frozen 777/195.
2. Legacy `outputs/logs/experiment_manifest.json` still records `modeling-frozen.commit = PENDING_USER_TAG` and an older stage-head commit.
3. The pre-existing observed-prevalence DCA artifact is outside this package and must not be consumed, recomputed, relabeled, deleted, or folded into this package.
4. QC6 operational any-channel rule, protocol omission, run-summary six-channel wording error, strict `>` execution semantics, zero QC6 failures, and absence of cohort/model impact.
5. The retired acquisition-location plan historically stored as `.cursor/plans/acquisition_context_robustness_audit_4d2cfd18.plan.md` is prohibited and is not a current filesystem dependency.
6. Exact historical QC5 membership identity is unrecoverable despite aggregate reconciliation.

## 10. Explicit exclusions and conservative reporting

Exclude all `collection_location` or disposition analyses; physical-location strata; strict pre-breath HGB analysis; clinical comparator or incremental models; propensity methods; causal adjustment; ComBat/residualization/harmonization; extra disease contrasts or subgroups; extra models or hyperparameter searches; forward temporal split; new threshold, probability calibrator/recalibration, or DCA; repeated sample-size/optimism/threshold-uncertainty work; mechanism or chemical inference; representative participants; visualization production in this package; reporting-library downloads; and external-validation, screening-performance, causal-effect, or clinical-benefit claims.

## 11. Session 7 and visualization authorization gate

Session 7 creates only the integrity audit and its completion log, leaves the Session 6 consolidated report byte-for-byte unchanged, and records `visualization_v2_authorized=false`. Visualization v2 remains blocked until Session 7 is committed and independently approved and the later additive visualization-release authorization file exists and validates completely. This package produces no figures.

## 12. Historical freeze and consolidation non-circularity

The original Session 0 methods snapshot, configuration, and manifest were created before any new score-derived result access or auxiliary fit and remain immutable in commit `3c0ff6b3b40460a2c340ad3193222ee54979109d`. The current manifest records that historical Git identity separately from the consolidated current config and methods hashes. It embeds no manifest self-hash and invents no future Commit B identity; the effective cutover commit can be validated only after the user creates it. Consolidation does not recreate Session 0 or imply that the amended slash parser existed in the original Session 0 blob.
