# Internal manuscript reporting scope decisions

Recorded at UTC: 2026-09-17T03:52:47Z.
Repository observation: branch `main`, HEAD `f49215e2db9b897192ca971cee4e7387049f6bb7`.
The live identity was rechecked; concurrent eFigure 2 changes were present and are outside this task.

This record implements the user's approved documentation-only reporting-maintenance instruction on 2026-09-17.
These reporting decisions were made with prior knowledge of results. They withdraw complete questions or redundant
output families from manuscript reporting, not selected favorable or unfavorable results. They are not a new
prespecification, a claim that an analysis never occurred, or permission to alter frozen evidence. No claim is made
that every retired item was absent from every earlier protocol. Frozen code, methods, outputs, historical approvals,
and original analysis-time identities remain preserved, including all adverse, inconclusive, and non-estimable results.
Mixed-purpose sources retain the cells and sections needed by retained analyses and objects.

The reasons below are internal only: no public retirement inventory or eResults section is authorized.
Retained analyses preserve their actual prospective, prespecified, frozen-before-evaluation, post hoc, exploratory,
or plan-orthogonal identities. The reviewed private execution plan remains unchanged:
`docs/plans/visualization_v2/01_visualization_execution_plan.md`,
SHA-256 `944a5b2e6a2182828429d6a6d154c30e3ca911eedd0e020044d7d87fc3730c73`.
The historical Plan 1 below provides historical context only, not new execution authority.

## R1. Redundant performance expansions

Withdraw the comprehensive training-in-sample performance expansion and additional unplotted, nonselected-model
stage/full-metric expansions because they duplicate the retained reporting questions. This is a partial reporting
disposition, not blanket invalidation of the CSVs. Retain all cells needed by retained objects, six-model core
OOF/internal-test comparisons, selected-model stage evaluations, and existing clinical/technical subgroup results.
Sharing a file with a redundant expansion is not a reason to remove a retained result.

- `outputs/analysis/three_view_all_models_full_set.csv` — SHA-256 `3bacbcac9d228665cb41fd4ddf741a77f2218abc7e42d89f8baeb3ec438d2908`.
- `outputs/analysis/subgroup_hgin_egc_all_models.csv` — SHA-256 `edc59e7e42f213cfe4dae528a5faad5ebf51f66b47185f4801a83dfc836b05b0`.
- `outputs/analysis/subgroup_agc_all_models.csv` — SHA-256 `a39ee2e69c22ff4fe58ae76ea39f971be251579325ad9a493fcbd2054cc8dad0`.

## R2. Additional optimism correction as validation evidence

Withdraw the complete additional optimism-correction result from manuscript performance evidence, preserving the
analysis and its history internally. Scoped code inspection confirmed a procedure mismatch:
`src/analysis/optimism_correction.py` fits MiniRocket once on all development signals before resampling the fixed
feature matrix and fitting a single plain `RidgeClassifier`. The final trainer and configuration use `cv=5`,
`ensemble=true` calibration. Label-blind fitting does not establish independence of the fitted transform from
observed signals; averaging calibrated classifiers is not generally a monotonic transformation of one full-data
Ridge model. No quantified bias is attributed to either simplification. This decision does not invalidate original
OOF/internal-test results and authorizes no rerun or repair.

- `outputs/analysis/posthoc_optimism_correction_minirocket_ridge.csv` — SHA-256 `db62ca86e4ef707f764e858a57ae7964b8412ddcca970f761b5895ed22c581a8`.
- `src/analysis/optimism_correction.py` — SHA-256 `0e42418fcd9431d9df7a9030ac4c926c765adddfa7c6210afc8636fa28efa934`.
- `src/training/train_minirocket_ridge.py` — SHA-256 `726fccbc3b466d52462713a1dd336e2ab6eb011dff7cbf5e2a6e995009c7d6df`.
- `configs/modeling_minirocket_ridge.yaml` — SHA-256 `fa2ba5e1d5cefea39139c1f0b42f2f6d7b182a8e26f5a2a283e7a45d768eb85b`.

Mixed-purpose historical context is preserved; only threshold-uncertainty sensitivity and calibration goodness-of-fit
testing remain in eTable 8's current `retained_posthoc` scope. The historical C01/third-repair attestation and review
continue to describe the original three analyses. They are not rewritten as evidence that only two existed then.
The current historical-plan digest is not a contemporaneous pre-analysis freeze record.

- `docs/modeling/post_hoc_supplementary_analyses.md` — SHA-256 `e81d91f96bdfff48f520b11b726c20f40ac592e5e23d377c517f4c47a93ce5b0`.
- `docs/plans/modeling_supplementary_plan_v1_b12558f4.plan.md` — SHA-256 `369f0c49c3ac5573fc2bdb3c846e0ad42fb307a9cc2539a64b981d8b304d02d2`.
- `configs/post_hoc_supplementary.yaml` — SHA-256 `cc48e3b0fabd596edcbc5dc005eafe038b8ee4a4b9437084e0778f7017572f9d`.

## R3. Ordinal/progression inference across six cross-sectional categories

Withdraw the Jonckheere–Terpstra test, Spearman category-score association, and all five adjacent-category
Mann–Whitney tests together. The retained question is composite-endpoint discrimination and cross-sectional score
distribution, not longitudinal progression or demonstrated stepwise separation across every category. Preserve
six-category distributions and the disease-category prediction matrix, including unfavorable findings. The aggregate
historical source was identity-checked; its participant-level companion CSV was not accessed.

- `outputs/analysis/progression_trajectory_minirocket_ridge.md` — SHA-256 `d921858f097aeed6c7b8b85d90c0fd367733911943b52f0a67b9bb22b35dfb11`.

## R4. Additional FP-LGIN enrichment significance claim

Withdraw the additional FP-LGIN enrichment inference recorded in the audit's
`key_findings.fp_lgin_fisher_odds_ratio` and `key_findings.fp_lgin_fisher_p_two_sided`.
It adds little to retained descriptive evidence and does not establish histologic upgrading or progression.
Retain actual within-category misclassification counts, including LGIN false positives and false negatives elsewhere.
The mixed-purpose audit remains a retained historical source; unrelated findings are unaffected.
No participant-level misclassification files were opened.

- `outputs/analysis/stage7_part_b_audit.json` — SHA-256 `583e3ec909c3abad1c3027fe78f91aafde5aa0784d7704632d5bb23b1fd47f79`.
- `src/analysis/misclassification.py` — SHA-256 `103649bd3bd2a75df5873c1e5999886de8709aec0ef41ead21dbe610158c8aee`.

## R5. Old median-date-half performance expansion

Withdraw this entire older reporting expansion, preserving both files. Retain later within-date concordance and
date-grouped auxiliary validation with their own limitations and analysis-time identities. The older and retained
questions are distinct and not interchangeable. This disposition establishes neither absence of temporal drift nor
forward temporal validation. The CSVs were hashed as opaque bytes without decoding their date columns or data rows.

- `outputs/analysis/temporal_drift_oof_all_models.csv` — SHA-256 `21def5c7296aef8ce11ae9f90dc3568ac9274f84fb52772bd8654413e6e35f8d`.
- `outputs/analysis/temporal_drift_test_all_models.csv` — SHA-256 `2d6c1e2a093288fd2c32b148e6bda88127f66a8a4b65ab47285e04161f885f6b`.

## R6. Additional fallback/padding-versus-misclassification significance tests

Withdraw the Fisher-test reporting family because the retained processing and sensitivity evidence addresses the
reporting question without these additional significance claims. Preserve the frozen tests internally. Retain
processing counts, all four technical sensitivity subsets, clinical subgroup findings, and the evidence used by the
existing technical-sensitivity caption. Non-significance in the withdrawn tests establishes neither equivalence nor
absence of processing effects. The sensitivity CSV remains retained.

- `outputs/analysis/fallback_padded_misclass_minirocket_ridge.md` — SHA-256 `740c116e1182577798fd29691f24444bd51220372934caf92f762b3bbf39cf04`.
- `outputs/analysis/fallback_padded_sensitivity_all_models.csv` — SHA-256 `0910201dd8eb04eed7beb26a47748fb29062e0f8cafa15d79d8402daffd68d3b`.

## R7. Complete-case tumor-marker comparison DCA overlay

Withdraw the entire complete-case tumor-marker DCA comparison family. Retained marker reporting focuses on paired
discrimination and incremental performance in the enriched complete-case subset. Preserve marker/fusion AUROC
comparisons, uncertainty, complete-case context, and unfavorable reclassification findings. Any fusion conclusion
must state no clear discrimination gain, not lack of benefit under every possible utility measure.
Separate primary-model DCA/threshold-uncertainty scenarios remain retained.

- `outputs/analysis/dca_overlay_minirocket_ridge.csv` — SHA-256 `5bfc5855a4c966a9e80fc32c015f07621a96dd3ab1f7fe21830361671df3b2ca`.

## Explicit retentions and interpretation boundaries

NNE is retained as the interpretive reciprocal of PPV, to accompany Figure 4 results with its frozen scenario
assumptions and composite endpoint. No new analysis, figure, table, or eResults section is authorized; Figure 4
is unchanged.

- `outputs/analysis/two_tier_risk_stratification_minirocket_ridge.csv` — SHA-256 `3ec3b31e36954c2bc2a8116e5a28bf019eaf1c2e09444936d2f588b189ba63f9`.

Retain Brier scores, calibration slope/intercept, and post hoc calibration goodness-of-fit results, including
mixed or unfavorable evidence. Retain threshold uncertainty and all already retained robustness, clinical-subgroup,
LGIN, and marker/fusion results, including unfavorable and non-estimable findings. The earlier Riley sample-size
reconstruction retirement remains unchanged.

- `outputs/analysis/posthoc_threshold_uncertainty_minirocket_ridge.csv` — SHA-256 `0187b40dba2c7d0bfcb9d53a8edfe0170df92b151b9c830572393a24a7df05fc`.
- `outputs/analysis/posthoc_threshold_uncertainty_minirocket_ridge_dca_curve_data.csv` — SHA-256 `77c84d126659cfeaed1330561dcb70bc94fc0b421cac9bd5992d577dac8189dd`.

The endpoint remains high-grade gastric lesion: Yes = HGIN + EGC + AGC; No = HC + BGL + LGIN. Scores are not
gastric-cancer probabilities. Validation remains a same-center, same-device, same-period random internal holdout
in enriched case-control/two-gate sampling, without external validation. Breath was collected after diagnostic
endoscopy and indicated biopsy but before treatment; validated pre-endoscopy triage or population screening is
not established. LGIN remains exploratory, with potential biopsy undergrading and differential reference standards.
No causal, deployment-readiness, or demonstrated clinical-benefit claim is added.

## Exact approved eFigure 9 caption amendment

Stable anchor `efigure9_six_model_binned_calibration`; display eFigure 9; session EF-09.
The following exact paragraph is inserted after the sentence ending “target-population absolute risks.” and before
“Abbreviations:”. This is an implementation record of approved text, not a competing editable manuscript.

For MiniRocket + Ridge (A), Brier scores in the internal testing set were 0.1193 (95% CI, 0.0959–0.1423) before and 0.1211 (95% CI, 0.0888–0.1528) after application of the Platt calibrator, yielding a difference (after minus before) of 0.0018 (95% CI, −0.0086 to 0.0121). For the calibrated probabilities, the calibration slope was 0.905 (95% CI, 0.729–1.221), and the intercept was 0.340 (95% CI, 0.002–0.924). Confidence intervals for these summary measures were percentile intervals from 2000 stratified bootstrap samples. Post hoc goodness-of-fit tests applied to the calibrated probabilities yielded a Hosmer–Lemeshow χ² statistic of 29.45 (8 degrees of freedom; *P* < .001) and a Spiegelhalter *z* statistic of 1.79 (*P* = .074).

- `outputs/metrics/binary_minirocket_ridge_test_metrics.csv` — SHA-256 `eb60d67d42383a0ff764633e09b271c98398ce461f20c2d094f7592a1b882835`.
- `outputs/analysis/posthoc_calibration_gof_minirocket_ridge.csv` — SHA-256 `9b2412b5badbfdca8d566b542b59975a982d97605184a2847657c35168d3d92b`.

These are new caption-only `source_02` and `source_03` bindings. Original `source_01` and its legacy
`outputs/analysis/efigure10_six_model_binned_calibration/` package remain unchanged. Six panels, 60 plotted points,
and Wilson intervals remain unchanged. Summary-measure intervals are percentile intervals from 2000 stratified
bootstrap samples; only the last two goodness-of-fit tests are labeled post hoc. The positive delta is after minus
before. Rounded text was checked against immutable machine-readable aggregates, not historical narrative values.

Scoped method evidence (read only; no execution):

- `src/evaluation/calibration.py` — SHA-256 `33e125ca16669758c94b2ecfc323f1fdf776926699175b8b398151c0c70b9ed9`.
- `src/training/common/bootstrap.py` — SHA-256 `33cd38f1bd6b53eff77bdd5aa7c6431b893f61dbd14ff57a25a98545f2f1cb9f`.
- `configs/modeling_global.yaml` — SHA-256 `c4520df8040adfebdfd1f6ae89376be5484ae3dfeda0cfd2481103df323a2e54`.
- `docs/modeling/validated_modeling_run_summary.md` — SHA-256 `9331d8465302cd4215411e8ff4375f7206d21698c51bb5a6b3994192cd02dad3`.

## Exact approved eTable 8 change and source repair

Stable anchor `etable8_temporal_status_provenance`; display eTable 8; session ET-08.
Part A row 6 retains its title, temporal-status cell, and evidence-basis cell. Its execution/result-access cell changes
from:

Included optimism correction, threshold-uncertainty sensitivity, and calibration goodness-of-fit testing; primary evidence was unchanged.

to exactly:

Included threshold-uncertainty sensitivity and calibration goodness-of-fit testing; primary evidence was unchanged.

All 24 scientific rows, all 14 Part B amendments, and seven footnote paragraphs are preserved; all other visible
wording remains unchanged. Current source roles for `source_25`, `source_26`, and `source_31` are narrowed to
threshold-uncertainty sensitivity and calibration goodness-of-fit testing. `source_27` is removed from the current
approved-source closure without renumbering other keys. Historical C01 and prior review evidence retain their
original meaning. No visible retirement row or explanation is added.

Current `source_01`: `docs/modeling/registered_sample_size_reporting_closure.md`.

- Stale pre-addendum SHA-256: `8f2c0619456e98b82c26fa3120164a5f0c0856cd548f59a9d6f552413def352d`.
- Verified current SHA-256: `3abd3a1231519ea69687df1337036ad6331aa20add7ceaf434b6767cbefdfa95`.

The existing 2026-09-12 addendum governs the statements it supersedes; the change is a documented additive
reporting update, not an unexplained source mutation. Source bytes were unchanged by this task. The old digest
remains in the source addendum and historical records; only current dependent bindings are repaired.
No sample-size recalculation, new sample-size number, figure, or table is authorized. Other sample-size
reporting decisions remain unchanged.

## Contract identities and verification

| Contract | Before SHA-256 | After SHA-256 |
|---|---|---|
| `docs/plans/visualization_v2/object_contracts/efigure9_six_model_binned_calibration.contract.md` | `9be2e6318ef9874309d4056330e57d7d8ddaa1c78f75bfc051d578c0be1825b8` | `cc4e35fb7600018a681302cc69aec175ff7df6188f781f617b7e5987d12daa3c` |
| `docs/plans/visualization_v2/object_contracts/etable8_temporal_status_provenance.contract.md` | `4b4854f8092796d98377cccf28bfda82e1ac80485fe7e0f20ae2fb0c3c770f1b` | `9e03dae73e47833b5940e3be42ec747290f5ea988df14410a6b13e1aa01b1e64` |

Verification checkpoint UTC: 2026-09-17T03:52:47Z. Documentation self-checks passed: approved text and source roles,
caption rounding/direction/method labels, preserved table rows and footnotes, source pre/post hashes, JSON parsing,
current contract/registry/closure/snapshot links, historical-review separation, scoped diffs, direct Markdown
whitespace and privacy inspection. Source hashing did not load participant data, dates, signals, models, or
representative-participant outputs. No project content was sent externally.

Registration used sorted existing-directory locks, fresh reads, target conflict checks, destination-local staging,
compare-and-swap checks, and atomic replacements. Unrelated JSON fields were compared against each actual write
boundary and preserved, including EF-02 records and session/production counts. Final shared-authority hashes are
reported in the CLI handoff to avoid self-hash cycles; later disjoint registrations may change whole-file hashes.

The two amendments are implemented/self-checked, pending independent review. Prior independent reviews apply
only to their original bytes. EF-09 and ET-08 remain unproduced and not authorized for production.
No scientific or visualization execution, test suite, environment change, source mutation, release approval, or
Git mutation was performed.
