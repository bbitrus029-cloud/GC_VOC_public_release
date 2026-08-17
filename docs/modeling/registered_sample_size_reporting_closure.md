# Registered Sample-Size Source Verification and Reporting Closure

## Scope and non-impact statement

This additive, reviewer-facing document closes reporting questions about the prospectively registered sample-size
planning and its relationship to the study as executed. It does not rewrite or correct the registered record. It changes
no cohort, split, preprocessing method, threshold, model, result, or frozen artifact. It performs no new sample-size
analysis and authorizes no visualization, manuscript release, or Git operation.

This closure changes only the permitted downstream interpretation of preserved historical sample-size materials. It
does not alter their content or affect unrelated post-hoc analyses.

## Registration-source verification status

Four evidence layers must remain distinct:

1. **Contemporaneous registered wording.** The registered document planned recruitment of approximately 1,000
   participants. It described sample-size planning as based mainly on model-development stability using at least 10
   events per candidate predictor parameter, with diagnostic-performance power also considered. Its stated inputs
   included power 0.90, alpha 0.05, anticipated AUC approximately 0.90, approximately 25 candidate predictor
   parameters, and an anticipated approximately 50% outcome-event fraction in the enriched study sample. It reported
   a net requirement of 625, a main recruitment target of 834 after allowing for approximately 25% potential dropout or
   non-analyzability, and prospective inclusion of approximately 100 participants before the main study for a
   preliminary clinical-validation phase after SOP establishment. Elsewhere, it stated a 7:3 development/test
   allocation.
2. **Later investigator source verification.** On 2026-08-17, the investigator verified the preceding wording against
   the contemporaneously uploaded prospective registration document and separately clarified the original two-track
   calculation history and operational intent. This is investigator source verification, not independent third-party
   verification.
3. **Unavailable original calculation details.** The original Python code, exact Track 1 output, exact formula or
   package, rounding convention, and one-sided versus two-sided setting are unavailable. No exact Track 1 sample size
   can therefore be attributed to the original calculation, and no modern calculation is substituted here.
4. **Frozen as-executed evidence.** Repository-frozen protocol, configuration, run-summary, preprocessing, cohort-flow,
   and train/test composition records establish the later 80:20 modeling rule and the aggregate cohort counts reported
   below. These records describe execution; they do not rewrite the registration source.

## Registered two-track sample-size rationale

**Track 1: diagnostic discrimination.** The registered inputs were power 0.90, alpha 0.05, and anticipated AUC
approximately 0.90. The investigator recalls a single-ROC calculation against chance discrimination, AUC 0.50, and
that it produced a smaller requirement than Track 2. Track 1 was therefore non-determining. Because its original code
and exact output are unavailable, this closure assigns it no exact numerical result and performs no replacement
calculation.

**Track 2: model-development stability.** The investigator-confirmed calculation was:

1. `25 candidate predictor parameters × 10 events per parameter = 250 outcome-positive events` required in the
   development set.
2. At an anticipated outcome-event fraction of 0.50 in the enriched study sample,
   `250 / 0.50 = 500` participants were required in the development set.
3. Under the intended 80% development allocation, `500 / 0.80 = 625` analyzable participants were required overall.
4. Allowing for 25% potential dropout or non-analyzability gave `625 / 0.75 = 833.33`, rounded upward to 834 for the
   main recruitment target.

The anticipated 50% value is an anticipated outcome-event fraction in the enriched study sample, not target-population
prevalence. The approximately 25 candidate predictor parameters were a prospective planning proxy. They must not be
equated with the effective degrees of freedom, feature dimension, or parameter count of the final MiniRocket plus Ridge
model.

## Operational rounding to approximately 1,000 participants

Adding the separately planned approximately 100-participant preliminary phase to the main target of 834 produced an
operational total of approximately 934 participants. This was rounded and reported as approximately 1,000 participants.
The approximately 1,000 target included the preliminary phase; it did not mean approximately 1,000 participants plus a
further approximately 100 participants.

## Registered split discrepancy and as-executed allocation

The registered 7:3 development/test statement remains part of the historical record. The arithmetic yielding the
registered net requirement of 625 is internally consistent with an 80:20 allocation, not a 7:3 allocation. The later
frozen modeling-stage protocol and configuration set `train_fraction = 0.80`, and the locked split as executed comprised
777 development participants and 195 internal-test participants.

The change from the registered 7:3 statement to the modeling-stage and as-executed 80:20 allocation is a protocol
discrepancy and execution deviation. The registered arithmetic and the investigator's attestation support 80:20 as the
intended basis of the sample-size calculation. They do not prove that the registered 7:3 text was a typographical error,
and this closure does not silently rewrite or correct it.

## Planned preliminary phase and as-executed preprocessing

The source wording described approximately 100 participants as a preliminary clinical-validation phase after SOP
establishment. Separately, the investigator later clarified that its intended operational role included using the
earliest approximately 100 acquisitions to establish preprocessing thresholds and a standardized preprocessing
workflow.

That separately staged phase was not independently implemented. Its non-implementation is an execution deviation. It
was not a completed pilot study, completed preliminary-validation cohort, independent validation set, or separately
analyzed approximately 100-participant cohort.

As executed, cross-sample preprocessing estimates were derived outcome-label-blindly from the full 1,040-sample source
cohort or the applicable stage-eligible subsets before preprocessing QC and before the locked random split. The
threshold-discovery population was stage dependent; not every threshold was derived from exactly 1,040 participants.
Outcome labels were not used in these cross-sample estimates. Internal-test signals nevertheless contributed to this
pre-split estimation, so the 195-participant internal test is not a fully end-to-end split-isolated validation. Label
blindness does not remove every possible source of optimism.

## Frozen cohort reconciliation

The aggregate machine-readable evidence reconciles as follows:

- Source cohort entering preprocessing: 1,040 participants.
- Preprocessing/QC exclusions: 68 participants.
- Final modeling cohort: `1,040 - 68 = 972` participants.
- Development set: 777 participants, including 331 outcome-positive events.
- Internal-test set: 195 participants, including 82 outcome-positive events.
- Split reconciliation: `777 + 195 = 972`.
- Event reconciliation: `331 + 82 = 413` outcome-positive events in the complete modeling cohort.

For comparison with the registered 25-parameter planning proxy only, the realized development-set ratio is
`331 / 25 = 13.24` events per candidate predictor parameter. The development denominator is 777 and its event count is
331. The historical `413 / 25 = 16.5` ratio uses the complete 972-participant modeling cohort and must not be labeled as
the development-set events-per-parameter ratio.

Numerically, the 1,040-participant source cohort exceeded the operational target of approximately 1,000, the
972-participant final modeling cohort exceeded the registered net requirement of 625 and main recruitment target of
834, and the 777-participant development set exceeded the planned development count of 500. These comparisons do not
prove model-specific sample-size sufficiency for the final complex model, model stability, or adequate calibration.
They also do not imply that the separately planned approximately 100-participant phase was implemented.

## Disposition of the historical Riley B1-B4 reconstruction

The Riley B1-B4 materials are preserved historical post-hoc methodological exploration created after model freezing.
They were not the prospectively registered sample-size method and are non-authoritative both for reconstructing the
registered decision and for deciding whether the registered target was met. The formulas are not declared intrinsically
invalid. The reporting problem is their retrospective, nonregistered role, the historical implementation's use of all
972 modeling participants and 413 events as though they were the development cohort, and their unsuitability for
rewriting the original planning rationale. The actual development subset contained 777 participants and 331 events.

All conclusions about sample-size adequacy in those historical materials, including the historical criterion-specific
shortfall claim, are retired from manuscript, supplement, limitation, figure, table, abstract, checklist, and
reviewer-response use. They are not evidence that the registered planning failed and are neither a correction nor a
replacement of the registered calculation. They are ineligible as the basis for eFigure 6 or any replacement
visualization.

Every historical Riley file remains unchanged and retains an audit-provenance role only. Nothing is deleted, archived,
renamed, disabled, edited, regenerated, or rerun. The sample-size-specific downstream-consumption statements in
`docs/modeling/post_hoc_supplementary_analyses.md` are superseded by this closure; that file remains unchanged, and its
unrelated post-hoc analyses are unaffected. The existing exclusion of repeated sample-size work in
`docs/modeling/final_post_hoc_supplementary_methods.md` and
`configs/post_hoc_final_supplementary.yaml` remains in force.

## Permitted reporting formulation

The following formulation is permitted for manuscript-facing reporting:

> Sample size was planned prospectively using two parallel considerations. The determining calculation used a
> conventional model-stability heuristic of at least 10 outcome events per approximately 25 candidate predictor
> parameters, requiring 250 outcome-positive events. With an anticipated 50% event fraction in the enriched study
> sample, this corresponded to 500 participants in the development set and 625 analyzable participants under the
> intended 80:20 development/internal-test allocation. Allowing for 25% potential dropout or non-analyzability yielded
> a main recruitment target of 834. A separately planned approximately 100-participant preliminary phase brought the
> operational total to approximately 934, which was rounded to approximately 1,000. A parallel
> diagnostic-discrimination power calculation used power 0.90, alpha 0.05, and anticipated AUC approximately 0.90 but
> produced a smaller, non-determining requirement; because its original code and exact output are unavailable, no exact
> Track 1 sample size is reported.

The following short formulation is permitted for reporting execution:

> The separately planned approximately 100-participant phase was not conducted. Outcome-label-blind cross-sample
> preprocessing estimates were instead derived from the full source cohort or applicable stage-eligible subsets before
> QC and random splitting. The final modeling cohort comprised 972 participants, allocated to a 777-participant
> development set and a 195-participant internal test set. Although labels were not used for the cross-sample estimates,
> the internal test was not fully end-to-end split-isolated.

A separate transparent split statement is required: the registration contained a 7:3 development/test statement,
whereas the later frozen modeling protocol and the analysis as executed used 80:20; this is reported as a protocol
discrepancy and execution deviation, not as a proven typographical error.

## Prohibited interpretations and downstream use

Future reporting must not:

- attribute the registered calculation to Riley B1-B4 or repeat a Riley adequacy conclusion or shortfall as an active
  study conclusion or limitation;
- claim that the final complex model had only 25 parameters, or that an events-per-parameter ratio proves model
  stability, adequate calibration, or model-specific sample-size sufficiency;
- call the anticipated 50% event fraction natural-population prevalence;
- claim that the approximately 100-participant phase occurred or formed a pilot, preliminary-validation cohort,
  independent validation set, or separately analyzed cohort;
- call the internal test external, temporal, prospective validation, or fully end-to-end split-isolated;
- claim that all preprocessing was preregistered or that every preprocessing threshold used exactly the same 1,040
  participants;
- describe the registered 7:3 statement as definitively a typographical error;
- report an exact Track 1 sample size, substitute a modern Track 1 calculation, or authorize or propose another
  sample-size reconstruction; or
- use the Riley reconstruction in a manuscript, supplement, limitation, figure, table, abstract, checklist, or
  reviewer response.

The conditional sample-size eFigure 6 item is closed without production. The historical Riley reconstruction must not
be visualized, and no replacement sample-size figure or table is authorized. This documentation task makes no
supplement-layout, numbering, or visual-design decision beyond closing that conditional content item.

## Source and provenance ledger

- **Starting repository identity:** audited branch `main` at
  `a5a1396149f005eb81d60bc117c88a496300a9c2`.
- **Registration-source content:** investigator source verification dated 2026-08-17; source wording and later
  investigator interpretation are separated above. No repository path is assigned to this source.
- **Frozen modeling-stage allocation:** `docs/protocol/modeling_plan.md` records `split.train_fraction: 0.80`, and
  `configs/modeling_global.yaml` implements `split.train_fraction: 0.80`.
- **Validated modeling execution:** `docs/modeling/validated_modeling_run_summary.md` records the locked 80:20 split and
  its effective aggregate sizes.
- **Preprocessing reporting boundary:**
  `docs/preprocessing/preprocessing_as_executed_reporting_clarification.md` records outcome-label-blind execution-period
  estimation, stage-specific threshold-discovery populations, and the frozen cohort reconciliation.
- **Aggregate cohort flow:** `outputs/analysis/posthoc_final_cohort_flow.csv` is the machine-readable source for the
  1,040-participant source cohort, 68 exclusions, 972-participant modeling cohort, and 777/195 split.
- **Aggregate split composition:** `outputs/analysis/table_one_train_test.csv` is the machine-readable source for 331
  development-set and 82 internal-test outcome-positive events.
- **Historical Riley configuration:** `configs/post_hoc_supplementary.yaml`.
- **Historical Riley implementation and CLI:** `src/analysis/sample_size.py` and `scripts/run_sample_size.py`.
- **Historical Riley tests:** `tests/test_sample_size.py`.
- **Historical Riley outputs:** `outputs/analysis/posthoc_sample_size_reconstruction.json` and
  `outputs/analysis/posthoc_sample_size_reconstruction.md`.
- **Consolidated historical documentation:** `docs/modeling/post_hoc_supplementary_analyses.md`; its sample-size
  downstream-consumption statements are superseded, while the file and its unrelated analyses remain unchanged.
- **Preserved historical operational-plan provenance:**
  `docs/plans/modeling_supplementary_plan_v1_b12558f4.plan.md`; it remains untouched and does not replace registration
  authority.
- **Final exclusion of repeated work:** `docs/modeling/final_post_hoc_supplementary_methods.md` and
  `configs/post_hoc_final_supplementary.yaml` retain the exclusion of repeated sample-size work.
- **Unaffected final post-hoc record:** `docs/modeling/final_post_hoc_supplementary_analyses.md` remains unchanged; this
  closure does not alter its unrelated analyses.
