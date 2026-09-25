# Additive clinical and reporting clarifications — 2026-09-25

Created at UTC: 2026-09-25T13:29:58Z.

This dated additive clarification records investigator-confirmed reporting facts supplied in the
W26-01/W26-02/W26-03 task. These confirmations were recorded now; they are not newly verified
participant-level evidence, contemporaneous freeze records, or retrospectively prespecified procedures.
The historical sources below remain byte-for-byte unchanged. Their superseded statements describe the
earlier reporting record only; this document governs the identified current-reporting clarifications.
The documentation commit is pending and will be supplied by the user; no commit identity is inferred.

## Reference standard, timing, and personnel

- HC classification was based on endoscopic morphology, without biopsy.
- All BGL participants had biopsy pathology. The group was predominantly gastritis with very few
  gastric ulcers; their exact composition was not recorded. The earlier mixed endoscopy/biopsy
  description is superseded for current reporting.
- HGIN and EGC were classified using complete endoscopic or surgical resection pathology. For AGC
  with a resection specimen, pathology established invasion beyond the submucosa. AGC without a
  complete resection specimen had gastric cancer confirmed by gastroscopic biopsy and unresectable
  locally advanced or metastatic disease established using imaging or operative findings. Those AGC
  participants had gastroscopy/biopsy information before breath collection. Counts by confirmation
  pathway were not recorded.
- Breath collection followed diagnostic gastroscopy and indicated biopsy and preceded treatment.
  Final resection pathology for relevant participants could become available after collection. All
  diagnoses used for research grouping were confirmed before breath-signal preprocessing began.
  The exact endoscopy/biopsy-to-collection intervals were not systematically recorded.
- The investigator was the sole breath collector and knew the available clinical diagnosis. Signal
  preprocessing did not use outcome labels. These facts do not establish formal collector blinding,
  a research-specific blinded adjudication procedure, or a double-blind design.
- The investigator assigned final groups from routine endoscopy, pathology, and applicable clinical
  assessment. Clinical reports underwent routine review; there was no study-specific central review
  or adjudication system. Reviewer counts, qualifications, agreement statistics, and
  confirmation-pathway proportions are not established by these confirmations.

The following crosswalk applies to methods section 8.3, the analyses section
“Reference-standard and timing facts,” configuration
`outputs.scientific_and_closure.reference_standard_timing_audit.planned_fact_records`, and the
audit's `facts` records, authenticated below:

| Legacy fact | Historical statement and current applicability |
|---|---|
| RS01 | The earlier morphology-based HC description with biopsy generally not indicated is refined by the current confirmation that HC had no biopsy. The morphology basis is unchanged. |
| RS02 | The earlier mixed endoscopic-morphology/biopsy description of BGL is historical only; all BGL participants had biopsy pathology. Exact gastritis/ulcer composition remains unavailable. |
| RS05 | The earlier universal complete-resection description across HGIN/EGC/AGC is historical only for AGC confirmation pathways. HGIN/EGC complete-resection classification and possible availability of final resection pathology after breath collection remain valid. |
| RS06 | The earlier broad clinician result-unawareness/future-participation statement is retained as historical attestation, not current evidence of collector blinding or research adjudication. Current reporting uses the specific collector, grouping, and routine-review confirmations above. |
| RS07 | The earlier undocumented collector-blinding statement records the old reporting state. Current reporting specifies that the sole collector knew the available diagnosis; no formal blinding is established. |
| RS11 | The earlier absence of a supported central-review claim is consistent with the new explicit confirmation that there was no study-specific central review or adjudication system. The old audit did not independently verify this later confirmation. |

The unaffected endpoint remains high-grade gastric lesion: Yes = HGIN + EGC + AGC;
No = HC + BGL + LGIN. Differential reference standards remain a limitation. LGIN biopsy may
undergrade disease and is not equivalent to complete resection pathology; LGIN findings remain
exploratory. Postdiagnosis disposition is not a pathological reference standard. Post-endoscopy,
pre-treatment collection, unavailable exact interval distributions, and the absence of direct
pre-endoscopy-triage validation remain applicable. The old audit retains its original fact statuses
and legitimate unaffected uses; it is not independent verification of the later confirmations.

## Clinical variable definitions

Smoking and alcohol use histories were obtained from medical records and categorized as ever use (current or former) or never use. Exposure was not quantified.

Family history was based on existing medical records and concerned gastric cancer in a first-degree
relative. Height and weight came from existing medical records; standardized new on-site
measurements are not established.

Helicobacter pylori status used the most recent available carbon-13 urea breath test before breath
collection, without additional participant self-report. Missing denotes no available test record;
this is not a lifetime/current-and-past infection definition.

CEA, CA19-9, and CA72-4 used the most recent available results following gastroscopy/indicated
biopsy and before breath collection and treatment. No uniform testing time window or formal
prespecified multiple-result selection procedure was established. “Most recent” describes actual
acquisition, not a retrospectively invented protocol.

Individual reasons for unavailable marker measurements were not systematically established.
Selective testing during routine care, including the frequent absence of an indication in HC/BGL,
is the investigator's clinical explanation, not participant-by-participant verification.

## Marker missingness and interpretation

The frozen modeling plan section 4.13 treats tumor-marker missingness as an MNAR working assumption
and prohibits imputation. Its all-three-marker complete-case approach remains unchanged. All-three-marker
complete cases were selected within the original development/testing partitions; comparisons used the
same complete-case testing subset.

Association of missingness with observed diagnostic categories does not establish MNAR or distinguish
it from MAR. The complete-case analysis did not specifically model or correct MNAR. The categorical
mechanism assertions in the validated summary's “Tumor-marker selection bias” paragraph, the opening
paragraph of the selection-bias report, and the matching generator wording are historical reporting
language only, not proof of a missingness mechanism. Current reporting uses the working-assumption
and interpretation boundaries here. The existing selection-background and selection-bias results
remain unchanged; generalization is bounded to the complete-case analysis population.

This clarification changes no endpoint, cohort, partition, frozen analysis identity, method, estimate,
or result. It authorizes no imputation, missingness-mechanism test, relabeling, recalculation, refit, or
numerical change. The original internal random validation remains same-center, same-device, and
same-period, with enriched case-control/two-gate sampling and no external validation; a model score
is not a gastric-cancer probability. Existing post-hoc and exploratory classifications remain intact.

## Authenticated historical sources

Paths are repository-relative. Hashes identify the exact unchanged legacy bytes, not independent
verification of the new investigator confirmations. No generator was executed and no participant
data were accessed for this reporting clarification.

| Legacy path | SHA-256 | Relevant section or fact IDs |
|---|---|---|
| [docs/modeling/final_post_hoc_supplementary_methods.md](final_post_hoc_supplementary_methods.md) | `9a2a2878f8738b716ae9243e79945d46fb013e05c456da19597d53fafdffbc2a` | Section 8.3, Reference-standard and timing fact hierarchy |
| [docs/modeling/final_post_hoc_supplementary_analyses.md](final_post_hoc_supplementary_analyses.md) | `4f3843aa4db65741a0f65e06bb6cb16597ca816befb0eb06cca1dbf0e4b59273` | Reference-standard and timing facts; RS01, RS02, RS05, RS06, RS07, RS11 |
| [configs/post_hoc_final_supplementary.yaml](../../configs/post_hoc_final_supplementary.yaml) | `32a0ffc546053f01e4f2a2b63542b4404d5ce4f9d3f85fcd91b43d4cf06158f9` | outputs.scientific_and_closure.reference_standard_timing_audit.planned_fact_records; RS01, RS02, RS05, RS06, RS07, RS11 |
| [outputs/analysis/posthoc_final_reference_standard_timing_audit.json](../../outputs/analysis/posthoc_final_reference_standard_timing_audit.json) | `a532ed3cbf41e350187b3ce5b9d546f30601f5419768e87f1cb327ed6e721236` | facts: RS01, RS02, RS05, RS06, RS07, RS11; unaffected fact roles retained |
| [docs/protocol/modeling_plan.md](../protocol/modeling_plan.md) | `ee954a01c352189ce0ac64f020cb9d391284af13c13ee63f4572b500dcc4865e` | Section 4.13, Tumor-Marker Complete-Case Rule |
| [docs/modeling/validated_modeling_run_summary.md](validated_modeling_run_summary.md) | `9331d8465302cd4215411e8ff4375f7206d21698c51bb5a6b3994192cd02dad3` | Tumor-marker selection bias paragraph |
| [outputs/analysis/tumor_marker_selection_bias.md](../../outputs/analysis/tumor_marker_selection_bias.md) | `8c410dea724d336b753525c3cc2019cfb5d95edf474e1b1da2e7e62d948490c4` | Opening interpretation paragraph; existing results retained |
| [src/analysis/tumor_marker_model.py](../../src/analysis/tumor_marker_model.py) | `e3c46218d0de7fda76c152376745a9c145a485ef98a44cfe87214d9d12801022` | _build_selection_bias_markdown, matching opening report wording; generator unchanged |
