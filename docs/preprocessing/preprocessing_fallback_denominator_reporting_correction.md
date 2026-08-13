# Preprocessing Fallback-Denominator Reporting Correction

Date: 2026-08-12

## Status and scope

This is a dated additive reporting and provenance correction. It preserves the historical preprocessing run summary
and `docs/preprocessing/preprocessing_as_executed_reporting_clarification.md` unchanged. It closes a denominator-labeling
error without rewriting the original record or changing any scientific execution.

The historical run summary described 60 fallback resolutions as though they occurred among the 972 final records.
Frozen preprocessing evidence establishes the following exact denominators instead:

- 60 fallback resolutions among all 1,040 source records;
- 54 fallback resolutions among the 972 final records;
- 6 fallback resolutions among the 68 excluded records;
- exact reconciliation: `60 = 54 + 6`;
- 41 tail-padded records among the final 972, unchanged.

## Frozen execution explanation

The frozen preprocessing implementation performs baseline detection and resolves any fallback before QC and before
final-cohort selection. It increments the `n_fallback_used` counter immediately after fallback resolution for every
source record. QC is applied afterward, and only then is `in_final_output` determined. Consequently, the historical
counter value 60 has the 1,040-record source cohort as its denominator; it is not the fallback count among the 972
final records.

This distinction is consistent with the frozen execution topology: excluded records can contribute to the
source-cohort fallback counter even though they do not undergo the post-QC transformations. The final-cohort fallback
count is therefore 54, while the excluded-cohort fallback count is 6.

## Non-impact statement

This correction is reporting and provenance closure only. It changes no:

- preprocessing code or configuration;
- baseline rule or resolved fallback value;
- QC decision;
- cohort membership;
- Pipeline A or Pipeline B artifact;
- split, model, prediction, threshold, or scientific result.

It does not authorize a new preprocessing execution, participant-level investigation, eFigure computation, figure
design, or release. The aggregate diagnostic exposed no participant-level material and created no scientific output.
No eFigure 4 quantile, signal-distribution aggregate, raw-signal aggregate, Pipeline A value, clinical outcome, or
figure result was generated or inspected in establishing this correction.

## Evidence anchors

| Evidence | Identity |
| --- | --- |
| Repository HEAD | `8c44846477c250a03380eda24a5c99ab63294426` |
| `preprocessing-frozen` target | `622cb3664774b3940f7bcf092004cfff62d06822` |
| `src/data/preprocessing.py` HEAD blob | `929e1f592a56afb76b01ad71ce89bd5f02974d1c` |
| `data/processed/sample_index.csv` HEAD blob | `5010bc6ba77d33a6419ab54ccd8c2c18dc4312f3` |
| Historical preprocessing run-summary HEAD blob | `10233024d95acec400b849a769d33b724ac84e3c` |
| Original eFigure 4 plan SHA-256 | `33381038e69e153a7b4cf401535c50b8a5123de78d7d2a16644804d6a532515e` |

These anchors identify the frozen execution and historical reporting evidence. The aggregate-safe diagnostic used
only the previously authorized minimal audit columns, displayed and persisted no row, identifier, filename, raw path,
raw signal, clinical information, prediction, model value, or participant-level diagnostic, and produced no
scientific artifact.
