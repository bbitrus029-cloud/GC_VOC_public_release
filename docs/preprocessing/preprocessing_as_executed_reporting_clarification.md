# Additive As-Executed Preprocessing Reporting Clarification

## Scope and non-impact statement

This reviewer-facing document consolidates the final as-executed K-of-N onset and QC1-QC6 reporting facts and closes
the observed `configs/preprocessing.yaml` integrity-provenance anomaly additively. It changes no preprocessing method,
threshold, cohort decision, model input, result, or scientific conclusion. It does not edit or supersede any frozen
protocol, configuration, run summary, code, or scientific artifact, and it does not authorize visualization design,
generation, rendering, or release.

## Authority and migration provenance

The evidence has five distinct time layers:

1. `docs/protocol/preprocessing_plan.md` is the immutable prespecified protocol, anchored by the annotated tag
   `preprocessing-prespecified` at commit `e3bc1a3db29555046724e95509a7b5b64135d7b0` before preprocessing
   implementation.
2. `docs/plans/preprocessing_plan_v1_539bc1eb.plan.md` is the local historical operational plan. Its current migrated
   copy has SHA-256 `0f413a7384410ee976a8d4eb38472be4d93762fd84d5707f1223e9479a4d5f6c`; it is intentionally
   untracked and is not itself the frozen protocol evidence.
3. Errata E002-E005 record deterministic, outcome-label-blind execution-period resolutions or revisions. They are
   deviations from, or completions of, the earlier specification and must not be relabeled as prespecified decisions.
4. The as-executed code and artifacts were frozen after execution. The annotated tag `preprocessing-frozen` resolves
   to the run-summary commit `622cb3664774b3940f7bcf092004cfff62d06822`.
5. `docs/preprocessing/qc6_lineage_correction.md` is a later additive reporting-lineage correction. It documents the
   QC6 rule already executed in frozen code; it did not change analysis, cohort membership, or models.

Accordingly, preprocessing should be described as transparently prespecified, amended during label-blind execution
where documented, and reported as executed. It is inaccurate to describe every preprocessing decision as
preregistered or prespecified.

## As-executed K-of-N onset semantics

- The gas-channel set is S1-S11. Each channel's initial baseline mean and standard deviation use its first five
  frames.
- A channel onset is the first frame of a run of three consecutive frames for which the absolute deviation is
  strictly greater than `3 × sigma` in that channel.
- The quorum is K=4 of N=11. The sample-level `t_baseline_end` is the fourth-earliest valid channel onset. That one
  sample-level boundary is shared by all 11 gas channels for the participant.
- This procedure aligns onset. It neither time-warps channels nor selects a sentinel channel.
- If fewer than four channels trigger, the deterministic cohort fallback is used. Of 1,040 samples, 980 triggered
  normally and 60 used the fallback; the final fallback was 19 frames. The pre-coercion median was exactly 19.0.
- `t_baseline_end` is the first response frame: `[0, t_baseline_end)` is the half-open baseline interval and
  `[t_baseline_end, n_rows)` is the half-open response interval.

## As-executed QC topology

QC1 is evaluated first. A QC1 failure short-circuits QC2-QC6 and leaves those pass flags and diagnostics unavailable.
After QC1 passes, QC2-QC6 are all evaluated; none short-circuits another, and their failure reasons can accumulate in
the implementation. The frozen excluded cohort was separately validated to contain exactly one reason from the closed
set of QC1-QC6 plus clip-length insufficiency for each of its 68 excluded participants. That frozen-cohort property
does not convert the implementation into six mutually exclusive sequential gates.

## Final QC definitions, thresholds, and frozen counts

| Check | Actual decision variable | Final threshold | Sample-level rule | Threshold-discovery population | Frozen exclusions | Primary authority |
| --- | --- | ---: | --- | --- | ---: | --- |
| QC1 environmental sensor fault | Mean of the first 5 S12 values; S13 retained only as a diagnostic | `833.2593749999999`, enabled, high side | Fail when S12 mean is strictly above threshold; equality passes | All 1,040 source samples | 66 | `scripts/run_discovery.py`; `src/data/qc.py`; `configs/preprocessing.yaml` |
| QC2 acquisition truncated | Trace row count | `1000` rows | Pass when `n_rows >= 1000` | Not data-derived | 2 | `src/config.py`; `src/data/qc.py` |
| QC3 unstable baseline | Per-channel baseline CV | `0.08817505173290488` | A channel is unstable only for CV `> threshold`; fail with at least 6 unstable channels | 974 QC1-passing samples | 0 | `scripts/run_discovery.py`; `src/data/qc.py`; `data/processed/discovery_report.json` |
| QC4 insufficient response | Absolute signed response amplitude by channel | `0.0` | A channel responds for `abs(response_amplitude) >= threshold`; pass with at least 3 responding channels | 972 QC1+QC2-passing samples | 0 | Erratum E004; `scripts/run_discovery.py`; `src/data/qc.py` |
| QC5 sensor saturation | Fraction of trace points within one raw recorded unit of that channel's trace maximum | `0.003` | A channel is saturated only for fraction `> 0.003`; fail with at least 7 saturated channels; equality passes | 972 QC1+QC2-passing samples | 0 | Errata E004-E005; `src/data/qc.py`; `configs/preprocessing.yaml` |
| QC6 excessive transient spikes | Per-channel count above that channel's Tukey upper inner fence for absolute first differences | `211.0` | Fail if any gas channel has spike count `> 211`; equality passes | 972 QC1+QC2-passing samples | 0 | `src/data/qc.py`; `docs/preprocessing/qc6_lineage_correction.md` |

The QC1 threshold remains in raw recorded units and must not be reported as degrees Celsius. QC5 is a within-trace
near-maximum fraction and is not equivalent to confirmed ADC-rail saturation. On the finite frozen data, QC4's zero
threshold is operationally nonrestrictive; its zero exclusions do not prove that all traces had strong physiological
responses. More generally, zero QC3-QC6 exclusions do not prove elimination of all signal artifacts or environmental
confounding.

## Execution-period Errata E002-E005

- **E002:** resolved the unspecified QC1 Otsu fault side deterministically as the smaller-mass class, with the
  higher-mean class as the tie-break. The execution remained outcome-label-blind; the frozen result was the high-side
  rule.
- **E003:** resolved a potentially noninteger K-of-N fallback median by deterministic floor coercion before use as an
  index. The frozen median was already exactly 19.0, so coercion returned 19 without operational effect.
- **E004:** after the initial QC4 and QC5 extraction rules eliminated the usable cohort, revised their threshold
  extraction outcome-label-blindly: QC4 uses the Tukey lower inner fence clipped at zero, and QC5 uses the documented
  95th-percentile fallback when its pooled interquartile range is zero. The final thresholds were 0.0 and 0.003.
- **E005:** retained the E004 QC5 channel threshold and revised only sample-level aggregation to failure at seven or
  more saturated channels.

The later QC6 reporting-lineage correction is not another preprocessing erratum. It is reporting closure documenting
the any-channel strict `>211` rule already executed in code, with no analysis or cohort change.

## Historical wording resolved for future reporting

The following historical wording must not be reused in future Methods, captions, flow diagrams, or figure source
contracts:

1. `outputs/analysis/preprocessing_run_summary.md` groups all QC3-QC6 threshold extraction under the QC1+QC2-passing
   cohort. QC3 actually used the 974 QC1-passing samples; QC4-QC6 used the 972 QC1+QC2-passing samples.
2. The run summary can be read as six sequential gates. The executed topology is QC1 first, then complete evaluation
   of QC2-QC6 for every QC1-passing sample.
3. The run summary gives QC6 a six-channel quorum. The executed and future-reporting rule is failure when any channel
   has a spike count strictly above 211.
4. The operational plan's line-level QC3 wording uses `[0, t_baseline_end]`. The binding convention and code use the
   half-open interval `[0, t_baseline_end)`.
5. E003's rationale states that floor coercion leaves more samples in the baseline interval and implies a later onset.
   Mathematically, lowering `t_baseline_end` shortens `[0, t_baseline_end)` and begins the response interval earlier.
   Because this cohort's fallback median was already exactly 19.0, the explanatory-direction error had no operational,
   cohort, or model impact.

These are reporting clarifications. The historical files remain unchanged.

## Cohort arithmetic and downstream clip-length check

The frozen source preprocessing cohort contains 1,040 samples. QC exclusions reconcile as
`66 + 2 + 0 + 0 + 0 + 0 = 68`. A separate downstream clip-length-insufficiency check excluded 0 samples; it is not
QC7. The final modeling cohort is therefore `1040 - 68 - 0 = 972`, and the locked split reconciles as
`777 + 195 = 972`. Each frozen excluded participant had exactly one validated closed-set exclusion reason. Tail
padding and K-of-N fallback are processing outcomes, not exclusion reasons; 41 included samples were tail-padded and
60 samples used the K-of-N fallback.

## `configs/preprocessing.yaml` integrity closure

The bounded local investigation reproduced the following facts without modifying the configuration:

| Diagnostic | Revalidated value |
| --- | --- |
| Exact byte size | `16564` |
| Worktree and HEAD Git blob | `b47eec6917588d82f3e866801aff0aede2ea62d0` |
| SHA-256 returned by GNU `sha256sum`, Python `hashlib.sha256`, OpenSSL, and the HEAD-blob stream | `e3b0d8d9331eda2685926120a3dc7de1a62ba1f5a7f2d411fce87838a63a0664` |
| SHA-1 diagnostic | `b4242cd41a51fa95a659c58311d27497825580e7` |
| MD5 diagnostic | `0edb86875f0df5bd84c4e9e3627ac7a5` |
| BLAKE2b-512 | `ffe081c644b04d571d693af8c7626151282d19ba5161c3c1f76af6a18e8e480667076eb18ac691653e3e9393031dc30139bd882ac57d7b8b1b36d1941e2b41fb` |
| SHA-256 after streaming all but the final byte | `3db02fc49f74bba92d99fb7b8f94a5d4b41d390fa068a7e1c6d01fa1bfd9a53a` |
| SHA-256 after streaming the content plus one `x` byte | `a9642219bda6b0443544522b02fdd68d114a40c50016af345423ff9d5bd278cd` |

The file is nonempty, readable, valid YAML with the required tracked configuration structure, and byte-identical to
the HEAD blob. Its content was last changed at commit `8e092fabf9838498c8d452d244e57701b2c54806` and remains
unchanged at the audited HEAD `9906e9a845f18136e8328d9f4d8136c432668cdd`. The one-byte perturbations produced
ordinary nonempty SHA-256 values.

The anomalous empty-file SHA-256 value is also literally retained for this path in
`configs/post_hoc_final_supplementary.yaml` and `outputs/analysis/posthoc_final_freeze_manifest.json`; read-only blame
attributes both frozen records to commit `3c0ff6b3b40460a2c340ad3193222ee54979109d`. Those fields remain historical
records. For this one path, the value is non-discriminating, is unsuitable as the sole content-identity check, and must
not be treated as affirmative evidence that the file is empty.

No reproducible local evidence established a root cause. The bounded conclusion is therefore **unresolved local
SHA-256 computation/provenance anomaly**. This document does not claim a SHA-256 collision, preimage, cryptographic
break, file corruption, or malicious manipulation. Current source identity is anchored additively by the audited HEAD
and last content commit, the Git blob, exact byte size, and revalidated BLAKE2b-512 value. SHA-1 and MD5 are only
corroborating diagnostics, not the primary cryptographic integrity guarantee. No preprocessing value, participant
inclusion decision, model input, result, or scientific conclusion changes.

## Permitted future use and prohibited interpretations

This clarification may be used as an audited source for a separately reviewed future plan. It does not authorize
visualization production and specifies no figure layout, panel, wording, typography, color, or rendering choice. It
does not establish chemical specificity, identify an individual VOC, infer sensor chemistry, confirm ADC bit depth,
support a biological mechanism, or demonstrate hardware superiority. Any future attempt to establish a conventional
non-anomalous SHA-256 for the configuration requires separate authorization and must be additive; frozen fields and
artifacts must remain unchanged.

## Source and commit ledger

Primary file authorities were `docs/protocol/preprocessing_plan.md`, the local operational plan,
`src/config.py`, `src/data/baseline.py`, `src/data/qc.py`, `src/data/preprocessing.py`,
`scripts/run_discovery.py`, `configs/preprocessing.yaml`, `data/processed/discovery_report.json`,
`outputs/analysis/preprocessing_run_summary.md`, `outputs/analysis/posthoc_final_cohort_flow.csv`,
`docs/preprocessing/qc6_lineage_correction.md`, the cohort-flow and QC6 contracts in
`configs/post_hoc_final_supplementary.yaml`, `src/analysis/posthoc_final_reporting.py`, and the final post-hoc freeze,
execution, and integrity manifests.

| Commit | Audited role |
| --- | --- |
| `e3bc1a3db29555046724e95509a7b5b64135d7b0` | Prespecified protocol anchor |
| `45b8204591c963d50ddf9aede0ba60ac040840a5` | K-of-N implementation |
| `491041b50391d9ad0d2fb4819560af35b7be2084` | Discovery orchestration implementing E002 and E003 |
| `421254714d7af7ba4466e1424bb443f26270d464` | Initial discovery artifacts and QC6 threshold provenance |
| `595d7db1525560e5e2e61936a222a71abbfe2762` | QC implementation, including actual QC topology and QC6 rule |
| `15b5179ac2a465962c6e9542cf0399c17a028b47` | E004 QC4/QC5 extraction implementation |
| `a1600896135070c33cc2f77e6e00578bb264e365` | E004 QC5 threshold-method provenance field |
| `afd6b0c97b5f7d8e2df891924b59059dc81e0d34` | E004 discovery-artifact regeneration |
| `337214fb6d7ef3c3a836cfb92cb5f75b1dcc9785` | QC5 threshold-method provenance-label completion |
| `8e092fabf9838498c8d452d244e57701b2c54806` | Final discovery regeneration and current configuration content |
| `92b015fb8c9f7d29d117becdadf533aed6189e2e` | E005 seven-channel QC5 aggregation implementation |
| `650ce0e12c5f638f324ddad18f36812b14bc90ee` | Final preprocessing artifacts |
| `622cb3664774b3940f7bcf092004cfff62d06822` | Frozen preprocessing run summary and `preprocessing-frozen` target |
| `3c0ff6b3b40460a2c340ad3193222ee54979109d` | Frozen post-hoc records containing the anomalous SHA-256 field |
| `77a5d8fdf0483f63d0bcdcc53c6b77a757133266` | Additive QC6 correction and frozen cohort-flow closure |
| `3478442b26bf42e7963b61916d8ac6180984ebbc` | Final post-hoc integrity audit |
