# Final Post-Hoc Supplementary Pre-Execution Amendment 01

**Amendment ID:** `posthoc_final_pre_execution_amendment_01`  
**Package ID:** `posthoc_final_supplementary`  
**Status:** `approved_for_additive_freeze`  
**Review status:** `approved`  
**Independent review complete:** `true`  
**Effective freeze requires external commit validation:** `true`  
**Amendment freeze commit ID:** `null`  
**Immediate preflight retry authorized at materialization:** `false`  
**Participant score access authorized:** `false`  
**Score-derived computation authorized:** `false`  
**Auxiliary fit authorized:** `false`  
**Session 4 execution authorized:** `false`  
**Visualization v2 authorized:** `false`  
**Results status:** result-free  
**Materialization status:** independent final review approved and approval evidence materialized  

**Machine-readable overlay:** `configs/post_hoc_final_supplementary_pre_execution_amendment_01.yaml`  
**Pre-commit amendment freeze manifest:** `outputs/analysis/posthoc_final_pre_execution_amendment_01_manifest.json`  

This document is an approved, results-free pre-commit additive-freeze package for pre-execution amendment 01. Approval evidence from independent final review has been materialized. No amendment freeze commit exists yet; effective freeze requires later user-controlled commit validation. This package does not authorize a second preflight, score access, Session 4 execution, auxiliary fitting, or visualization v2.

## 1. Purpose

Record the first failed score-blind Session 1 preflight transparently; correct the mismatch between the frozen Session 0 `%Y-%m-%d` date-parser assumption and the actual frozen source encoding; preserve the original Session 0 freeze unchanged; freeze the implementation defects that must be corrected before any second preflight; define deterministic overlay and dual-freeze semantics; and materialize independent approval evidence for a later user-controlled additive freeze commit without establishing that effective freeze in this step.

## 2. Authority and immutability

- Active reviewed plan: `.cursor/plans/acquisition_context_robustness_audit_97707a26.plan.md`  
  SHA-256 `c1fac4e6ec408f06a9be2efb9748c5ec48d43d981af0faf36bf305c5386b7ef6`
- Original Session 0 freeze commit: `3c0ff6b3b40460a2c340ad3193222ee54979109d`
- Original Session 0 parent: `cbb87b0380d7129798002ff4ea8b869c4b786582`
- Original Session 0 path set (unchanged):
  - `configs/post_hoc_final_supplementary.yaml`
  - `docs/modeling/final_post_hoc_supplementary_methods.md`
  - `outputs/analysis/posthoc_final_freeze_manifest.json`
- Original Session 0 artifact SHA-256 values remain the authority hashes recorded in the machine-readable overlay.
- This amendment does not modify the active plan, active rules, Session 0 config, Session 0 methods snapshot, Session 0 freeze manifest, frozen inputs, or the five current Session 1 implementation/test candidates.

## 3. Failed score-blind preflight record

The first Session 1 attempt used `--preflight-score-blind` and failed closed.

Verified or code-path-supported facts:

- Freeze validation completed successfully.
- Opaque frozen-input whole-container hashing completed successfully.
- Prediction semantic loads requested only `sample_id` and `y_true`.
- No `p_raw`, `p_cal`, or `collection_location` was requested.
- Failure stage: modeling-cohort date parsing under the Session 0 `%Y-%m-%d` contract.
- Sanitized failure: `invalid_date_format_count=972`.
- No preflight JSON was promoted; no temporary candidate remained.
- No Session 1 completion log was created.
- No Session 4+ scientific output was created.
- QC5 real-data loading/reconciliation was not reached.
- No Git mutation and no external project-data transmission occurred.

### 3.1 Failed-attempt access state

True:

- `prediction_container_bytes_hashed`
- `prediction_schema_or_header_inspected`
- `participant_level_label_values_read`
- `participant_level_date_values_read`

False:

- `participant_level_score_values_read`
- `participant_level_qc_values_read`
- `new_score_derived_results_computed`
- `auxiliary_model_fit_executed`
- `privacy_reference_loader_executed`

The cumulative state through the failed attempt preserves Session 0 opaque hashing/header inspection plus the actual failed-attempt label and date reads. It does not claim QC access merely because a successful preflight would later reach QC5. These flags describe the failed score-blind preflight attempt only, not the entire historical modeling project.

### 3.2 Amendment-drafting-session access state

**session_activity** for the drafting/inspection activity:

- `prediction_container_bytes_hashed`: true
- `prediction_schema_or_header_inspected`: false
- `participant_level_label_values_read`: true
- `participant_level_score_values_read`: false
- `participant_level_date_values_read`: true
- `participant_level_qc_values_read`: false
- `new_score_derived_results_computed`: false
- `auxiliary_model_fit_executed`: false
- `privacy_reference_loader_executed`: false

**cumulative_through_session** through the end of amendment drafting (same nine ordered keys):

- `prediction_container_bytes_hashed`: true
- `prediction_schema_or_header_inspected`: true
- `participant_level_label_values_read`: true
- `participant_level_score_values_read`: false
- `participant_level_date_values_read`: true
- `participant_level_qc_values_read`: false
- `new_score_derived_results_computed`: false
- `auxiliary_model_fit_executed`: false
- `privacy_reference_loader_executed`: false

Header inspection is cumulative true because it occurred during Session 0 / the failed attempt, even though amendment-drafting activity itself did not re-inspect headers. Scope: no `p_raw`, `p_cal`, QC value, score-derived metric, or model fit occurred during the failed preflight or amendment-drafting activities; this does not restate historical primary-modeling score access outside those activities.

### 3.3 Correction-pass access state

Both document-only correction passes revised only the three amendment candidate documents and had identical all-false `session_activity`. Neither pass reread participant-level source rows, dates, labels, scores, or QC values. The previously recorded aggregate date-format audit was reused without repetition.

**session_activity** for the document-only correction passes (all false):

- `prediction_container_bytes_hashed`: false
- `prediction_schema_or_header_inspected`: false
- `participant_level_label_values_read`: false
- `participant_level_score_values_read`: false
- `participant_level_date_values_read`: false
- `participant_level_qc_values_read`: false
- `new_score_derived_results_computed`: false
- `auxiliary_model_fit_executed`: false
- `privacy_reference_loader_executed`: false

**cumulative_through_session** through completion of this second document-only correction pass:

- `prediction_container_bytes_hashed`: true
- `prediction_schema_or_header_inspected`: true
- `participant_level_label_values_read`: true
- `participant_level_score_values_read`: false
- `participant_level_date_values_read`: true
- `participant_level_qc_values_read`: false
- `new_score_derived_results_computed`: false
- `auxiliary_model_fit_executed`: false
- `privacy_reference_loader_executed`: false

This cumulative state carries forward Session 0, failed-preflight, and amendment-drafting activities. It does not imply any participant-level reread during either document-only correction pass.

## 4. Aggregate source-format audit

Privacy-safe aggregate inspection previously performed on the 972-member modeling cohort (`in_final_output` true) found:

| Aggregate count | Value |
| --- | ---: |
| Source cohort n | 1040 |
| Modeling cohort n | 972 |
| Matches original `%Y-%m-%d` contract | 0 |
| Slash-encoded date-only values | 972 |
| Missing or non-string | 0 |
| Whitespace-containing | 0 |
| Time-of-day markers | 0 |
| Invalid calendar dates under proposed parser | 0 |
| Unique raw date strings | 40 |
| Unique normalized calendar dates | 40 |
| Raw-to-normalized one-to-one | true |
| Normalization collision present | false |
| Normalized dates containing both endpoint classes | 34 |

Digit-length distribution (aggregate only): month 1-digit = 972; month 2-digit = 0; day 1-digit = 305; day 2-digit = 667.

These counts reconcile the known 40/34 date-feasibility anchors. No date string, sample ID, filename, row, date hash, chronological label, or pseudonymous date identifier is persisted by this amendment.

## 5. Date-parser amendment

### 5.1 Historical Session 0 contract (superseded assumption)

Session 0 froze an exact `%Y-%m-%d` date-only contract under the nested base-config key path `loaders.date_parsing`, with regex full-match, `pd.to_datetime(..., exact=True, errors="raise")`, midnight normalization, and exact round-trip text. That contract remains in the historical Session 0 record as a superseded, result-free implementation assumption. This amendment does not rewrite Session 0 history to imply that a slash parser was frozen then.

### 5.2 Proposed replacement contract (methods/provenance description)

The object `proposed_date_parsing_replacement_contract` is a methods/provenance description. It is not a second runtime overlay payload. The sole runtime overlay payload is `overlay_application_contract.overlay_date_parsing_payload`. The descriptive contract must remain semantically consistent with that runtime payload; any semantic conflict fails validation.

Accept only ASCII slash-encoded date-only source grammar matching `^([0-9]{4})/([0-9]{1,2})/([0-9]{1,2})$`:

1. Exactly four ASCII year digits and one or two ASCII digits for month and day.
2. Reject missing values, non-strings, whitespace, time-of-day content, extra tokens, ISO forms, Unicode/non-ASCII digits, and other ambiguous forms.
3. Parse the three numeric components explicitly without generic inference.
4. Validate the real calendar date with an exact parser and `errors="raise"`.
5. Require midnight-normalized parsed values.
6. Require parsed year, month, and day to equal the original numeric tokens.
7. Canonicalize only in memory to a stable date representation suitable for grouping.
8. Require raw unique-string to normalized-date mapping to be one-to-one and collision-free.
9. Persist no raw or normalized date labels, hashes, lookup tables, participant mappings, or date-group identifiers.
10. Report parsing failures only as aggregate counts without echoing a value.
11. Do not rewrite, regenerate, or normalize `data/processed/sample_index.csv` on disk.
12. Treat canonicalization as lossless representation conversion of the frozen date-only field, not as imputation or scientific-value correction.

## 6. Deterministic overlay and dual-freeze contract

`configs/post_hoc_final_supplementary.yaml` remains immutable and remains the base authority. This amendment YAML is the sole additive machine-readable overlay authority.

Exact runtime mapping:

- immutable base configuration: `configs/post_hoc_final_supplementary.yaml`
- exact base target key path: `loaders.date_parsing`
- sole runtime overlay payload source key path: `overlay_application_contract.overlay_date_parsing_payload`
- operation: replace the entire `loaders.date_parsing` object
- recursive or broad merging: forbidden
- all other base configuration key paths: remain authoritative and unchanged
- unknown, missing, duplicate, conflicting, broader override, or descriptive/runtime semantic conflict: fail closed

Overlay rules for future implementation (not implemented in this correction pass):

1. Load and validate the base configuration from the original Session 0 freeze blob first.
2. Load and validate the exact commit-anchored amendment overlay second.
3. Apply only the allowlisted key-path override: `loaders.date_parsing` ← `overlay_application_contract.overlay_date_parsing_payload`.
4. All non-allowlisted base key paths remain authoritative and byte-for-byte unchanged.
5. Future CLI receives both `--freeze-commit` (original Session 0) and `--amendment-freeze-commit` (additive amendment), and uses the fixed project-relative overlay path that must match the amendment-freeze blob.
6. Validate original and amendment paths, sole parents, exact changed-path sets, blobs, hashes, and approval evidence.
7. Failure of either authority chain halts before any participant-level value access.

## 7. Approval materialization and freeze gate

### Current package state (Step 2A materialized)

Approval evidence has been materialized. The package is `approved_for_additive_freeze` with `review_status: approved` and `independent_review_complete: true`. `amendment_freeze_commit_id` remains null. `effective_freeze_requires_external_commit_validation: true`. `immediate_preflight_retry_authorized_at_materialization: false`. No unqualified `amendment_frozen` boolean is used.

Ordered approval evidence:

- reviewed candidate YAML SHA-256: `a11a7cf4db23d352d0c72b6149102777ae2ee3216cd23534fc15a424080cec6e`
- reviewed candidate Markdown SHA-256: `ead4eab9b6f6c52f337d17305d69126535a461cfb3ddea7ec429ccdb56338273`
- reviewed candidate manifest SHA-256: `11f22aa34f286dbb9a262f77754f4c2ef84f749d29a27ebc1d64f2f8803f5ada` (prior reviewed candidate version; not a final-manifest self-hash)
- `review_status`: approved
- reviewer designation or role: Independent read-only methods and provenance reviewer (Codex)
- approval channel or reference: Codex project-governance conversation — independent final review of pre-execution amendment 01 candidate and Step 2 authorization, 2026-07-17
- `approval_date_utc`: `2026-07-17T02:55:07Z`
- `user_confirmation`: true

### A. Approval materialization before commit — completed

Phase A is completed/materialized. The package must not claim that the amendment freeze commit already exists.

### B. Effective freeze after commit — pending

Phase B remains pending external commit validation. The effective amendment freeze exists only after the user creates the exact three-path commit, supplies its full commit ID, and the commit, parent, paths, and blobs validate. The committed package will not require rewriting after commit validation.

### C. Retry authorization remains separate — immediately unauthorized

Phase C remains immediately unauthorized: `immediate_preflight_retry_authorized_at_materialization: false`. The approved amendment freeze alone does not authorize a preflight retry. Exactly one retry becomes conditionally authorized only after approval evidence validates, the external amendment freeze commit validates, the reviewed Session 1 corrections are implemented, required synthetic tests pass, and all other score-blind gates pass. The future CLI determines effective authorization from the frozen conditional contract and current external evidence.

### Step 2A approval-materialization access state

**session_activity** (all false):

- `prediction_container_bytes_hashed`: false
- `prediction_schema_or_header_inspected`: false
- `participant_level_label_values_read`: false
- `participant_level_score_values_read`: false
- `participant_level_date_values_read`: false
- `participant_level_qc_values_read`: false
- `new_score_derived_results_computed`: false
- `auxiliary_model_fit_executed`: false
- `privacy_reference_loader_executed`: false

**cumulative_through_session:**

- `prediction_container_bytes_hashed`: true
- `prediction_schema_or_header_inspected`: true
- `participant_level_label_values_read`: true
- `participant_level_score_values_read`: false
- `participant_level_date_values_read`: true
- `participant_level_qc_values_read`: false
- `new_score_derived_results_computed`: false
- `auxiliary_model_fit_executed`: false
- `privacy_reference_loader_executed`: false

This Step 2A activity is document-only approval materialization. It does not reread participant-level data or prediction containers.

## 8. Exact future amendment freeze commit contract

- The amendment freeze commit has exactly one parent.
- That sole parent is `3c0ff6b3b40460a2c340ad3193222ee54979109d`.
- Its changed-path set is exactly the three amendment paths named in this package.
- None of the five Session 1 implementation/test candidates may enter that commit.
- The commit ID is externally supplied after the user creates it.
- The original Session 0 freeze remains unchanged.

## 9. Required future Session 1 corrections

Do not repair Session 1 code in this Step 2A materialization. Before any second preflight attempt, independently reviewed corrections must include:

1. **C01** — Implement `--execute-approved` as the real Session 4 execution path instead of a placeholder raise.
2. **C02** — Implement the complete Analyses 1, 2, and 4 framework that Session 4 will execute, including exact real-data minimal loaders; difficult-spectrum metrics and composition; strict-QC5 metrics, correctness, and composition; frozen-primary-test LR augmentation; within-date concordance; four exact scientific CSV schemas and row inventories; two isolated deterministic candidate computations; schema/privacy/source validation; individual atomic promotion; and sanitized completion log promoted last.
3. **C03** — Replace the test that prohibits Session 4 output paths from appearing in Session 1 implementation.
4. **C04** — Calculate difficult-spectrum disease counts from the restricted population as six zero-preserving rows; HC and AGC counts must be zero and the six counts must sum to restricted `n`.
5. **C05** — Use exact configured QC5 `strict_status` values: `strict_pass`, `strict_fail`, and `not_applicable`.
6. **C06** — Replace the chained vector-length comparison with an exact all-vector alignment check.
7. **C07** — Pre-access authority and entry gate before participant-level value access: active plan and four rule hashes; required annotated-tag ancestry; original Session 0 commit/parent/paths/blobs; amendment commit/parent/paths/blobs; original and amendment approval evidence; immutable base config and bounded `loaders.date_parsing` overlay; stale targets and completion logs; entry hashes for all required frozen inputs. Does not require an exit re-hash before participant access.
8. **C08** — Machine-enforce complete nested schema semantics, row inventories, dtype/nullability rules, status/reason dependencies, access-state values, and hash-map ordering.
9. **C09** — Synthetic unit tests for the ASCII slash-format parser covering valid one-/two-digit components, invalid calendars, whitespace/time rejection, ISO rejection, extra tokens, Unicode/non-ASCII digit rejection, collision detection, and sanitized no-echo failures.
10. **C10** — Allow no second preflight until reviewed corrections are implemented, synthetic tests pass, dual-freeze validates, and all conditional retry requirements are satisfied.
11. **C11** — Set access-state flags from actual access; do not hard-code QC access when QC was not reached.
12. **C12** — Privacy validators must recognize both ISO-like and slash-like leaked dates without echoing matches.
13. **C13** — Replace the ISO-only parser used by real preflight and later date-based analyses with the exact amended ASCII slash parser; single-value and series parsers share the same frozen contract; canonicalize only in memory.
14. **C14** — Persist computed aggregates into the preflight candidate; frozen constants are comparison anchors only; do not assert equality and then write separate literal 40/34 or 974/298/676 values; cohort/view/disease-composition/access-state values must reflect computed or observed state.
15. **C15** — Post-computation, pre-promotion integrity gate: after permitted computation and immediately before candidate promotion, re-hash every required source/container; require exit hashes to equal entry hashes and frozen authority hashes; fail closed with no promotion or leftover temporary artifacts on mismatch. Entry hashing belongs to C07.
16. **C16** — Full orchestration and call-order synthetic tests of the real preflight orchestrator verifying dual-freeze gates, stale-target checks, usecols/header enforcement, score-blind column prohibitions, ordered label/date/QC access, truthful failure access-state, before/after hashes, no failed promotion/temp leftovers, success-only exact-target promotion, and privacy checks for ISO-like and slash-like leaks.

## 10. Current Session 1 candidates preserved

The following uncommitted implementation/test candidates are recorded by hash, were not modified by amendment drafting, correction passes, or this Step 2A materialization, and must not enter the future amendment freeze commit:

- `src/analysis/posthoc_final_common.py`
- `src/analysis/posthoc_final_locked_scores.py`
- `scripts/run_posthoc_final_locked_scores.py`
- `tests/test_posthoc_final_common.py`
- `tests/test_posthoc_final_locked_scores.py`

Exact SHA-256 digests are in the machine-readable overlay and pre-commit amendment freeze manifest.

## 11. Exclusions

This amendment excludes rewrite of Session 0 artifacts; modification of the active plan or rules; on-disk rewrite of `sample_index.csv`; score-derived results; auxiliary model fit; visualization v2; collection-location analyses; creating the amendment freeze commit in this step; and any second preflight before freeze validation, code repair, and required synthetic tests.

## 12. Non-circularity and final pre-commit status

- Approval evidence has been materialized from externally supplied values.
- The package is approved for a user-controlled additive freeze commit.
- No amendment commit exists yet; `amendment_freeze_commit_id` remains null.
- Effective freeze is established only after the user creates and externally supplies the exact commit.
- Amendment manifest self-hash: forbidden / not embedded.
- The future commit ID is not invented here.
- Final status: approved pre-commit amendment package materialized; user-controlled additive freeze commit pending; effective freeze not yet established; immediate preflight retry not authorized.
