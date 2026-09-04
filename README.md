# Gastric Cancer VOC Clinical Study

Doctoral research project on exhaled volatile organic compound (VOC) analysis for gastric cancer risk stratification, using a custom MOS MEMS gas sensor array.

## Project Structure

This repository implements a TRIPOD-AI 2024 aligned analysis pipeline:

- **Preprocessing** — Single-pass quality control and signal processing of 1040 raw breath samples, producing two parallel outputs (a time-series tensor for deep learning and a 99-feature matrix for traditional ML). Locked at the `preprocessing-frozen` annotated tag.
- **Modeling** — Nested cross-validation across six models (MiniRocket+Ridge, XGBoost, SVM, Elastic Net LR, InceptionTime, CNN-BiLSTM-Attention), three-step model selection, final training on 777 samples, and a 5-tier prespecified analysis hierarchy on the 195-sample test set. Best model selected: MiniRocket+Ridge with test AUC-ROC 0.9223 [0.8831, 0.9585]. Locked at the `modeling-frozen` annotated tag.

Six annotated git tags anchor the prespecification timestamps and methodological commitments: `preprocessing-prespecified`, `preprocessing-frozen`, `modeling-prespecified`, `data-split-locked`, `best-model-selected`, `modeling-frozen`.

## Local Agent Workspace and Migration Provenance

Current local agent guidance is root `AGENTS.md`, and current local operational plans are under `docs/plans/`. Project-level Codex configuration may later live under `.codex/`, and project skills may later live under `.agents/skills/`. These local execution files are intentionally excluded from Git and will not be reconstructed by a fresh clone.

The tracked prespecification evidence remains `docs/protocol/preprocessing_plan.md`, `docs/protocol/modeling_plan.md`, and the annotated Git tags and their history. Legacy `.cursor/plans/...` strings retained in frozen protocols, configurations, manifests, code provenance fields, and run summaries are historical path literals, not current runtime locations. Those frozen historical documents must not be edited merely to clean up legacy path wording. The literal `TBD-at-commit-time` text in a frozen snapshot is a preserved historical placeholder; its actual provenance anchor is the corresponding annotated tag resolved from Git.

Read-only history inspection confirms that the `.cursor/` plan files themselves were never tracked by Git. Any old statement that the operational `.cursor` plan itself was committed is historical documentation imprecision; the tracked protocol snapshots and Git anchors are the repository evidence. After the externally reviewed Commit B cutover, `.cursor/` is no longer a runtime source. Legacy `.cursor` paths remain only where needed to identify historical approval and migration provenance.

The destination hashes below are migration-time local hashes, not frozen scientific artifact hashes.

| Legacy local path | Legacy SHA-256 | New local path | New SHA-256 | Role/status |
| --- | --- | --- | --- | --- |
| `.cursor/plans/acquisition_context_robustness_audit_97707a26.plan.md` | `c1fac4e6ec408f06a9be2efb9748c5ec48d43d981af0faf36bf305c5386b7ef6` | `docs/plans/acquisition_context_robustness_audit_97707a26.plan.md` | `ea38735dce9383adeff839ca5c8bdcd084b5f065f5fd68eeda924cdaa522a452` | Current local operational copy after externally reviewed Commit B cutover; the legacy path/hash remains the exact Session 0 approval provenance, and relocation does not initiate, reopen, or recreate Session 0 |
| `.cursor/plans/modeling_plan_v1_a7c3f9b2_dc789bb5.plan.md` | `771861cc69e96dbf27172528086a42bd4e83045b51aa4bb6055527d0b50652ff` | `docs/plans/modeling_plan_v1_a7c3f9b2_dc789bb5.plan.md` | `542303ec561d2c272853a82c5dbbac59a1ee932c39b42b3373ce4a67dd187029` | Historical modeling operational plan; modeling is frozen |
| `.cursor/plans/modeling_supplementary_plan_v1_b12558f4.plan.md` | `b8221dce5c3b2b70c7ec4ff7204c498fed8ffd0bd8ea8bffe137ec7b19c229eb` | `docs/plans/modeling_supplementary_plan_v1_b12558f4.plan.md` | `369f0c49c3ac5573fc2bdb3c846e0ad42fb307a9cc2539a64b981d8b304d02d2` | Historical post-hoc supplementary Plan 1 operational record |
| `.cursor/plans/preprocessing_plan_v1_539bc1eb.plan.md` | `169f3bc91cd06cf39065ff495b6339052327d6a0de42d65fa26bdcaf5059941e` | `docs/plans/preprocessing_plan_v1_539bc1eb.plan.md` | `0f413a7384410ee976a8d4eb38472be4d93762fd84d5707f1223e9479a4d5f6c` | Historical preprocessing operational plan; preprocessing is frozen |

| Legacy rule path | Legacy SHA-256 | Destination | Migration-time local `AGENTS.md` SHA-256 |
| --- | --- | --- | --- |
| `.cursor/rules/karpathy-guidelines.mdc` | `259cf32ac1a493b7bf863992bbb968079a2b6ffdce99dad092ec6f4e2764717d` | `AGENTS.md` | `1cb134c06871ba1837e0ff235dca9ca6ee0c10c1d08a123ff3912354299f9c9a` |
| `.cursor/rules/modeling-guardrails.mdc` | `9c91e770c31c42e8f0467fcfb5788b218f46a583eecb909221d31a6e20134be3` | `AGENTS.md` | `1cb134c06871ba1837e0ff235dca9ca6ee0c10c1d08a123ff3912354299f9c9a` |
| `.cursor/rules/project-conventions.mdc` | `20bc6fab69a3ce833f2bd1b19fae52d3a4609231a854ac3c74aec709a4cc6583` | `AGENTS.md` | `1cb134c06871ba1837e0ff235dca9ca6ee0c10c1d08a123ff3912354299f9c9a` |
| `.cursor/rules/python-coding-standards.mdc` | `e156928096fbdf23d412ed9e03cb2889f6def4ec1d02d7294a435da6919d20b7` | `AGENTS.md` | `1cb134c06871ba1837e0ff235dca9ca6ee0c10c1d08a123ff3912354299f9c9a` |

Additive reviewer-facing preprocessing reporting and provenance closure is documented in
[`docs/preprocessing/preprocessing_as_executed_reporting_clarification.md`](docs/preprocessing/preprocessing_as_executed_reporting_clarification.md).
It consolidates the final as-executed K-of-N/QC facts and the preprocessing-configuration integrity anomaly; it does not
edit the frozen protocol, configuration, run summary, or scientific artifacts, and does not authorize visualization
production.

## Current Stage — Final Post-Hoc Robustness and Reporting-Closure Package

The original approved legacy plan has SHA-256 `c1fac4e6ec408f06a9be2efb9748c5ec48d43d981af0faf36bf305c5386b7ef6`. Session 0 completed at `3c0ff6b3b40460a2c340ad3193222ee54979109d`, additive Amendment 01 completed at `2e08e8fb89d99bb62cca2e5204a13ff09e47d432`, Session 1 completed at `d19dcbdb7bd9ef20c8534cca0a87384ed1c6fa9f`, and Commit B completed at `ee042999ebd93eefd393fe6446145f8cbf6b75c0`. Session 2 implementation was independently reviewed and committed at `73807ae9b941320aff822a5e53bd11b57b3a310d`; it added exactly the date-grouped analysis module, CLI, and synthetic tests. Session 2 did not read real participant-level values or scores, compute real score-derived scientific results, run real auxiliary fits, or create Session 5 scientific outputs. The current final post-hoc package has one consolidated active record bundle: `configs/post_hoc_final_supplementary.yaml`, `docs/modeling/final_post_hoc_supplementary_methods.md`, and `outputs/analysis/posthoc_final_freeze_manifest.json`. Original Session 0 and Amendment 01 remain independently recoverable and verifiable from Git history; consolidation changes neither their scientific content nor their historical timing and creates no new Session 0. The successful Session 1 score-blind preflight remains byte-preserved.

## Environment

Python 3.11 with PyTorch 2.10 + CUDA 12.9. Full dependencies pinned in `environment.yml`. Reproduce via:

```
conda env create -f environment.yml
conda activate gcvocs
```

Visualization dependencies are likewise pinned in `environment.yml`. The headless LibreOffice renderer is a
separately managed Ubuntu dependency, and licensed Arial fonts are a separately managed local dependency. Exact
runtime versions and font provenance are verified during environment preparation and validation rather than embedded
as scientific source data.

## Visualization Layout and Tracking

- `src/visualization/` contains reusable visualization implementation.
- `scripts/visualization/` contains visualization production entry points.
- `configs/visualization/` contains tracked runtime profiles and object-specific configurations.
- `tests/visualization/` contains synthetic or deidentified visualization tests.
- `outputs/visualization/` contains ignored working, QA, release, supplement-preflight, and handoff products.
- `assets/visualization/source/` contains ignored private source media.

Implementation code, configurations, and explanatory Markdown or JSON outside ignored output namespaces are tracked.
Rendered or binary products and private source media are ignored. Environment and layout preparation does not
authorize visualization production; production sessions remain separately gated by the authority snapshot, execution
plan, registry, and applicable object or configuration contracts.

## Status

Preprocessing and modeling are frozen. This administrative reconciliation changes no scientific or statistical method and grants no new execution authority. Session 3 and all later sessions remain separately gated. Session 5 remains unauthorized and not runnable; it requires prior Session 4 completion, separate review, and explicit later go authorization. Visualization production remains separately gated.
