# Gastric-lesion VOC study

This repository contains scientific code, processed sensor data, frozen predictions, fitted models and
method records for a study of exhaled volatile organic compounds (VOCs) and gastric-lesion risk stratification.

The endpoint is **high-grade gastric lesion**: HGIN, EGC and AGC are positive; HC, BGL and LGIN are negative.
Model scores are not gastric-cancer probabilities. Validation uses an internal random holdout from the same
center, device and period. The enriched case-control/two-gate sampling and breath collection after diagnostic
endoscopy and indicated biopsy, but before treatment, limit interpretation. Population screening, validated
pre-endoscopy triage, clinical benefit, deployment readiness, and prospective or external validation have not
been established. LGIN findings are exploratory because biopsy can undergrade disease and reference standards
differ. Cross-responsive MOS channels support sensor-response-pattern evidence; they do not identify individual
molecules or establish disease-specific VOCs or causal effects.

## Repository contents

- Processed signal arrays and 99-feature matrices for **972 participants**, with **777 training** and
  **195 internal-test** participants.
- Opaque public IDs, outcome labels and disease categories, with locked train/test, cross-validation and
  early-stopping memberships preserved. No split or membership was regenerated for distribution.
- Existing nested-CV OOF, final-hyperparameter OOF where available, internal-test and training in-sample
  predictions, plus minimal participant analysis projections and aggregate numerical results.
  Training in-sample predictions are not generalization estimates.
- Existing OOF attention for 777 training participants, distinct from the representative MiniRocket SHAP and
  processed-signal exports for 20 test participants. Complete participant-level SHAP for every model or
  participant is not included.
- All **14 fitted artifacts**: six final candidate models, six threshold/calibrator artifacts, a marker-only
  model and a fusion model, including their fitted inference dependencies.
- Scientific source code, runtime configurations, frozen protocols, method records, corrections and
  privacy-filtered scientific history.

[DATA_DICTIONARY.md](DATA_DICTIONARY.md) defines fields, units, label codes, ordering and membership semantics.
[MODEL_INDEX.json](MODEL_INDEX.json) lists the exact fitted files, checksums and input order;
[MODEL_CARD.md](MODEL_CARD.md) describes score meanings, thresholds and verification coverage.
[RELEASE_MANIFEST.json](RELEASE_MANIFEST.json) is the explicit file allowlist with artifact hashes, schemas
and license scopes. Numerical datasets and results are distinct from finished manuscript tables.

## Environment and inference

Use the dependency declaration in [environment.yml](environment.yml) and the recorded verification versions in
[RUNTIME_VERSIONS.json](RUNTIME_VERSIONS.json). The commands below use an existing `gcvocs` environment.
The deep-model GPU comparison requires one CUDA device and the recorded runtime contract.

Run the selected MiniRocket model's internal-test comparison from the repository root:

~~~bash
conda activate gcvocs
PYTHONDONTWRITEBYTECODE=1 python -B scripts/verify_release_models.py --model minirocket_ridge --split test
~~~

The recorded six-model comparison uses the explicit GPU runtime for the two deep models:

~~~bash
PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python -B scripts/verify_release_models.py \
  --model all --split test --deep-runtime frozen_gpu
~~~

**Recorded outcome: exit status 1.** CNN–BiLSTM–Attention's raw maximum absolute difference is
**5.4370611906051636e-6**, exceeding the unchanged **atol=1e-6, rtol=0** comparison.
The other five models—the four classical families and InceptionTime—pass this public-order internal-test
comparison, and threshold decisions match for all six. This is a retained failed score comparison, not an
all-model pass. The command does not refit models or write participant predictions.

The Python entry points are `release_inference.load_model` and `release_inference.predict_scores` in
[release_inference.py](release_inference.py). Loading requires matching [MODEL_INDEX.json](MODEL_INDEX.json)
checksums; use only the authenticated package model files. The API defaults to stored inference behavior;
`frozen_gpu` is an explicit runtime option. Inputs, class order, raw/calibrated score meanings and fixed
threshold conventions are documented in [MODEL_CARD.md](MODEL_CARD.md).

## Verification and reproduction limits

[VERIFICATION.json](VERIFICATION.json) preserves successful and failed checks. Source-model and exported-model
inference agreed exactly on identical authorized inputs, including public row order. Both deep models' stored
CPU wrappers failed to reproduce frozen scores within 1e-6. The separate recorded GPU contract—one normalization
pass, batch size 32 and no warmup—passed in original input order. Public reordering retains the CNN discrepancy
described above. Original input order is withheld for privacy; the original-order GPU result does not establish
a pass for the public-order command. Frozen prediction files remain authoritative and were not replaced.

All 14 exported fitted states and sanitized metadata were preserved. Repeated data/result exports were
byte-identical with the same private identifier key; model reserialization can change pickle container bytes
without changing fitted state. Marker-only and fusion models passed fitted-state, loading and three-row synthetic
inference checks. Participant biomarkers are outside the payload, so participant-level prediction parity for
those two models remains unverified. These checks do not establish behavior on untested inputs.

Inference with final fitted models does not recreate nested-CV OOF predictions. The repository provides the
existing frozen predictions and explanations separately. Historical training and analysis entry points retain
their original contracts, including integer identifiers and private-source requirements; they are not asserted
to run unchanged on the public schema. No training was performed for the release, and bitwise full-study
retraining is not reproduced by these inference checks.

Raw breath recordings, original clinical sources and linkage, exact collection dates/times, original filenames,
per-sample raw paths/hashes and original row order are excluded. Public IDs and positional memberships were
transformed consistently without publishing a correspondence or key. Additional participant demographics,
biomarkers, histories and other clinical covariates are withheld. Full raw preprocessing, date-dependent
analyses and clinical-covariate subgroup/marker analyses therefore cannot be fully reproduced from this payload.
Finished manuscript figures/tables, rendered documents, private visualization assets, execution logs,
editor/agent settings and private operational material are also excluded.

## Licenses and attribution

Original software, associated configurations and technical documentation, and all 14 fitted model/calibrator
artifacts use [Apache-2.0](LICENSE). Approved data and numerical results—including processed signals/features,
public IDs, labels/categories, locked memberships, predictions, explanations and aggregate results—use
[CC-BY-4.0](LICENSE-DATA). These are separate file scopes, not a choice of license for every file.
Both permit commercial reuse, modification and redistribution under their terms.

[LICENSING.md](LICENSING.md) and the manifest give exact path scopes and practical attribution instructions.
Use the project label “Gastric-lesion VOC study,” identify the relevant files and revision or hashes, retain
the applicable license/notices and indicate changes as required. [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
preserves upstream attribution and terms; third-party material and separately installed dependencies are not
relicensed as original project work. The grants cover distributed material, not withheld raw/private sources,
manuscripts or finished figures/tables. Scientific limitations describe the evidence and add no research-only
or noncommercial license restriction.

## Provenance

[PROVENANCE.md](PROVENANCE.md) records the original-to-sanitized correspondence for 161 historical commits,
the six annotated sanitized freeze counterparts, and the two collection-date redactions in the preprocessing
protocol. Scientific rules remain unchanged; the documented source-worktree/Git line-ending distinction is
preserved. Historical references can point to deliberately withheld files or original commit identities.
Filtering changes commit identities and may invalidate signatures; this is sanitized history, not an untouched
original. Git timestamps alone do not prove external preregistration or absence of prior result access.

Dated administrative statements elsewhere describe earlier preparation stages; read the licenses and preserved
scientific verification evidence in their stated scopes, including the failed comparisons described above.
