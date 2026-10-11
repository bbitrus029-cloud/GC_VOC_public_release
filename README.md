# Gastric-lesion VOC study: local public-release candidate

**Access status:** local candidate; publication and upload are deferred.
The original payload commit `82695e68c73e182470728ae4c8bd8fbc425f167d` and its postcommit scope review are complete
(388 files, 103,075,170 bytes). The 161-commit sanitized history remains verified at
`0d773f541ebc313ffd31dee48c53e3b9832ef781`; the baseline has 162 commits in total.
The researcher confirmed sharing permission for the existing approved deidentified data and fitted models.
This records the researcher's confirmation; institutional or consent documents were not independently inspected.
The current licensing/documentation changes are prepared locally, pending a new user commit and verification
of that actual commit. The historical 388-file review does not cover the new license files.
The current scope is in RELEASE_MANIFEST.json; commit correspondence remains in PROVENANCE.md and VERIFICATION.json.

## Licenses

Original software, associated configurations/technical documentation, and all 14 fitted model/calibrator files
are licensed under [Apache-2.0](LICENSE). Approved data and numerical results, including processed signals,
features, public identifiers/labels/categories, locked memberships, predictions and explanations, are licensed
under [CC-BY-4.0](LICENSE-DATA). These are separate file scopes, not a choice of license for every file.
Both permit commercial reuse, modification and redistribution under their terms.
[LICENSING.md](LICENSING.md) and the manifest give the exact scope; [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
preserves upstream terms. The scientific research and validation limitations below are factual limitations,
not noncommercial or research-only license restrictions.

The prediction endpoint is **high-grade gastric lesion**: HGIN, EGC and AGC are positive; HC, BGL and LGIN are negative.
Model scores are not gastric-cancer probabilities. Validation is an internal random holdout from the same center,
device and period. The enriched case-control/two-gate design and post-endoscopy, pre-treatment breath collection
limit interpretation. Population screening, validated pre-endoscopy triage, clinical benefit, deployment readiness,
causality, and prospective or external validation have not been established. LGIN findings are exploratory and
subject to differential reference standards and possible biopsy undergrading. Cross-responsive MOS channels support
sensor-response-pattern evidence, not identification of individual disease-specific VOC molecules.

## Contents

- Scientific source code, configurations, frozen protocols and scientific method/correction records.
- Processed signal arrays and 99-feature matrices for 972 retained participants.
- Opaque public IDs, binary labels, disease categories, and unchanged locked training/testing, CV and early-stopping membership.
- Existing OOF, final-hyperparameter OOF where available, internal-test and training in-sample predictions.
- Existing OOF attention for 777 training samples and representative MiniRocket SHAP/signals for 20 test samples.
- All 14 existing fitted model/calibration files, including six final candidates and marker-only/fusion models.
- Existing numerical results, supplementary numerical packages, and aggregate export verification.

See [DATA_DICTIONARY.md](DATA_DICTIONARY.md), [MODEL_CARD.md](MODEL_CARD.md),
[PROVENANCE.md](PROVENANCE.md), [RELEASE_MANIFEST.json](RELEASE_MANIFEST.json), and
[VERIFICATION.json](VERIFICATION.json).

## Setup and actual entry points

The dependency declaration is [environment.yml](environment.yml); verification used the installed versions in
[RUNTIME_VERSIONS.json](RUNTIME_VERSIONS.json). Activate the existing gcvocs environment. A future recipient may
reconstruct that environment from the declaration; no packages were installed or changed during this preparation.

A verified release-contained inference check for the selected MiniRocket model is:

~~~bash
conda activate gcvocs
PYTHONDONTWRITEBYTECODE=1 python -B scripts/verify_release_models.py --model minirocket_ridge --split test
~~~

The full audit uses the existing GPU contract for the two deep models:

~~~bash
PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python -B scripts/verify_release_models.py   --model all --split test --deep-runtime frozen_gpu
~~~

The latter command intentionally reports the retained CNN–BiLSTM–Attention public-row-order discrepancy and returns
a nonzero verification status under the unchanged 1e-6 tolerance. It does not refit models or write participant
predictions. The four classical families and InceptionTime passed this public-entry-point comparison.
The Python API is release_inference.load_model and release_inference.predict_scores. Loading requires matching
MODEL_INDEX.json checksums. Use only the authenticated package model files.

## Verified limits

Exported model inference is exactly equal to source-model inference on identical inputs, including public row order.
The stored CPU wrappers for both deep models do not reproduce the frozen scores within 1e-6. The already-recorded
GPU/no-warmup/batch-32 contract does reproduce them in original input order. Public row reordering leaves a small
CNN–BiLSTM–Attention difference above that tolerance; threshold decisions were unchanged. All outcomes, including
failed checks, are in VERIFICATION.json. Frozen prediction files remain authoritative and were not replaced.
Repeated data/result exports are byte-identical with the private ID key. All 14 reconstructed model payloads
match exactly in fitted and sanitized metadata state; pickle storage/string memoization can change container bytes.

Final fitted models do not recreate nested-CV OOF predictions. Marker/fusion participant parity remains unverified
because participant biomarkers are outside the approved payload; only fitted-state and synthetic-input checks ran.
No complete participant-level SHAP dataset exists here for every model or every participant.

Raw breath recordings, clinical linkage, exact collection dates/times, source filenames and original row order are
withheld. Public IDs and positional memberships are transformed together. Full raw preprocessing, date-dependent
analyses, clinical-covariate subgroup/marker analyses, and bitwise retraining cannot be reproduced from this payload.
Legacy training/analysis entry points retain historical source contracts, including integer IDs and private-source
requirements; they are not asserted to run unchanged on the public schema. No training was performed for this release.

Finished manuscript figures/tables, rendered documents, visualization assets, execution logs, editor/agent settings
and private operational plans are excluded. Historical scientific references may point to deliberately withheld
files or original commit IDs; PROVENANCE.md explains the distinction. Do not archive the preparation directory or
the Git directory. Publication is deferred in this preparation workflow. A later public transfer should use
the explicit allowlist and separately verified Git refs; private preparation is outside the licensed package.
