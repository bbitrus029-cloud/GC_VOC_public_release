# Gastric-lesion VOC study: local public-release candidate

**Access status:** locally assembled for the user's preparation commit. This package is not approved for public distribution.
Sanitized history is verified at 0d773f541ebc313ffd31dee48c53e3b9832ef781. The payload commit and final committed-scope verification
remain outstanding. Full original/public commit correspondence is in PROVENANCE.md and VERIFICATION.json.
Institutional permission, participant consent/rights, licensing, final committed-history checks, and hosting decisions
remain unresolved. Nothing in this package asserts author approval or grants a license.

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
the Git directory. Distribution, if later authorized, must use the explicit allowlist and separately verified Git refs.
