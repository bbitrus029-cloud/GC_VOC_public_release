# Licensing and attribution

## Project grants and boundaries

The researcher confirmed sharing permission/rights for the existing approved deidentified participant data and
fitted models and selected the licenses below. This records researcher confirmation; it is not an independent
review of institutional or participant-consent documents. No author identities or copyright ownership are
inferred from withheld Git identities.

Original project software, associated configurations and technical documentation are licensed under Apache-2.0
in LICENSE. The exact 14 fitted model/threshold/calibrator files below also use Apache-2.0. Approved datasets and
numerical results use CC-BY-4.0 in LICENSE-DATA. Both permit commercial reuse, modification and redistribution
under their terms. These separate scopes are not an unrestricted choice of either license for each file.
No noncommercial, research-only, no-derivatives or compulsory-coauthorship condition is added.

The grant covers distributed material in RELEASE_MANIFEST.json. It does not cover withheld raw sources,
clinical linkage, private preparation/settings, manuscripts, or finished figures/tables. Third-party material
retains original terms and attribution. Recipient-installed dependency distributions are not relicensed or
claimed as project-owned. Standard license texts are reproduced unchanged as legal texts, not offered as
original project works; the Apache appendix remains an upstream example, not a project ownership claim.

## Exact file scopes

The manifest's per-artifact `license_scope` values enumerate exact public paths; the manifest itself is Apache-2.0
technical package metadata under `manifest_license_scope`. Only allowlisted files are covered by this mapping.

| Scope | Actual paths and roles |
| --- | --- |
| Apache-2.0, original project portions | `release_inference.py`; allowlisted `src/`, `scripts/` and `tests/` code; `configs/`, `environment.yml`, `RUNTIME_VERSIONS.json`; protocols and method records under `docs/`; root technical documentation, `MODEL_INDEX.json`, `.gitignore`, `.gitattributes` and `RELEASE_MANIFEST.json`. |
| Apache-2.0, fitted project artifacts | The 14 exact `outputs/models/` paths below. |
| CC-BY-4.0 | All allowlisted `data/processed/` signals/features, public index and locked memberships; `outputs/predictions/`; numerical `outputs/analysis/`, `outputs/metrics/` and `outputs/shap/` files; aggregate `VERIFICATION.json`. |
| Original upstream terms | Standard legal texts `LICENSE`, `LICENSE-DATA` and the two exact `LICENSES/` files; third-party routine portions described below and in THIRD_PARTY_NOTICES.md. |

A JSON suffix does not determine the license. For example, `outputs/analysis/binary_svm_threshold_mapping.json`
and other numerical/audit JSON results are CC-BY-4.0. `MODEL_INDEX.json` and `RUNTIME_VERSIONS.json` are Apache-2.0
technical metadata. Aggregate `VERIFICATION.json`, including administrative annotations, is assigned CC-BY-4.0
as one result/verification document. Narrative elsewhere does not change the underlying dataset licenses.
No dataset, model, threshold, prediction or explanation was regenerated for licensing.

The `_compute_midrank` and `_fast_delong` portions of `src/evaluation/statistical_tests.py` have close structural
correspondence with public Yandex/Netflix implementations. Their precise historical derivation revision is not
recorded. Compatible upstream MIT and BSD-2-Clause-Patent notices are conservatively retained for those portions
in LICENSES/ROC-COMPARISON-MIT.txt and LICENSES/VMAF-BSD-2-Clause-Patent.txt. Apache-2.0 covers original project
contributions; it does not replace upstream notices. The manifest records these function-level exceptions.

## Apache-2.0 fitted artifact paths

- `outputs/models/binary_cnn_bilstm_attention_final.pkl`
- `outputs/models/binary_cnn_bilstm_attention_threshold_calibrator.pkl`
- `outputs/models/binary_elastic_net_lr_final.pkl`
- `outputs/models/binary_elastic_net_lr_threshold_calibrator.pkl`
- `outputs/models/binary_inception_time_final.pkl`
- `outputs/models/binary_inception_time_threshold_calibrator.pkl`
- `outputs/models/binary_minirocket_ridge_final.pkl`
- `outputs/models/binary_minirocket_ridge_fusion_lr.pkl`
- `outputs/models/binary_minirocket_ridge_marker_lr.pkl`
- `outputs/models/binary_minirocket_ridge_threshold_calibrator.pkl`
- `outputs/models/binary_svm_final.pkl`
- `outputs/models/binary_svm_threshold_calibrator.pkl`
- `outputs/models/binary_xgboost_final.pkl`
- `outputs/models/binary_xgboost_threshold_calibrator.pkl`

MODEL_INDEX.json supplies unchanged checksums, feature/channel ordering and thresholds for this set.

## Practical attribution

Use the public project label **Gastric-lesion VOC study** and identify actual relative file paths and SHA-256
values or the actual revision received. For unchanged scientific artifacts, the original public payload commit
is `82695e68c73e182470728ae4c8bd8fbc425f167d`; it is not the future licensing commit.
A practical attribution is: “Gastric-lesion VOC study; [relative file path and version/hash]; [applicable license];
[description of changes, or unchanged].” Replace bracketed fields with verified values. Do not invent author
names, a paper citation, DOI, repository URL or publication date.

For CC-BY-4.0 material, retain supplied attribution/notices, provide the license or a link, indicate modifications
and retain previous modification notices; do not imply endorsement. For Apache-2.0 material, give recipients
LICENSE, retain applicable notices and prominently mark changed files. Preserve THIRD_PARTY_NOTICES.md and
applicable LICENSES/ texts when redistributing covered routine portions. The full legal texts control all
conditions and warranty/liability provisions.

## Status and scientific limits

Sanitized history, the original payload commit and its authenticated postcommit scope review are complete.
Researcher-confirmed sharing permission and license selection are recorded. This update is locally prepared,
unstaged and uncommitted; a new ordinary user commit and its verification remain. Publication/upload,
hosting configuration and any release/DOI creation are deferred, not performed.

Licensing does not establish clinical validation, deployment readiness or external validation. The internal
same-center/device/period design, enriched sampling and post-endoscopy collection chronology remain limitations.
The recorded test command still exits 1 because the CNN comparison exceeds atol=1e-6, rtol=0; five other models
pass and threshold decisions match. CPU deep-model limits, separate original-order GPU evidence, marker/fusion
synthetic-only coverage with participant parity unverified, and no OOF/full-study-retraining claim remain in
MODEL_CARD.md and VERIFICATION.json. No inference or scientific analysis was run for this update.
