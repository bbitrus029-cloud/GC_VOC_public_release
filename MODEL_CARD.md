# Model card

Research model for prediction of high-grade gastric lesion under internal, same-center/device/period validation.
The case-control sampling and post-endoscopy collection chronology preclude claims of validated population
screening, pre-endoscopy triage, clinical utility or external validation. Scores are not gastric-cancer probabilities.

## Fitted resources and outputs

All 14 source model files were separately authenticated, exported and reopened. The six threshold/calibrator
artifacts agree exactly with their final models. All fitted parameters, feature/channel orders, normalizers,
transformers and calibration state are retained. Internal training IDs are removed; SVM support indices are
rebased to the public training order while fitted support vectors remain unchanged.

| Model key | Input order | Raw threshold | Calibrated equivalent |
| --- | --- | ---: | ---: |
| cnn_bilstm_attention | S1–S11 processed sequences | 0.45293188095092773 | 0.3964746355255372 |
| elastic_net_lr | 99 saved feature names | 0.40580526748804524 | 0.3146791566257776 |
| inception_time | S1–S11 processed sequences | 0.39413952827453613 | 0.3557557653304626 |
| minirocket_ridge | S1–S11 processed sequences | 0.3418938786147648 | 0.2195889732130093 |
| minirocket_ridge_fusion_lr | voc_p_raw, cea, ca199, ca724 | 0.38825113602665146 | 0.3748217645722812 |
| minirocket_ridge_marker_lr | cea, ca199, ca724 | 0.49617533637602745 | 0.5698330564814111 |
| svm | 99 saved feature names | 0.35077951167481025 | 0.3122020511687266 |
| xgboost | 99 saved feature names | 0.3070446848869324 | 0.23217942911948775 |

MODEL_INDEX.json is the authoritative file/checksum/input-order index. Output class order is [0, 1].
p_raw is the positive-class model output, p_cal is the fixed Platt transform of p_raw, and decisions use
the existing raw-score threshold. No threshold or calibrator was selected or fitted for release preparation.

## Verification coverage

Every source/export score and decision comparison passed exactly on the same inputs. Classical-model checks
used all 777 training in-sample and 195 internal-test participants. Both deep models were also checked in
those populations using their stored CPU behavior and the separately recorded GPU contract. OOF reproduction
was not attempted with final fitted models.

Stored CPU inference for CNN–BiLSTM–Attention and InceptionTime fails the unchanged 1e-6 frozen-score tolerance.
The recorded e010_gpu_direct_no_warmup_v1 contract (CUDA, batch 32, one normalization pass, no warmup)
passes in original input order. The public-order entry point passes for InceptionTime and all four classical
families; CNN–BiLSTM–Attention has a retained public-order score discrepancy above 1e-6. Threshold decisions
remain identical. The source model and exported model agree exactly even in that public order.

The actual measured differences and all failed checks are preserved in VERIFICATION.json and the existing
E010 numerical diagnostic package. No tolerance was relaxed, prediction replaced, or model changed to improve
these checks. Original input ordering is withheld for privacy.

Marker-only and fusion models retain their fitted state and pass file-loading and three-row synthetic
inference parity. Participant biomarker inputs are not included; participant-level parity is unverified.
Their raw feature names and order are in MODEL_INDEX.json. No laboratory-unit harmonization or missing-value
imputation is added by the public API. The original laboratory-input conventions remain necessary.

## Dependencies and operational limits

Use environment.yml and RUNTIME_VERSIONS.json. CPU inference remains available; reproducing the recorded
deep GPU contract requires one CUDA device. Only authenticated local pickles should be loaded.
The Python API defaults to stored inference behavior; frozen_gpu is an explicit runtime option.
Fitted-model payload verification is deterministic. Torch pickle containers may have different internal
storage identifiers when reserialized; numeric/state checks, not container identity alone, establish preservation.

The public payload omits raw acquisitions, collection dates and participant covariates. It supports verified
inference and access to frozen predictions/results, not a claim of bitwise full-study retraining or external validation.
The researcher confirmed sharing permission/rights for these fitted artifacts. All 14 model and
threshold/calibrator files enumerated in MODEL_INDEX.json are licensed under Apache-2.0; their exact paths
are listed in LICENSING.md. Original software/configurations/technical documentation use Apache-2.0,
while approved datasets and numerical results use CC-BY-4.0. Upstream software retains its own terms.
Commercial reuse, modification and redistribution are permitted under the applicable licenses. Research status
and validation limits describe the evidence; they do not impose a research-only use restriction.
The current licensing update awaits a new user commit and its verification. Publication/upload remains deferred.
