# Public data dictionary

The per-file schemas, explicit member/column order, sizes and checksums are declared in RELEASE_MANIFEST.json.
Missing numeric CSV values retain empty fields; existing non-estimable status/reason fields and adverse results
are retained. No computation or presentation rounding was applied to scientific CSV values. CSV floats use their
original decimal text; arrays preserve their source dtypes and values. Use the original pandas default parser for
feature inference, as the frozen training loader did; use round-trip parsing when comparing saved score text.

## Identity, labels and ordering

- sample_id: opaque string with prefix pub_ and 32 hexadecimal characters. It carries no published lookup key or
  collection-order label. All participant tables and arrays are ordered by this public ID. No correspondence or
  mapping key is distributed.
- label / y_true: integer 0 or 1. 1 = HGIN + EGC + AGC; 0 = HC + BGL + LGIN.
- disease_category: integer code ordered 0 HC, 1 BGL, 2 LGIN, 3 HGIN, 4 EGC, 5 AGC.
- data/processed/sample_index.csv has exactly sample_id, label, disease_category for the 972 retained participants.
  Excluded participants are represented only by existing aggregate evidence where available.

## Processed signals and features

data/processed/sequences/pipeline_a_raw.npz contains x (float64, 972 by 960 by 11), sample_id (Unicode),
label (int64), channel_names (S1 through S11), and scalar clip_length_L (960).
The name raw means before fitted model normalization; these are processed scientific inputs, not raw recordings.
x is fractional baseline-corrected, dimensionless sensor response after the documented preprocessing.
The time axis is response-frame position, not a participant collection date/time.

data/processed/features/pipeline_b_raw.csv has 99 float64 feature columns, then sample_id and label.
Order is family-first, then S1 through S11 within each family:
MPD, MND, AUC, Mean, MAS, MDS, T90, ERR, LMR.
MPD/MND/Mean/LMR use dimensionless response units; AUC integrates over unit frame intervals;
MAS/MDS are one-frame response differences; T90 is a zero-based frame index (960 is the source sentinel);
ERR is a dimensionless early/full response-mean ratio. Exact definitions and guards remain in
src/features/extraction.py. No channel identifies a specific molecule.

## Locked positional membership

All index arrays are int64, zero-based, unique within each declared set and sorted in the public positional base.
No membership was regenerated or selected again.

- train_test_split.npz, train_idx.npz and test_idx.npz index the full 972-row public signal/feature order:
  777 training and 195 internal-testing rows.
- outer_fold_0.npz through outer_fold_4.npz index the 777-row array selected by public train_idx, in its stored order.
  train_idx and val_idx preserve the five outer partitions. inner_train_0/1/2 and inner_val_0/1/2 use that same
  777-row base, partition each outer-training subset, and replace the original object-dtype nested containers.
- dl_holdout_0.npz through dl_holdout_4.npz and final_dl_es_holdout.npz use the full 972-row base.
  train_idx / holdout_idx preserve the original outer-fold or final early-stopping membership.
- outer_fold remains the original zero-based fold identity. None of these fields encodes a collection date.

## Predictions and explanations

p_raw is the positive-class model output before the study-level Platt calibrator; MiniRocket's fitted internal
classifier calibration remains part of that model. p_cal is the frozen study-level Platt output, not a population
gastric-cancer probability. y_pred uses p_raw >= the frozen threshold_p_raw. quadrant is TP, TN, FP or FN.
best_hparams_json retains the original per-fold hyperparameters. outer_fold is unchanged.

File identities distinguish nested_cv_oof, final_hp_oof (only where existing), test_predictions and train_in_sample.
Training in-sample predictions retain their warning and must not be used as a generalization estimate.
Minimal participant result projections remove demographics, histories, infection status and biomarkers.
The LGIN projection is the disease-category-defined test subset; no disposition variable is included.

The attention archive contains sample_id, float32 attention_weights (777 by 960), outer_fold,
T_attention and explicitly classified Git/training provenance. This is OOF attention, not SHAP.
The representative MiniRocket archives contain 20 test IDs, quadrants, float32 SHAP arrays (20 by 11)
and float64 processed signals (20 by 960 by 11). Their identities and quadrants agree. Quadrants are safe Unicode,
not object arrays. All public NPZ files load with allow_pickle=False.

## Aggregate results and exclusions

Aggregate results retain original ordering, denominators, uncertainty, missingness, non-estimable reasons,
source-specific CI methods, score scales and classifications. No confidence intervals were recomputed.
The numeric stage/frame summaries are not manuscript artwork. Corrected CA19-9 aggregate statistics and the
original corresponding scientific corrections remain distinguishable. Temporal-half aggregate results omit exact
range endpoints and contain no participant-level chronological membership or date labels.

Exact collection dates/times, filenames, per-sample raw paths/hashes, linkage keys, demographic and biomarker
participant values, disposition and histories are excluded. Provenance timestamps are permitted only in documented
training/Git/generation fields; fixed ZIP timestamps are serialization metadata. Date-grouped analyses and
clinical-covariate analyses consequently have explicit public-data reproduction limits.

## Data license and access status

The approved datasets and numerical results are licensed under CC-BY-4.0; see LICENSE-DATA and the exact
manifest path scopes in LICENSING.md. This includes the unchanged 972-participant payload (777 training,
195 internal test), existing CV/early-stopping membership, predictions, minimal participant analysis projections,
777-participant OOF attention, 20-participant representative SHAP/signals and aggregate results. Numerical JSON
results and aggregate verification remain data/result material even though their suffix is also used for software.
Commercial reuse, modification and redistribution are permitted under the license. Provide project/file
attribution, the applicable license and an indication of changes as described in LICENSING.md.
The researcher confirmed sharing permission/rights for this exact scope; no additional participant variables
were approved or added. The license does not include withheld raw sources or private linkage. Publication/upload
is deferred; this local licensing update awaits the user's new commit and its subsequent scope verification.
