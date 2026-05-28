# Gastric Cancer VOC Clinical Study

Doctoral research project on exhaled volatile organic compound (VOC) analysis for gastric cancer risk stratification, using a custom MOS MEMS gas sensor array.

## Project Structure

This repository implements a TRIPOD-AI 2024 aligned analysis pipeline:

- **Preprocessing** — Single-pass quality control and signal processing of 1040 raw breath samples, producing two parallel outputs (a time-series tensor for deep learning and a 99-feature matrix for traditional ML). Locked at the `preprocessing-frozen` annotated tag.
- **Modeling** — Nested cross-validation across six models (MiniRocket+Ridge, XGBoost, SVM, Elastic Net LR, InceptionTime, CNN-BiLSTM-Attention), three-step model selection, final training on 777 samples, and a 5-tier prespecified analysis hierarchy on the 195-sample test set. Best model selected: MiniRocket+Ridge with test AUC-ROC 0.9223 [0.8831, 0.9585]. Locked at the `modeling-frozen` annotated tag.

Plans, rules, and agent state live under `.cursor/` (untracked by git). Frozen plan snapshots that serve as the TRIPOD prespecification evidence live under `docs/protocol/`. Stage-specific methodology documents that supplement the prespecification chain live under `docs/<stage>/`.

Six annotated git tags anchor the prespecification timestamps and methodological commitments: `preprocessing-prespecified`, `preprocessing-frozen`, `modeling-prespecified`, `data-split-locked`, `best-model-selected`, `modeling-frozen`.

## Active Stage — Post-Hoc Supplementary Analyses

Post-hoc supplementary analyses (Plan 1) are in progress in preparation for SCI submission. These analyses are conducted AFTER the `modeling-frozen` tag and do NOT modify any model parameter, threshold, calibrator, normalizer, transformer, train/test split, or fold structure. They are documented at `docs/modeling/post_hoc_supplementary_analyses.md` per TRIPOD-AI 2024 transparency guidance for post-hoc work. Scope includes prospective sample size reconstruction, bootstrap-based optimism-corrected validation, decision-curve threshold sensitivity, and calibration goodness-of-fit tests.

## Environment

Python 3.11 with PyTorch 2.10 + CUDA 12.9. Full dependencies pinned in `environment.yml`. Reproduce via:

```
conda env create -f environment.yml
conda activate gcvocs
```

## Status

Preprocessing and modeling stages complete; post-hoc supplementary analyses in progress.