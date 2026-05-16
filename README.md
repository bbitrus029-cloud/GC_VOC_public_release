# Gastric Cancer VOC Clinical Study

Doctoral research project on exhaled volatile organic compound (VOC) analysis for gastric cancer risk stratification, using a custom MOS MEMS gas sensor array.

## Project Structure

This repository implements a TRIPOD-aligned analysis pipeline:

- **Preprocessing** — Single-pass quality control and signal processing of 1040 raw breath samples, producing two parallel outputs (a time-series tensor for deep learning and a 99-feature matrix for traditional ML).
- **Modeling** — Modeling-stage methodology will be specified and locked separately in a future stage.

Plans, rules, and agent state live under `.cursor/` (untracked by git). Frozen plan snapshots that serve as the TRIPOD prespecification evidence live under `docs/protocol/`. Two annotated git tags — `preprocessing-prespecified` and `modeling-prespecified` — anchor the prespecification timestamps.

## Modeling Stage (Active)

Modeling-stage repository initialization, Stage 1 verification, and protocol snapshot authoring are active. The modeling methodology snapshot is prepared in `docs/protocol/modeling_plan.md` and is intended to be frozen at the user-executed `modeling-prespecified` tag before any modeling implementation commit.

## Environment

Python 3.11 with PyTorch 2.10 + CUDA 12.9. Full dependencies pinned in `environment.yml`. Reproduce via:

```
conda env create -f environment.yml
conda activate gcvocs
```

## Status

Repository initialized — preprocessing implementation pending.