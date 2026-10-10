"""Check release-contained final-model inference against existing frozen predictions."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from release_inference import FAMILIES, load_model, predict_scores  # noqa: E402

def main() -> int:
    """Run inference only and print aggregate verification diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("all", *FAMILIES), default="all")
    parser.add_argument("--split", choices=("test", "train"), default="test")
    parser.add_argument("--deep-runtime", choices=("stored", "frozen_gpu"), default="stored")
    args = parser.parse_args()
    os.environ.setdefault("NUMBA_CACHE_DIR", str(PROJECT_ROOT / ".release-runtime" / "numba"))
    import torch

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")
    with np.load(PROJECT_ROOT / "data/processed/sequences/pipeline_a_raw.npz", allow_pickle=False) as archive:
        signals = archive["x"]
        ids = archive["sample_id"]
    with np.load(PROJECT_ROOT / "data/processed/splits/train_test_split.npz", allow_pickle=False) as archive:
        selection = archive[args.split + "_idx"]
    features = pd.read_csv(PROJECT_ROOT / "data/processed/features/pipeline_b_raw.csv")
    if not np.array_equal(features["sample_id"].to_numpy(), ids):
        raise ValueError("Public signal/feature alignment failed")
    records = []
    for family in FAMILIES if args.model == "all" else (args.model,):
        model = load_model(family, PROJECT_ROOT)
        inputs = (
            signals[selection] if family in {"cnn_bilstm_attention", "inception_time", "minirocket_ridge"}
            else features.loc[selection, model["feature_names"]].reset_index(drop=True)
        )
        scores = predict_scores(model, inputs, args.deep_runtime)
        suffix = "test_predictions" if args.split == "test" else "train_in_sample"
        reference = pd.read_csv(
            PROJECT_ROOT / f"outputs/predictions/binary_{family}_{suffix}.csv",
            comment="#", float_precision="round_trip",
        ).set_index("sample_id").loc[ids[selection]]
        raw_diff = float(np.max(np.abs(scores["p_raw"] - reference["p_raw"].to_numpy())))
        calibrated_diff = float(np.max(np.abs(scores["p_cal"] - reference["p_cal"].to_numpy())))
        decisions_equal = bool(np.array_equal(scores["y_pred"], reference["y_pred"].to_numpy()))
        records.append({
            "model": family, "n": len(selection), "deep_runtime": args.deep_runtime,
            "raw_max_absolute_difference": raw_diff,
            "calibrated_max_absolute_difference": calibrated_diff,
            "threshold_decisions_identical": decisions_equal,
            "passed": raw_diff <= 1e-6 and calibrated_diff <= 1e-6 and decisions_equal,
        })
    passed = all(record["passed"] for record in records)
    print(json.dumps({"passed": passed, "atol": 1e-6, "rtol": 0.0, "models": records}, allow_nan=False))
    return 0 if passed else 1

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"passed": False, "error_type": type(exc).__name__,
                          "message": "Inference verification failed; no participant values were logged."}))
        raise SystemExit(1) from None
