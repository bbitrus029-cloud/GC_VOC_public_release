"""Project-level configuration constants for preprocessing code."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANDOM_SEED = 42  # [domain-physical-constant: project convention fixed before implementation]

S0_DROPPED = "S0"  # [domain-physical-constant: hardware channel layout]
GAS_CHANNELS = [f"S{index}" for index in range(1, 12)]  # [domain-physical-constant: hardware channel layout]
TEMPERATURE_CHANNEL = "S12"  # [domain-physical-constant: hardware channel layout]
HUMIDITY_CHANNEL = "S13"  # [domain-physical-constant: hardware channel layout]
N_GAS_CHANNELS = 11  # [domain-physical-constant: hardware channel layout]

RAW_DATA_ROOT = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_ROOT = PROJECT_ROOT / "data" / "processed"
CONFIGS_ROOT = PROJECT_ROOT / "configs"
OUTPUTS_LOGS_ROOT = PROJECT_ROOT / "outputs" / "logs"
OUTPUTS_ANALYSIS_ROOT = PROJECT_ROOT / "outputs" / "analysis"
METADATA_PATH = RAW_DATA_ROOT / "clinical_data" / "metadata.csv"
BREATH_DATA_ROOT = RAW_DATA_ROOT / "breath_data"
RAW_CHANNELS: list[str] = [S0_DROPPED] + GAS_CHANNELS + [TEMPERATURE_CHANNEL, HUMIDITY_CHANNEL]  # [domain-physical-constant: hardware channel layout]
N_RAW_CHANNELS = 14  # [domain-physical-constant: hardware channel layout]
N_TOTAL_SAMPLES = 1040  # [domain-physical-constant: source cohort size supplied before plan authoring]
RAW_CSV_ENCODING = "utf-8-sig"  # [domain-physical-constant: file-format fact supplied before plan authoring]
