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
T_BASELINE_WINDOW_INIT = 5  # [domain-physical-constant: SOP-derived early baseline window; acquisition starts before subject blow]
T_BASELINE_END_MIN = 5  # [domain-physical-constant: same SOP-derived minimum early baseline bound as t_baseline_window_init]
T_BASELINE_END_MAX = 50  # [domain-physical-constant: approximately 6 seconds at 8.33 Hz, from prompt user-confirmed SOP requiring prompt blow onset]
BASELINE_SIGMA_MULTIPLIER = 3.0  # [signal-processing-literature: Shewhart 3-sigma control limit; NIST SPC handbook]
BASELINE_CONSECUTIVE_EXCEED = 3  # [signal-processing-literature: short sustained run confirmation adapted from control-chart run-rule logic; Western Electric 1956; Montgomery SPC]
BASELINE_K_OF_N = 4  # [signal-processing-literature: weak-consensus multisensor voting rule; ceil(11 / 3) for 11 gas channels and one-third quorum without sentinel selection]
SG_WINDOW_LENGTH = 11  # [signal-processing-literature: odd-window Savitzky-Golay local polynomial smoothing; at approximately 8.33 Hz this is approximately 1.32 seconds, much shorter than MOS response/recovery dynamics on the order of 10-30 seconds; Savitzky and Golay 1964; Press et al.; MOS denoising practice]
SG_POLYORDER = 3  # [signal-processing-literature: cubic local polynomial commonly used to preserve peak/curve shape while smoothing; Savitzky and Golay 1964; Press et al.]
