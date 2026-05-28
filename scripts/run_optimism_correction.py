from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.optimism_correction import generate_optimism_correction_outputs
from src.analysis.sample_size import load_post_hoc_config
from src.config import PROJECT_ROOT as SRC_PROJECT_ROOT


LOGGER = logging.getLogger(__name__)


def _resolve_project_path(path_str: str) -> Path:
    """Resolve one project-relative or absolute path."""
    candidate = Path(path_str)
    return candidate if candidate.is_absolute() else SRC_PROJECT_ROOT / candidate


def _format_project_path(path: Path) -> str:
    """Format one path relative to the project root when possible."""
    try:
        return str(path.resolve().relative_to(SRC_PROJECT_ROOT))
    except ValueError:
        return str(path)


def _configure_logging(log_path: Path) -> None:
    """Configure console + file logging for the Plan 1 §3.4 CLI."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        ],
        force=True,
    )


def main() -> int:
    """Run the Plan 1 optimism-correction CLI.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Run Plan 1 post-hoc optimism correction.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to configs/post_hoc_supplementary.yaml.",
    )
    args = parser.parse_args()

    config = load_post_hoc_config(args.config)
    log_path = _resolve_project_path(config["paths"]["outputs_logs_dir"]) / "posthoc_optimism_correction.log"
    _configure_logging(log_path)

    LOGGER.info("Running post-hoc optimism correction with config=%s.", args.config)
    LOGGER.info("Logging to %s.", _format_project_path(log_path))

    result = generate_optimism_correction_outputs(config_path=args.config)
    LOGGER.info(
        "Completed optimism correction: auc_full_apparent=%.6f, mean_optimism=%.6f, auc_optimism_corrected=%.6f, corrected_ci=[%.6f, %.6f].",
        result.auc_full_apparent,
        result.mean_optimism,
        result.auc_optimism_corrected,
        result.corrected_ci_low,
        result.corrected_ci_high,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
