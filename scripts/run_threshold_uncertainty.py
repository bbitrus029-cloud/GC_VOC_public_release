from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.threshold_uncertainty import generate_threshold_uncertainty_outputs


LOGGER = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Configure console logging for the Session 3 CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def main() -> int:
    """Run the Plan 1 threshold-uncertainty CLI.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Run Plan 1 post-hoc threshold-uncertainty analysis.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to configs/post_hoc_supplementary.yaml.",
    )
    args = parser.parse_args()

    _configure_logging()
    LOGGER.info("Running post-hoc threshold uncertainty with config=%s.", args.config)

    result = generate_threshold_uncertainty_outputs(config_path=args.config)
    LOGGER.info(
        "Completed threshold uncertainty: threshold_p_raw=%.15f, agreement_delta=%.15f, ci=[%.15f, %.15f], operating_rows=%s, dca_curve_rows=%s.",
        result.recomputed_threshold_p_raw,
        result.agreement_delta,
        result.threshold_ci_low,
        result.threshold_ci_high,
        len(result.operating_point_frame),
        len(result.dca_curve_frame),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
