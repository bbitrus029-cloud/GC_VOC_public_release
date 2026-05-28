from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.calibration_tests import generate_calibration_test_outputs


LOGGER = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Configure console logging for the Session 4 CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def main() -> int:
    """Run the Plan 1 calibration goodness-of-fit CLI.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Run Plan 1 post-hoc calibration goodness-of-fit tests.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to configs/post_hoc_supplementary.yaml.",
    )
    args = parser.parse_args()

    _configure_logging()
    LOGGER.info("Running post-hoc calibration GOF tests with config=%s.", args.config)

    result = generate_calibration_test_outputs(config_path=args.config)
    LOGGER.info(
        "Completed calibration GOF tests: hl_statistic=%.10f, hl_df=%s, hl_p_value=%.10f, "
        "spiegelhalter_z=%.10f, spiegelhalter_p_value=%.10f.",
        result.hosmer_lemeshow_statistic,
        result.hosmer_lemeshow_df,
        result.hosmer_lemeshow_p_value,
        result.spiegelhalter_z,
        result.spiegelhalter_p_value,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
