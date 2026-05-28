from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.sample_size import generate_sample_size_outputs


LOGGER = logging.getLogger(__name__)


def main() -> int:
    """Run the Plan 1 sample-size reconstruction CLI.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Run Plan 1 post-hoc sample-size reconstruction.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to configs/post_hoc_supplementary.yaml.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    LOGGER.info("Running post-hoc sample-size reconstruction with config=%s.", args.config)
    result = generate_sample_size_outputs(config_path=args.config)
    LOGGER.info(
        "Completed sample-size reconstruction: cohort_meets_all_criteria=%s, shortfall_criteria=%s.",
        result.cohort_meets_all_criteria,
        result.shortfall_criteria,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
