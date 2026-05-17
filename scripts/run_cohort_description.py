from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.cohort_description import generate_pos_vs_neg_table, generate_six_categories_table


LOGGER = logging.getLogger(__name__)


def main() -> int:
    """Run the full-cohort descriptive table workflow for Session 5.5.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Generate the Session 5.5 cohort-description tables.")
    parser.add_argument("config_path", type=Path)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    LOGGER.info("Generating Table 1 (Pos vs Neg)...")
    generate_pos_vs_neg_table(args.config_path)
    LOGGER.info("Generating Table 2 (Six Categories)...")
    generate_six_categories_table(args.config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
