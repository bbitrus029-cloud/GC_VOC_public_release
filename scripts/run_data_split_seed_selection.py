from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.data_split import evaluate_candidate_seeds


def main() -> int:
    """Run the Session 5 Phase 1 seed-selection workflow once."""
    if len(sys.argv) != 2:
        raise ValueError("Usage: python scripts/run_data_split_seed_selection.py <config_path>")

    config_path = Path(sys.argv[1])
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.info("Starting data split seed selection for %s.", config_path)
    evaluate_candidate_seeds(config_path=config_path)
    logging.info("Completed data split seed selection for %s.", config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
