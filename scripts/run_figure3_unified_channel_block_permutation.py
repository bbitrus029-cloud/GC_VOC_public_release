from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.figure3_unified_channel_block_permutation import execute_workflow


LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parse the frozen analysis CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run the exploratory unified channel-block permutation importance analysis."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Project-relative frozen analysis configuration path.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the fail-closed scientific workflow and return its process exit code."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    args = _parse_args()
    try:
        result = execute_workflow(args.config)
    except Exception as exc:
        LOGGER.error("Analysis stopped fail-closed at aggregate step; exception_type=%s.", type(exc).__name__)
        return 1
    LOGGER.info("Completed analysis with final_output_dir=%s.", result["final_output_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
