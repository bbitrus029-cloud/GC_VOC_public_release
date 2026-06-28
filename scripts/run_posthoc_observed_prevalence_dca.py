from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.observed_prevalence_dca import generate_observed_prevalence_dca_outputs


LOGGER = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Configure console logging for the Session 9A CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def main() -> int:
    """Run the Session 9A observed-prevalence DCA CLI.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Generate the post-hoc observed/internal-test prevalence DCA artifact."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to configs/post_hoc_supplementary.yaml.",
    )
    args = parser.parse_args()

    _configure_logging()
    LOGGER.info("Running observed-prevalence DCA addendum with config=%s.", args.config)
    result = generate_observed_prevalence_dca_outputs(config_path=args.config)
    LOGGER.info(
        "Completed observed-prevalence DCA: prevalence=%.15f, n=%s, n_events=%s, n_nonevents=%s, rows=%s.",
        result.observed_prevalence,
        result.n,
        result.n_events,
        result.n_nonevents,
        len(result.output_frame),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
