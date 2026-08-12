"""CLI for the deterministic eFigure 9 aggregate prerequisite."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.efigure9_disease_category_prediction_matrix import run_aggregation


def _build_parser() -> argparse.ArgumentParser:
    """Build the fixed aggregation command-line parser."""
    parser = argparse.ArgumentParser(description="Build the aggregate eFigure 9 prerequisite matrix.")
    parser.add_argument("--config", required=True, type=Path, help="Project-relative task configuration path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the aggregate-only workflow behind an aggregate-safe error boundary.

    Args:
        argv: Optional argument sequence for programmatic use.

    Returns:
        Process status: zero on success and one on any closed failure.
    """
    args = _build_parser().parse_args(argv)
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    try:
        result = run_aggregation(config_path=config_path, project_root=PROJECT_ROOT)
    except Exception:
        print("ERROR: eFigure 9 aggregate prerequisite generation failed a closed validation gate.", file=sys.stderr)
        return 1

    print("eFigure 9 aggregate prerequisite generation completed.")
    print("Aggregate matrix rows: 12")
    print(f"Evaluation-set total: {sum(result['calculated_disease_category_totals'].values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
