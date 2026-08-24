#!/usr/bin/env python
"""Run the approved aggregate QC-retained/excluded characteristics package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.qc_retained_excluded_characteristics import execute_approved


def build_parser() -> argparse.ArgumentParser:
    """Build the fail-closed command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--execute-approved",
        action="store_true",
        required=True,
        help="Required confirmation for the single authorized participant-capable aggregate execution.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the approved command and run the aggregate-only pipeline."""
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    result = execute_approved(args.config)
    print(
        "Aggregate package created: "
        f"complete_source={result['complete_source']} "
        f"qc_retained={result['qc_retained']} "
        f"qc_excluded={result['qc_excluded']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
