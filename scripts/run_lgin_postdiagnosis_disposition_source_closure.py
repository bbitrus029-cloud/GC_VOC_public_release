#!/usr/bin/env python
"""Run the approved aggregate LGIN disposition source-closure package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.lgin_postdiagnosis_disposition_source_closure import execute_approved


def build_parser() -> argparse.ArgumentParser:
    """Build the fail-closed command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--execute-approved",
        action="store_true",
        required=True,
        help="Required confirmation for the single authorized aggregate source-closure execution.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the approved command and run the aggregate-only package."""
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        result = execute_approved(args.config)
    except Exception:
        print("Aggregate source closure failed closed.", file=sys.stderr)
        return 1
    print(
        "Aggregate source closure created: "
        f"primary_rows={result['primary_rows']} "
        f"baseline_rows={result['baseline_rows']} "
        f"stability_rows={result['stability_rows']} "
        f"primary_sha256={result['primary_sha256']} "
        f"baseline_sha256={result['baseline_sha256']} "
        f"stability_sha256={result['stability_sha256']} "
        f"manifest_sha256={result['manifest_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
