#!/usr/bin/env python
"""Run the explicitly approved date-grouped MiniRocket auxiliary validation."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.posthoc_final_date_grouped import execute_approved


def _commit_sha(value: str) -> str:
    """Validate one exact lowercase Git commit identifier."""
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise argparse.ArgumentTypeError("Commit must be exactly 40 lowercase hexadecimal characters.")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Build the fail-closed Session 5 command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--freeze-commit", type=_commit_sha, required=True)
    parser.add_argument("--amendment-freeze-commit", type=_commit_sha, required=True)
    parser.add_argument(
        "--execute-approved",
        action="store_true",
        required=True,
        help="Required explicit confirmation for the approved execution path.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the exact command shape and invoke the authority-gated pipeline."""
    args = build_parser().parse_args(argv)
    execute_approved(args.config, args.freeze_commit, args.amendment_freeze_commit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
