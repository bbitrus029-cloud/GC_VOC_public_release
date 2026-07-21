"""CLI entry point for score-blind preflight and later approved execution."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.posthoc_final_locked_scores import execute_approved, run_score_blind_preflight


_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _freeze_commit_argument(value: str) -> str:
    """Validate a full lowercase freeze-commit SHA argument."""
    if _COMMIT_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("freeze commit must be exactly 40 lowercase hexadecimal characters")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Build the explicit Session 1 command-line contract."""
    parser = argparse.ArgumentParser(description="Run final post-hoc score-blind preflight or approved execution.")
    parser.add_argument("--config", type=Path, required=True, help="Consolidated current post-hoc YAML configuration.")
    parser.add_argument(
        "--freeze-commit",
        type=_freeze_commit_argument,
        required=True,
        help="Exact historical Session 0 commit for immutable Git-chain validation.",
    )
    parser.add_argument(
        "--amendment-freeze-commit",
        type=_freeze_commit_argument,
        required=True,
        help="Exact historical Amendment 01 commit for immutable Git-chain validation.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--preflight-score-blind",
        action="store_true",
        help="Run score-blind Session 1 preflight without loading p_raw or p_cal.",
    )
    mode.add_argument(
        "--execute-approved",
        action="store_true",
        help="Run later approved locked-score framework; never selected by default.",
    )
    parser.add_argument(
        "--confirm-reviewed-session1-corrections-and-tests",
        action="store_true",
        default=False,
        help=(
            "Explicitly confirm that the reviewed Session 1 corrections and their synthetic "
            "tests passed. Accepted only with --preflight-score-blind."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected explicit mode with concise non-sensitive status output."""
    parser = build_parser()
    args = parser.parse_args(argv)
    confirmed = args.confirm_reviewed_session1_corrections_and_tests
    if args.execute_approved and confirmed:
        parser.error(
            "--confirm-reviewed-session1-corrections-and-tests is accepted only with --preflight-score-blind"
        )
    if args.preflight_score_blind:
        run_score_blind_preflight(
            args.config,
            args.freeze_commit,
            args.amendment_freeze_commit,
            synthetic_tests_passed=confirmed,
        )
        print("Score-blind preflight promoted.")
        return 0
    execute_approved(args.config, args.freeze_commit, args.amendment_freeze_commit)
    print("Approved locked-score outputs promoted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
