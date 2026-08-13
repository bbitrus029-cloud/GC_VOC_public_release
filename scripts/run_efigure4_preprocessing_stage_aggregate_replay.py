"""CLI for the Gate-2-authorized eFigure 4 aggregate preprocessing replay."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from src.analysis.efigure4_preprocessing_stage_aggregate_replay import (
    FailureCode,
    ReplayExecutionError,
    run_authorized_replay,
)
from src.config import PROJECT_ROOT


CONTRACT_PATH = PROJECT_ROOT / "configs" / "efigure4_preprocessing_stage_aggregate_replay.yaml"


def _build_parser() -> argparse.ArgumentParser:
    """Build the explicit-authorization command-line parser."""
    parser = argparse.ArgumentParser(description="Run the aggregate-only eFigure 4 preprocessing-stage replay.")
    parser.add_argument(
        "--authorization",
        required=True,
        help="Exact Mandatory Gate 2 authorization line.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the aggregate replay behind a sanitized CLI error boundary.

    Args:
        argv: Optional argument sequence for tests or programmatic invocation.

    Returns:
        Process exit status: zero on success, one on any sanitized failure.
    """
    args = _build_parser().parse_args(argv)
    try:
        result = run_authorized_replay(
            authorization=str(args.authorization),
            project_root=Path(PROJECT_ROOT),
            contract_path=CONTRACT_PATH,
        )
    except ReplayExecutionError as exc:
        failure_code = exc.failure_code
        if not isinstance(failure_code, FailureCode):
            failure_code = FailureCode.UNCLASSIFIED_FAILURE
        print(
            f"ERROR: Aggregate replay execution failed. FAILURE_CODE={failure_code.value}",
            file=sys.stderr,
        )
        return 1
    except Exception:
        print(
            "ERROR: Aggregate replay execution failed. FAILURE_CODE=UNCLASSIFIED_FAILURE",
            file=sys.stderr,
        )
        return 1

    print("Aggregate replay execution completed.")
    print(f"Final output: {result['final_output_directory']}")
    print(f"Aggregate rows: {result['aggregate_row_count']}")
    print(f"Manifest SHA-256: {result['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
