#!/usr/bin/env python
"""Run the future additive Session 7 post-hoc integrity audit."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Callable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import posthoc_final_audit as audit


def _commit_sha(value: str) -> str:
    """Validate one exact lowercase Git commit identifier."""
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise argparse.ArgumentTypeError("Commit must be exactly 40 lowercase hexadecimal characters.")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Build the fail-closed Session 7 parser without an execution-authorization flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--freeze-commit", type=_commit_sha, required=True)
    parser.add_argument("--amendment-freeze-commit", type=_commit_sha, required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    project_root: Path | None = None,
    git_runner: Callable[..., str] | None = None,
    invocation_clock: Callable[[], str] | None = None,
) -> int:
    """Parse the exact command shape and cross the authority-gated audit pipeline."""
    args = build_parser().parse_args(argv)
    kwargs = {"project_root": project_root, "invocation_clock": invocation_clock}
    if git_runner is not None:
        kwargs["git_runner"] = git_runner
    audit.execute_audit(
        args.config,
        args.freeze_commit,
        args.amendment_freeze_commit,
        **kwargs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
