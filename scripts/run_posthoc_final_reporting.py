#!/usr/bin/env python
"""Run the explicitly approved Session 6 reporting-closure pipeline."""

from __future__ import annotations

import argparse
import re
import shlex
import sys
from pathlib import Path
from typing import Callable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import posthoc_final_reporting as reporting


def _commit_sha(value: str) -> str:
    """Validate one exact lowercase Git commit identifier."""
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise argparse.ArgumentTypeError("Commit must be exactly 40 lowercase hexadecimal characters.")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Build the fail-closed Session 6 command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--freeze-commit", type=_commit_sha, required=True)
    parser.add_argument("--amendment-freeze-commit", type=_commit_sha, required=True)
    parser.add_argument(
        "--execute-approved",
        action="store_true",
        required=True,
        help="Required explicit confirmation for the later authorized participant-capable execution path.",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    project_root: Path | None = None,
    git_runner: Callable[..., str] | None = None,
    invocation_clock: Callable[[], str] | None = None,
) -> int:
    """Parse the command and cross the authority-gated reporting pipeline."""
    argument_list = list(argv) if argv is not None else sys.argv[1:]
    args = build_parser().parse_args(argument_list)
    command = shlex.join(["python", "scripts/run_posthoc_final_reporting.py", *argument_list])
    kwargs = {
        "project_root": project_root,
        "invocation_clock": invocation_clock,
        "command": command,
    }
    if git_runner is not None:
        kwargs["git_runner"] = git_runner
    reporting.execute_approved(
        args.config,
        args.freeze_commit,
        args.amendment_freeze_commit,
        **kwargs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
