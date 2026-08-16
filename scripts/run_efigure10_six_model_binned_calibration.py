"""CLI entry point for the eFigure 10 six-model calibration aggregate."""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Sequence
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_ROOT_TEXT = str(PROJECT_ROOT)
sys.path[:] = [entry for entry in sys.path if entry != _PROJECT_ROOT_TEXT]
sys.path.insert(0, _PROJECT_ROOT_TEXT)

from src.analysis.efigure10_six_model_binned_calibration import run_analysis  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the aggregate and emit only sanitized post-validation status."""
    args = build_parser().parse_args(argv)
    final_root = PROJECT_ROOT / "outputs/analysis/efigure10_six_model_binned_calibration"
    try:
        final_paths = run_analysis(args.config, project_root=PROJECT_ROOT)
        expected_paths = tuple(
            final_root / name
            for name in ("calibration_points.csv", "audit.json", "manifest.json", "report.md")
        )
        if tuple(final_paths) != expected_paths:
            raise ValueError("Returned final artifact inventory is invalid")
        output_lines = []
        for path in final_paths:
            if path.is_symlink() or not path.is_file():
                raise ValueError("Returned final artifact is not a regular nonsymlink file")
            relative = path.resolve(strict=True).relative_to(PROJECT_ROOT.resolve(strict=True)).as_posix()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            output_lines.append(f"OUTPUT: {relative} SHA256: {digest}")
    except Exception:
        status = "INCOMPLETE_INTEGRITY" if final_root.exists() else "FAILED_BEFORE_PROMOTION"
        print(f"P01_STATUS: {status}")
        print("P01_ERROR: sanitized aggregate integrity failure")
        return 1
    print("P01_STATUS: VALIDATED")
    print("ANALYSIS_ID: efigure10_six_model_binned_calibration")
    for line in output_lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
