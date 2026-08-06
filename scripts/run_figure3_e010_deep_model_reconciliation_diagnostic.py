#!/usr/bin/env python3
"""Run the approved aggregate-only E010 deep-model diagnostic."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.figure3_e010_deep_model_reconciliation_diagnostic import run_diagnostic


def _parse_args() -> argparse.Namespace:
    """Parse the sole frozen configuration argument."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    """Execute the diagnostic and print only aggregate output provenance."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    try:
        result = run_diagnostic(args.config)
    except Exception as exc:
        logging.error("Diagnostic failed closed: %s", exc)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
