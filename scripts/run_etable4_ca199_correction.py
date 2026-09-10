"""Run only the authorized additive eTable 4 CA19-9 correction."""
from __future__ import annotations

import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.etable4_ca199_correction import run_correction


def main() -> int:
    """Execute with sanitized diagnostics and no legacy bulk-generator calls.

    Returns:
        Zero after verified promotion, one on a failed gate.
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    try:
        run_correction()
    except Exception:
        logging.error('Correction stopped; no private exception details emitted. Inspect the bounded gates locally.')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
