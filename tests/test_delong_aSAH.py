"""Ground-truth DeLong anchor against the pROC aSAH fixture.

Fixture metadata:
- USER-VERIFIED in Session 3 on 2026-05-16T08:57:07Z.
- R version: R version 4.6.0 (2026-04-24).
- pROC version: 1.19.0.1.
- Fixture generation command: the Session 3 §9.1 `Rscript --vanilla` block that
  writes `tests/fixtures/aSAH.csv` and prints the aSAH anchor statistics.

Note:
- `roc.test(roc_s100b, roc_ndka)` in pROC reports `Z = +1.3908` because s100b
  has the larger AUC. The operational plan records `-1.3908`, which corresponds
  to the opposite score-ordering convention. The test therefore anchors `abs(z)`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.statistical_tests import delong_roc_test


def test_delong_reproduces_pROC_asah_anchor() -> None:
    """DeLong should reproduce the pROC aSAH anchor numerics."""
    fixture_path = Path("tests/fixtures/aSAH.csv")
    dataframe = pd.read_csv(fixture_path)
    y_true = (dataframe["outcome"] == "Poor").astype(int).to_numpy()

    result = delong_roc_test(
        y_true=y_true,
        score_a=dataframe["s100b"].to_numpy(dtype=float),
        score_b=dataframe["ndka"].to_numpy(dtype=float),
    )

    assert result["auc_a"] == pytest.approx(0.7314, abs=1e-3)
    assert result["auc_b"] == pytest.approx(0.6120, abs=1e-3)
    assert abs(result["z"]) == pytest.approx(1.3908, abs=1e-2)
    assert result["p_two_sided"] == pytest.approx(0.1643, abs=1e-3)
