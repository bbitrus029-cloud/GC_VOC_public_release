"""Ground-truth Jonckheere-Terpstra anchor against the npordtests jdata fixture.

Fixture metadata:
- USER-VERIFIED in Session 3 on 2026-05-16T08:57:07Z.
- R version: R version 4.6.0 (2026-04-24).
- npordtests version: 1.2.
- Fixture generation command: the Session 3 §9.1 `Rscript --vanilla` block that
  writes `tests/fixtures/jdata.csv` and prints the jdata anchor statistics.

Note:
- The canonical npordtests call is `JtTest(Y ~ X, data = jdata)`.
- The function has no `alternative` parameter; its printed
  `Asymp. p-value` is **upper-tail one-sided** (R source:
  `p.value <- 1 - pnorm(Z)`). The anchor below therefore tests
  `p_one_sided_upper`, not `p_two_sided`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.statistical_tests import jonckheere_terpstra


def test_jonckheere_terpstra_reproduces_npordtests_jdata_anchor() -> None:
    """JT should reproduce the verified npordtests jdata anchor."""
    fixture_path = Path("tests/fixtures/jdata.csv")
    dataframe = pd.read_csv(fixture_path)

    result = jonckheere_terpstra(
        values=dataframe["Y"].to_numpy(dtype=float),
        ordered_groups=dataframe["X"].to_numpy(dtype=int),
    )

    assert result["J"] == 71
    assert result["z"] == pytest.approx(2.148, abs=1e-2)
    assert result["p_one_sided_upper"] == pytest.approx(0.0159, abs=1e-3)
    # Sanity check: two-sided is exactly 2x the one-sided here because z > 0.
    assert result["p_two_sided"] == pytest.approx(2 * 0.0158618, abs=1e-3)
