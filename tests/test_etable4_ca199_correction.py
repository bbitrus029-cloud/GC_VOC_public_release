"""Synthetic regression tests; never invoke a historical real-data generator."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis import cohort_description as legacy
from src.analysis import etable4_ca199_correction as correction


@pytest.mark.parametrize('value,expected', [(' 12,5 ', 12.5), ('>37,0', 37.0), ('<0,1', 0.1),
                                            (' 42.5 ', 42.5), (42.5, 42.5)])
def test_decimal_comma_and_bounds(value: object, expected: float) -> None:
    """CA19-9 uses decimal replacement and preserves censoring-bound handling."""
    assert legacy._parse_truncated_numeric(value, decimal_comma=True) == expected


@pytest.mark.parametrize('value', [None, np.nan, '', ' ', '-', '/', '?', 'private-token', '1,2,3'])
def test_missing_and_sanitized_diagnostics(value: object, caplog: pytest.LogCaptureFixture) -> None:
    """Missing/invalid values stay missing without disclosing the raw value."""
    assert legacy._parse_truncated_numeric(value, decimal_comma=True) is None
    assert 'private-token' not in caplog.text
    assert '1,2,3' not in caplog.text


def test_marker_specific_loader(tmp_path: Path) -> None:
    """Only CA19-9 accepts the decimal comma through the existing loader."""
    path = tmp_path / 'synthetic.csv'
    pd.DataFrame({'sample_id': [1, 2], **{key: ['12,5', '>38'] for key in legacy.TUMOR_MARKER_COLUMNS}}).to_csv(
        path, index=False)
    result = legacy._load_tumor_markers(path)
    assert result['ca199'].tolist() == [12.5, 38.0]
    assert result.loc[0, ['cea', 'ca724']].isna().all()
    assert result.loc[1, ['cea', 'ca724']].eq(38.0).all()


def test_threshold_and_denominators() -> None:
    """Exactly 37 is not elevated and missing measurements leave the denominator."""
    values = pd.Series([36.9, 37.0, 37.1, np.nan])
    indicator = legacy._make_elevated_indicator(values, 37.0)
    assert indicator.iloc[:3].tolist() == [0, 0, 1]
    assert pd.isna(indicator.iloc[3])
    assert legacy._format_count_percent(indicator, 1) == '1/3 (33.3%)'
    assert legacy._format_count_percent(values, 'missing') == '1/4 (25.0%)'
    assert correction._percent(1, 16) == '6'
    assert correction._percent(1, 80) == '1'
    assert correction._percent(1, 160) == '0.6'


def test_summaries_reuse_existing_methods() -> None:
    """Synthetic groups exercise quantiles, tests, SMDs, and deterministic summaries."""
    categories = pd.Series(np.repeat(np.arange(6), 5))
    values = pd.Series([1., 5., 37., 40., np.nan] * 6)
    values.iloc[0] = 2.0
    summaries, rows = correction._summarize(values, categories, 'legacy')
    repeated, repeated_rows = correction._summarize(values, categories, 'legacy')
    assert summaries == repeated
    pd.testing.assert_frame_equal(rows, repeated_rows)
    assert summaries[0]['total'] == 30
    assert summaries[0]['available'] == 24
    assert summaries[0]['elevated'] == 6
    assert summaries[0]['median'] == values.median()
    assert summaries[0]['q1'] == values.quantile(0.25)
    assert summaries[0]['q3'] == values.quantile(0.75)
    assert summaries[0]['continuous_p'] == legacy._omnibus_p_continuous(
        [values.loc[categories.eq(i)].to_numpy() for i in range(6)])
    assert list(rows.columns) == ['variable', *correction.COLUMNS, 'p value', '|SMD|']


def test_exclusive_output_protection(tmp_path: Path) -> None:
    """Occupied outputs remain byte-identical when creation is attempted."""
    path = tmp_path / 'existing.txt'
    path.write_text('protected')
    with pytest.raises(FileExistsError):
        correction._write_exclusive(path, 'replacement')
    assert path.read_text() == 'protected'


def test_entrypoint_collision_precedes_access(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An occupied package prevents even input identity or participant access."""
    date = correction.datetime.now(correction.timezone.utc).strftime('%Y-%m-%d')
    target = tmp_path / f'outputs/analysis/etable4_ca199_decimal_correction_{date}'
    target.mkdir(parents=True)
    monkeypatch.setattr(correction, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(correction, '_load_minimal', lambda: pytest.fail('must not load'))
    with pytest.raises(FileExistsError, match='occupied'):
        correction.run_correction()
    assert list(target.iterdir()) == []


def test_identity_failure_precedes_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A changed frozen source stops the correction before data access or writing."""
    path = tmp_path / 'synthetic.csv'
    path.write_text('changed')
    monkeypatch.setattr(correction, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(correction, 'INPUT_HASHES', {'synthetic.csv': '0' * 64})
    monkeypatch.setattr(correction.subprocess, 'run', lambda *args, **kwargs: type('Result', (), {'returncode': 1})())
    monkeypatch.setattr(correction, '_load_minimal', lambda: pytest.fail('must not load'))
    with pytest.raises(ValueError, match='identity mismatch'):
        correction.run_correction()
    assert path.read_text() == 'changed'


def test_privacy_readback_rejects_raw_value(tmp_path: Path) -> None:
    """Raw-value leakage fails without echoing the matched private value."""
    name = 'synthetic.txt'
    (tmp_path / name).write_text('private-token')
    with pytest.raises(ValueError, match='Private raw-value leak detected') as error:
        correction._validate_payload(tmp_path, {name: 'private-token'}, ['private-token'])
    assert 'private-token' not in str(error.value)


def test_nonestimable_legacy_conventions() -> None:
    """Existing non-estimability behavior remains unchanged."""
    assert np.isnan(legacy._omnibus_p_continuous([np.array([1.]), np.array([2., 3.])]))
    assert np.isnan(legacy._omnibus_p_categorical(np.zeros((6, 2)))[0])
