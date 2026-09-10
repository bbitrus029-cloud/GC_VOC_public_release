"""Bounded, aggregate-only retrospective eTable 4 parsing correction."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_EVEN
import hashlib
import importlib.metadata
from io import StringIO
import json
import logging
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal

from src.analysis import cohort_description as legacy
from src.config import PROJECT_ROOT

LOGGER = logging.getLogger(__name__)
CONTRACT = 'docs/plans/visualization_v2/object_contracts/etable4_six_category_characteristics.contract.md'
EF16 = 'docs/plans/visualization_v2/object_contracts/efigure16_voc_tumor_marker_comparison.contract.md'
BASELINE_CONTRACT_HASH = '0b81dc00dbd97c4202e2ece486196c3745d3189c3ad0a3497b1afdfdf6a2b7f1'
INPUT_HASHES = {
    'data/raw/clinical_data/metadata.csv': '58b7c08d9eea834e22d9aebe0e19049f5ca313c384262b3cba21ce9763523aa9',
    'data/processed/sample_index.csv': '3e31635c37c3cc142eade488c00a240ec5487175b531f9ed94837946ddcb403f',
    'data/processed/splits/train_idx.npz': '82f64e88f29228faa7dc93cf507013f14db32b80bf9d20e417c53f1ae6bb1131',
    'data/processed/splits/test_idx.npz': '7fc60acfc54f27e77f60a9813ca829fd61f174a8d3e4ab59aac18f5e05c981c3',
    'configs/modeling_global.yaml': 'c4520df8040adfebdfd1f6ae89376be5484ae3dfeda0cfd2481103df323a2e54',
    'outputs/analysis/table_two_six_categories.csv': '46820f1f5c583e7dcca9c3d76037d5bf1bbd1feb43f689d0edddab83c960b546',
    'outputs/analysis/table_two_six_categories.md': '1f555f3bb918e3457719bf170916add23999a17af0c818ec995d97d6166aa34d',
    'outputs/analysis/stage7_part_c_audit.json': '84cddc888bb7260d42b5cbb09d24dea28a418685d46cf9c2762aadcaf4ebfc03',
    'outputs/analysis/tumor_marker_selection_bias.csv': 'a41fbb08f93d28528919c2916dff5cf12c323a4a19b86863db98dc04819708a2',
    'outputs/analysis/tumor_marker_lr_comparison_minirocket_ridge.csv':
        'a2363085473a6e427e045e9dc19383c4f94d253a61f9a392be4d801b36ff7a51',
    'outputs/analysis/voc_marker_fusion_minirocket_ridge.csv':
        'bc10ce28b06c8d1f01dd5d70a14ee393fc6fb86a29101774c4b790a3d48083af',
    'src/analysis/stage7_part_c_common.py': '8dfb3800754d45bac8c5bf8d67223863c96dacc023a1307a40befb237463b237',
    'scripts/run_cohort_description.py': '55376ec044d8270b2560b753d73de130f0197e76d7a60cd7933d64ee443f2981',
}
CODE_PATHS = [
    'src/analysis/cohort_description.py', 'src/analysis/etable4_ca199_correction.py',
    'scripts/run_etable4_ca199_correction.py', 'tests/test_cohort_description.py',
    'tests/test_etable4_ca199_correction.py', 'src/analysis/data_split.py', 'environment.yml',
]
GROUPS = ['Overall', 'HC', 'BGL', 'LGIN', 'HGIN', 'EGC', 'AGC']
TOTALS = [972, 158, 352, 49, 71, 122, 220]
COLUMNS = ['Total (N=972)', *[f'{g} (n={n})' for g, n in zip(GROUPS[1:], TOTALS[1:])]]
ROWS = ['CA19-9, U/mL (median [IQR])', 'CA19-9 missing, n (%)', 'CA19-9 elevated (>37 U/mL), n (%)']
STAT_COLUMNS = [
    'parsing', 'group', 'total', 'available', 'missing', 'elevated', 'elevated_denominator', 'median', 'q1', 'q3',
    'continuous_p', 'continuous_smd', 'continuous_statistic', 'continuous_smd_pair',
    'missing_p', 'missing_smd', 'missing_statistic', 'missing_expected_below_5', 'missing_smd_group',
    'elevated_p', 'elevated_smd', 'elevated_statistic', 'elevated_expected_below_5', 'elevated_smd_group',
]


def _require(condition: bool, message: str) -> None:
    """Fail with an allowlisted diagnostic that cannot contain private values."""
    if not condition:
        raise ValueError(message)


def _hash(path: Path) -> str:
    """Hash bytes without decoding participant information."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(value: Any) -> str:
    """Serialize deterministic JSON, rejecting nonfinite values."""
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'


def _git(*args: str) -> str:
    """Read Git provenance without mutations."""
    return subprocess.run(['git', *args], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _identities(paths: list[str]) -> dict[str, str]:
    """Return project-relative file identities."""
    return {p: _hash(PROJECT_ROOT / p) for p in paths}


def _load_minimal() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Load only authorized alignment and marker fields, retaining joins in memory."""
    index = pd.read_csv(PROJECT_ROOT / 'data/processed/sample_index.csv',
                        usecols=['sample_id', 'label', 'disease_category', 'in_final_output'])
    _require(index['sample_id'].notna().all() and index['sample_id'].is_unique, 'Invalid sample-index identities.')
    eligible = index['in_final_output'].astype('string').str.lower()
    _require(eligible.isin(['true', 'false']).all(), 'Invalid eligibility encoding.')
    index = index.loc[eligible.eq('true')].copy()
    _require(pd.api.types.is_integer_dtype(index['sample_id']), 'Noninteger alignment identities.')
    index = index.sort_values('sample_id').reset_index(drop=True)
    raw_category = 'disease_category(0HC1BGL2LGIN3HGIN4EGC5AGC)'
    metadata = pd.read_csv(PROJECT_ROOT / 'data/raw/clinical_data/metadata.csv', encoding='utf-8-sig',
                           usecols=['sample_id', 'label', raw_category, *legacy.TUMOR_MARKER_COLUMNS])
    _require(metadata['sample_id'].notna().all() and metadata['sample_id'].is_unique, 'Invalid metadata identities.')
    _require(pd.api.types.is_integer_dtype(metadata['sample_id']), 'Noninteger metadata identities.')
    frame = index.merge(metadata, on='sample_id', how='left', validate='one_to_one', indicator=True,
                        suffixes=('', '_metadata'))
    _require(frame['_merge'].eq('both').all(), 'Incomplete final-cohort alignment.')
    _require(frame['sample_id'].equals(index['sample_id']), 'Numeric ordering changed during alignment.')
    _require(frame['label'].equals(frame['label_metadata']), 'Endpoint alignment failure.')
    _require(frame['disease_category'].equals(frame[raw_category]), 'Disease-category alignment failure.')
    _require(frame['label'].isin([0, 1]).all() and frame['disease_category'].isin(range(6)).all(),
             'Invalid endpoint/category domain.')
    _require(np.array_equal(frame['label'].to_numpy(), (frame['disease_category'] >= 3).astype(int)),
             'Endpoint/category disagreement.')
    counts = [len(frame), *[int(frame['disease_category'].eq(i).sum()) for i in range(6)]]
    _require(counts == TOTALS, 'Cohort reconciliation failure.')
    splits = {}
    for partition in ['train', 'test']:
        key = f'{partition}_idx'
        with np.load(PROJECT_ROOT / f'data/processed/splits/{key}.npz', allow_pickle=False) as archive:
            _require(archive.files == [key], 'Unexpected locked split schema.')
            values = archive[key]
        _require(values.ndim == 1 and np.issubdtype(values.dtype, np.integer), 'Invalid split dtype or shape.')
        _require(len(np.unique(values)) == len(values) and ((values >= 0) & (values < len(frame))).all(),
                 'Invalid split membership.')
        splits[partition] = values
    combined = np.concatenate(list(splits.values()))
    _require(np.array_equal(np.sort(combined), np.arange(len(frame))), 'Splits overlap or omit cohort members.')
    return frame, splits


def _summarize(values: pd.Series, categories: pd.Series, mode: str) -> tuple[list[dict], pd.DataFrame]:
    """Recompute only the three existing CA19-9 rows using frozen helper conventions."""
    masks = [np.ones(len(values), dtype=bool), *[categories.eq(i).to_numpy() for i in range(6)]]
    group_values = {g: values.loc[m] for g, m in zip(GROUPS[1:], masks[1:])}
    p_cont = legacy._omnibus_p_continuous([v.to_numpy() for v in group_values.values()], variable_name='CA19-9')
    smd_cont, pair = legacy._compute_max_pairwise_continuous_smd(group_values)
    global_stats = {
        'continuous_p': p_cont, 'continuous_smd': smd_cont,
        'continuous_statistic': float(kruskal(*[v.dropna().to_numpy() for v in group_values.values()]).statistic),
        'continuous_smd_pair': '|'.join(pair),
    }
    elevated = legacy._make_elevated_indicator(values, legacy.CA199_UPPER_NORMAL)
    for name, indicator in [('missing', values.isna().astype(int)), ('elevated', elevated)]:
        groups = {g: indicator.loc[m] for g, m in zip(GROUPS[1:], masks[1:])}
        table = legacy._build_binary_contingency_table(groups)
        p_value, low_cells = legacy._omnibus_p_categorical(table)
        smd, smd_group = legacy._compute_max_one_vs_rest_binary_smd(groups)
        global_stats.update({f'{name}_p': p_value, f'{name}_smd': smd,
                             f'{name}_statistic': float(chi2_contingency(table).statistic),
                             f'{name}_expected_below_5': low_cells, f'{name}_smd_group': smd_group})
    _require(all(np.isfinite(v) for v in global_stats.values() if isinstance(v, float)),
             'Unexpected non-estimable correction statistic.')
    summaries = []
    rows = [{ 'variable': label} for label in ROWS]
    for group, column, mask in zip(GROUPS, COLUMNS, masks):
        subset = values.loc[mask]
        clean = subset.dropna()
        summaries.append({
            'parsing': mode, 'group': group, 'total': len(subset), 'available': len(clean),
            'missing': int(subset.isna().sum()), 'elevated': int((clean > legacy.CA199_UPPER_NORMAL).sum()),
            'elevated_denominator': len(clean), 'median': float(clean.median()),
            'q1': float(clean.quantile(0.25)), 'q3': float(clean.quantile(0.75)),
            **(global_stats if group == 'Overall' else {}),
        })
        rows[0][column] = legacy._format_median_iqr(subset, decimals=1)
        rows[1][column] = legacy._format_count_percent(subset, target_value='missing')
        rows[2][column] = legacy._format_count_percent(elevated.loc[mask], target_value=1)
    for row, name in zip(rows, ['continuous', 'missing', 'elevated']):
        row['p value'] = legacy._format_p_value(global_stats[f'{name}_p'])
        row['|SMD|'] = legacy._format_smd_value(global_stats[f'{name}_smd'])
    return summaries, pd.DataFrame(rows, columns=['variable', *COLUMNS, 'p value', '|SMD|'])


def _percent(numerator: int, denominator: int) -> str:
    """Format an exact count ratio with the approved denominator-specific half-even rule."""
    quantum = Decimal('1') if denominator < 100 else Decimal('0.1')
    return str((Decimal(100) * Decimal(numerator) / Decimal(denominator)).quantize(quantum, ROUND_HALF_EVEN))


def visible_cells(summaries: list[dict]) -> dict[str, list[str]]:
    """Derive approved CA19-9 display cells from unrounded aggregate evidence.

    Args:
        summaries: Seven ordered, corrected group summaries.
    Returns:
        Availability, median/IQR, and elevated display strings in column order.
    """
    result = {'available': [], 'median': [], 'elevated': []}
    for row in summaries:
        a, n, e = row['available'], row['total'], row['elevated']
        result['available'].append(f'{a}/{n} ({_percent(a, n)})')
        quantiles = [str(Decimal(str(row[k])).quantize(Decimal('0.1'), ROUND_HALF_EVEN))
                     for k in ['median', 'q1', 'q3']]
        result['median'].append(f'{quantiles[0]} ({quantiles[1]}–{quantiles[2]})')
        result['elevated'].append(f'{e}/{a} ({_percent(e, a)})')
    return result


def _payload() -> tuple[dict[str, str], list[str]]:
    """Compute aggregate evidence without returning participant-level objects."""
    frame, splits = _load_minimal()
    legacy_markers = {}
    corrected_markers = {}
    for raw, alias in legacy.TUMOR_MARKER_COLUMNS.items():
        legacy_markers[alias] = frame[raw].map(legacy._parse_truncated_numeric).astype(float)
        corrected_markers[alias] = frame[raw].map(
            lambda value: legacy._parse_truncated_numeric(value, decimal_comma=alias == 'ca199')).astype(float)
    old, new = legacy_markers['ca199'], corrected_markers['ca199']
    changed = ~(old.eq(new) | (old.isna() & new.isna()))
    recovered = old.isna() & new.notna()
    _require(changed.equals(recovered) and int(changed.sum()) == 1, 'Unexpected parsing changes.')
    _require(frame.loc[changed, 'disease_category'].eq(4).all(), 'Unexpected recovered disease group.')
    _require(int(changed.iloc[splits['train']].sum()) == 1, 'Unexpected recovered partition.')
    for marker in ['cea', 'ca724']:
        _require(legacy_markers[marker].equals(corrected_markers[marker]), 'Other marker parsing changed.')
    raw_ca199 = frame[next(k for k, v in legacy.TUMOR_MARKER_COLUMNS.items() if v == 'ca199')]
    stage7_values = pd.to_numeric(raw_ca199.astype('string').str.replace(',', '.', regex=False), errors='coerce')
    _require(np.array_equal(new.to_numpy(), stage7_values.to_numpy(dtype=float), equal_nan=True),
             'Stage 7 decimal convention did not reconcile.')
    reconciliations = {}
    for mode, markers in [('legacy', legacy_markers), ('corrected', corrected_markers)]:
        complete = pd.DataFrame(markers).notna().all(axis=1)
        summary = {'availability': {key: int(value.notna().sum()) for key, value in markers.items()},
                   'complete_cases': int(complete.sum())}
        for partition, indices in splits.items():
            selected = complete.iloc[indices].to_numpy()
            labels = frame['label'].iloc[indices].to_numpy()[selected]
            summary[partition] = {'complete_cases': int(selected.sum()), 'positive': int((labels == 1).sum()),
                                  'negative': int((labels == 0).sum())}
        reconciliations[mode] = summary
    _require(reconciliations['legacy']['availability'] == {'cea': 665, 'ca199': 651, 'ca724': 651},
             'Legacy marker availability mismatch.')
    _require(reconciliations['corrected']['availability'] == {'cea': 665, 'ca199': 652, 'ca724': 651},
             'Corrected marker availability mismatch.')
    for mode, counts in [('legacy', [650, 515, 135]), ('corrected', [651, 516, 135])]:
        r = reconciliations[mode]
        _require([r['complete_cases'], r['train']['complete_cases'], r['test']['complete_cases']] == counts,
                 'Complete-case reconciliation failure.')
    frozen = json.loads((PROJECT_ROOT / 'outputs/analysis/stage7_part_c_audit.json').read_text())['tm_subset_sizes']
    for partition in ['train', 'test']:
        actual = reconciliations['corrected'][partition]
        for key, suffix in [('complete_cases', 'total'), ('positive', 'pos'), ('negative', 'neg')]:
            _require(actual[key] == frozen[f'tm_{partition}_subset_n_{suffix}'], 'Frozen subset mismatch.')
    _require(reconciliations['corrected']['train'] == {'complete_cases': 516, 'positive': 293, 'negative': 223}
             and reconciliations['corrected']['test'] == {'complete_cases': 135, 'positive': 76, 'negative': 59}
             and frozen['full_cohort_n_complete_case'] == 651, 'Frozen endpoint counts mismatch.')
    old_stats, old_rows = _summarize(old, frame['disease_category'], 'legacy')
    new_stats, new_rows = _summarize(new, frame['disease_category'], 'corrected')
    _require(old_stats[5]['available'] == 107 and new_stats[5]['available'] == 108, 'EGC availability mismatch.')
    original = pd.read_csv(PROJECT_ROOT / 'outputs/analysis/table_two_six_categories.csv',
                           dtype=str, keep_default_na=False)
    _require(list(original.columns) == list(old_rows.columns), 'Historical column schema mismatch.')
    selected = original['variable'].isin(ROWS)
    _require(original.loc[selected].reset_index(drop=True).equals(old_rows), 'Legacy CA19-9 source reproduction failed.')
    corrected = original.copy()
    corrected.loc[selected, :] = new_rows.to_numpy()
    changes = []
    for i in original.index:
        for column in original.columns:
            if original.at[i, column] != corrected.at[i, column]:
                _require(original.at[i, 'variable'] in ROWS, 'Unaffected source cell changed.')
                changes.append({'variable': original.at[i, 'variable'], 'column': column,
                                'before': original.at[i, column], 'after': corrected.at[i, column]})
    _require(original.loc[~selected].equals(corrected.loc[~selected]), 'Unaffected row change.')
    statistics = pd.DataFrame(old_stats + new_stats).reindex(columns=STAT_COLUMNS)
    audit = {
        'classification': 'retrospective_technical_parsing_correction',
        'status': 'implemented_self_checked_pending_independent_review',
        'cohort_counts': dict(zip(GROUPS, TOTALS)), 'reconciliation': reconciliations,
        'recovered_measurements': 1, 'recovered_category': 'EGC', 'recovered_partition': 'training',
        'frozen_complete_case_counts': {k: v for k, v in frozen.items() if k.endswith(('_total', '_pos', '_neg'))
                                       or k == 'full_cohort_n_complete_case'},
        'changed_cells': changes, 'visible_before': visible_cells(old_stats), 'visible_after': visible_cells(new_stats),
        'checks': {k: True for k in ['alignment', 'numeric_order', 'split_disjointness_and_coverage',
                                    'label_category_agreement', 'legacy_rows_reproduced_all_columns',
                                    'only_explained_recovery', 'stage7_normalization_reconciled',
                                    'unaffected_cells_preserved', 'all_seven_groups_recomputed']},
    }
    payload = {
        'table_two_six_categories_corrected.csv': corrected.to_csv(index=False),
        'table_two_six_categories_corrected.md': (
            '# Corrected six-category descriptive evidence\n\n'
            'Retrospective technical CA19-9 decimal-comma parsing correction; pending independent review.\n'
            'Historical table_two_six_categories files remain immutable. Only the three CA19-9 rows were recomputed.\n'
            'See correction_audit.json, ca199_correction_statistics.csv and manifest.json for provenance.\n'
            'Available measurements; no imputation; elevated means strictly >37 U/mL. Median and linear Q1/Q3; '
            'Kruskal-Wallis, chi-square, maximum pairwise continuous and one-versus-rest binary absolute SMD.\n'
            'P values are descriptive and unadjusted, without significance claims.\n\n'
            + legacy._render_markdown_table(corrected) + '\n'),
        'ca199_correction_statistics.csv': statistics.to_csv(index=False, float_format='%.17g'),
        'correction_audit.json': _json(audit),
    }
    # Identity privacy is enforced by the aggregate schema, not ambiguous numeric token matching.
    # The recovered raw string is checked in memory and is never persisted.
    forbidden = [str(value).strip() for value in raw_ca199.loc[changed]]
    readback = pd.read_csv(StringIO(payload['ca199_correction_statistics.csv']), float_precision='round_trip')
    numeric = statistics.select_dtypes(include='number').columns
    _require(np.array_equal(statistics[numeric].to_numpy(), readback[numeric].to_numpy(), equal_nan=True),
             'Full-precision statistics readback mismatch.')
    return payload, forbidden


def _validate_payload(directory: Path, payload: dict[str, str], forbidden: list[str]) -> None:
    """Read back each aggregate file and enforce schemas, exact bytes, and privacy."""
    for name, expected in payload.items():
        text = (directory / name).read_text()
        _require(text == expected, 'Output readback mismatch.')
        _require('/home/' not in text and not re.search(r'\b\d{4}-\d{2}-\d{2}\b', text),
                 'Prohibited path or date in scientific payload.')
        _require(not any(re.search(r'(?<![\w.])' + re.escape(value) + r'(?![\w.])', text)
                         for value in forbidden), 'Private raw-value leak detected.')
        _require(not any(key in text.lower() for key in ['sample_id', 'collection_date', 'prediction', 'breath_file']),
                 'Prohibited field in scientific payload.')
    frame = pd.read_csv(directory / 'table_two_six_categories_corrected.csv', dtype=str, keep_default_na=False)
    _require(frame.shape == (25, 10) and frame['variable'].isin(ROWS).sum() == 3, 'Corrected table shape failure.')
    stats = pd.read_csv(directory / 'ca199_correction_statistics.csv', float_precision='round_trip')
    _require(list(stats.columns) == STAT_COLUMNS and len(stats) == 14, 'Statistics schema failure.')
    _require(stats['group'].tolist() == GROUPS * 2 and stats['parsing'].tolist() == ['legacy'] * 7 + ['corrected'] * 7,
             'Statistics ordering failure.')
    for key in ['total', 'available', 'missing', 'elevated', 'elevated_denominator']:
        _require(pd.api.types.is_integer_dtype(stats[key]), 'Count dtype failure.')
    _require((stats['total'] == stats['available'] + stats['missing']).all()
             and (stats['elevated_denominator'] == stats['available']).all()
             and (stats['elevated'] <= stats['available']).all(), 'Count reconciliation failure.')
    _require((stats['q1'] <= stats['median']).all() and (stats['median'] <= stats['q3']).all(), 'Quantile order failure.')
    for mode in ['legacy', 'corrected']:
        part = stats.loc[stats['parsing'].eq(mode)]
        for key in ['total', 'available', 'missing', 'elevated']:
            _require(part.iloc[0][key] == part.iloc[1:][key].sum(), 'Group sum failure.')
    audit = json.loads((directory / 'correction_audit.json').read_text())
    _require(visible_cells(stats.iloc[7:].to_dict('records')) == audit['visible_after'], 'Display readback failure.')
    _require(legacy._render_markdown_table(frame) in (directory / 'table_two_six_categories_corrected.md').read_text(),
             'Markdown/CSV disagreement.')


def _write_exclusive(path: Path, text: str) -> None:
    """Create a new file without overwriting an existing or unknown target."""
    with path.open('x', encoding='utf-8', newline='') as stream:
        stream.write(text)


def run_correction() -> Path:
    """Execute the authorized correction twice and promote verified aggregate evidence.

    Returns:
        Project-relative correction package path.
    Raises:
        ValueError: An integrity, reconciliation, or privacy gate fails.
        FileExistsError: A destination is already occupied.
    """
    now = datetime.now(timezone.utc)
    date = now.strftime('%Y-%m-%d')
    relative = Path(f'outputs/analysis/etable4_ca199_decimal_correction_{date}')
    destination = PROJECT_ROOT / relative
    erratum = PROJECT_ROOT / f'docs/modeling/etable4_ca199_decimal_correction_{date}.md'
    if destination.exists() or erratum.exists():
        raise FileExistsError('Correction destination occupied; provenance review required.')
    for path in [relative, erratum.relative_to(PROJECT_ROOT)]:
        result = subprocess.run(['git', 'check-ignore', '-q', str(path)], cwd=PROJECT_ROOT)
        _require(result.returncode == 1, 'New correction destination is ignored or ignore check failed.')
    for path, expected in INPUT_HASHES.items():
        _require(_hash(PROJECT_ROOT / path) == expected, 'Immutable input identity mismatch.')
    ef_contract = (PROJECT_ROOT / EF16).read_text()
    for path, expected in INPUT_HASHES.items():
        if path.startswith('outputs/analysis/') and 'table_two' not in path:
            _require(path in ef_contract and expected in ef_contract, 'eFigure 16 source binding mismatch.')
    protected = _identities([*INPUT_HASHES, EF16, 'outputs/analysis/table_one_pos_vs_neg.csv',
                             'outputs/analysis/table_one_pos_vs_neg.md'])
    methods = _identities(CODE_PATHS)
    contract_hash = _hash(PROJECT_ROOT / CONTRACT)
    _require(contract_hash == BASELINE_CONTRACT_HASH, 'Target contract changed; focused authority review required.')
    invocation = 'python scripts/run_etable4_ca199_correction.py'
    provenance = {'branch': _git('branch', '--show-current'), 'head': _git('rev-parse', 'HEAD'),
                  'worktree': _git('status', '--short'), 'staged': _git('diff', '--cached', '--name-status'),
                  'correction_commit': 'pending_user_git_action'}
    specification = (
        f'# eTable 4 CA19-9 decimal correction — {date}\n\n'
        '## Pre-execution specification\n\n'
        'Classification: retrospective technical parsing correction to existing descriptive statistics; '
        'not a newly prespecified or exploratory analysis and not an ET-04 production session. '
        'Authority: the explicit bounded user task and reviewed eTable 4 baseline contract.\n\n'
        'Original execution stripped a leading censoring sign and called float(); a decimal-comma CA19-9 '
        'entry consequently became missing. Correct only CA19-9 by replacing commas with decimal points, '
        'matching the existing Stage 7 convention, while retaining missing and censoring-bound semantics. '
        'CEA and CA72-4 remain unchanged. No source value is edited.\n\n'
        'Before corrected result access: verify identities; minimally load eligibility, numeric alignment ID, '
        'endpoint, disease category and three markers; enforce unique one-to-one cohort alignment and locked splits. '
        'Reproduce all legacy CA19-9 cells, then recompute only its median, missing and elevated rows, '
        'including all seven groups and existing P values/SMDs. Preserve all other cells. '
        'Use pandas linear quantiles, strict >37, existing Kruskal-Wallis/chi-square and SMD helpers. '
        'No confidence intervals, fitting, bootstrap, threshold selection, imputation or randomness. '
        'Two isolated computations must have identical payload bytes. No participant-level persistence or network use.\n\n'
        'Stop on identity, alignment, schema, privacy, collision, legacy reproduction or reconciliation failure. '
        'Completion requires aggregate readback, full-precision statistics, unchanged protected hashes, '
        'correct current bindings and independent review explicitly pending.\n\n'
        f'Pre-execution UTC timestamp: {now.isoformat()}\n\n'
        f'Invocation: `{invocation}`\n\n'
        f'Target contract SHA-256: `{contract_hash}`\n\n'
        'Actual Git provenance:\n```json\n' + _json(provenance) + '```\n\n'
        'Protected pre-execution identities:\n```json\n' + _json(protected) + '```\n\n'
        'Method/environment identities:\n```json\n' + _json(methods) + '```\n')
    _write_exclusive(erratum, specification)
    with tempfile.TemporaryDirectory(prefix='.etable4-correction-', dir=destination.parent) as work:
        root = Path(work)
        runs = []
        for name in ['first', 'second']:
            folder = root / name
            folder.mkdir()
            payload, forbidden = _payload()
            for filename, content in payload.items():
                _write_exclusive(folder / filename, content)
            _validate_payload(folder, payload, forbidden)
            runs.append((folder, payload))
        _require(runs[0][1] == runs[1][1], 'Deterministic scientific payload mismatch.')
        _require(_identities(list(protected)) == protected, 'Protected source changed during execution.')
        _require(_identities(CODE_PATHS) == methods and _hash(PROJECT_ROOT / CONTRACT) == contract_hash,
                 'Relevant method or authority changed during execution.')
        folder, payload = runs[0]
        audit = json.loads(payload['correction_audit.json'])
        outputs = {name: _hash(folder / name) for name in payload}
        manifest = {
            'schema_version': '1.0', 'classification': audit['classification'], 'status': audit['status'],
            'created_at_utc': now.isoformat(), 'invocation': invocation, 'git': provenance,
            'git_at_completion': {'head': _git('rev-parse', 'HEAD'), 'branch': _git('branch', '--show-current')},
            'input_hashes': protected, 'protected_post_hashes': _identities(list(protected)), 'code_hashes': methods,
            'authority': {'contract_path': CONTRACT, 'baseline_sha256': contract_hash,
                          'scope': 'explicit_user_authorized_bounded_correction'},
            'versions': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scipy', 'pytest']},
            'seed': None, 'seed_reason': 'No stochastic operations; locked split arrays read without regeneration.',
            'output_hashes': outputs,
            'schemas': {
                'table_two_six_categories_corrected.csv': {
                    'role': 'current_aggregate_descriptive_source', 'columns': ['variable', *COLUMNS, 'p value', '|SMD|'],
                    'dtypes': 'all strings', 'rows': 25, 'row_order': 'identical to historical source',
                    'missing': 'empty structural cells; legacy NA/em dash where applicable',
                    'units': 'legacy variable labels', 'precision': 'legacy formatted presentation only'},
                'ca199_correction_statistics.csv': {
                    'role': 'full_precision_CA19-9_evidence', 'columns': STAT_COLUMNS, 'rows': 14,
                    'key': ['parsing', 'group'], 'row_order': ['legacy', 'corrected'], 'group_order': GROUPS,
                    'dtypes': 'parsing/group/pair/group-name strings; counts integers; other statistics float64',
                    'missing': 'empty for global tests/SMDs outside Overall; no non-estimable tests observed',
                    'global_statistics': 'Overall rows contain six-category omnibus/SMD results, not Overall-only tests',
                    'precision': '17 significant digits; float64 round-trip readback', 'units': 'CA19-9 U/mL',
                    'threshold': 'strict >37 U/mL', 'quantiles': 'pandas linear interpolation',
                    'confidence_intervals': 'not part of historical method; none computed'},
                'correction_audit.json': {'role': 'aggregate_reconciliation_changed_cells_and_display',
                                           'key_order': list(audit), 'missing': 'none', 'dtypes': 'typed JSON'},
                'table_two_six_categories_corrected.md': {'role': 'sanitized CSV-derived companion'},
            },
            'verification': {'deterministic_isolated_runs': 2, 'identical_scientific_bytes': True,
                             'all_outputs_read_back': True, 'protected_pre_post_equal': True,
                             'privacy_schema_and_in_memory_raw_value_checks': True,
                             'network_used': False, 'git_mutation': False, 'ET04_production': False,
                             'independent_review': 'pending'},
        }
        _write_exclusive(folder / 'manifest.json', _json(manifest))
        _require(json.loads((folder / 'manifest.json').read_text()) == manifest, 'Manifest readback failure.')
        _require(not destination.exists(), 'Destination occupied before promotion.')
        folder.rename(destination)
    result_text = ('\n## Execution and verification\n\n'
                   'Implemented and self-checked; independent review pending. Correction commit remains pending '
                   'the user’s Git action. Historical bytes and eFigure 16 contract/sources are unchanged. '
                   'Both isolated real computations produced identical scientific payloads; all files were read back. '
                   'The dangerous legacy real-data generator test was excluded.\n\n'
                   'Machine-derived reconciliation:\n```json\n' + _json(audit['reconciliation']) + '```\n\n'
                   'CA19-9 availability is marker-specific; all-three-marker complete cases use the intersection. '
                   'Frozen Stage 7 complete-case counts reconcile; eFigure 16 needs no statistical correction.\n\n'
                   'Machine-derived changed historical cells:\n```json\n' + _json(audit['changed_cells']) + '```\n\n'
                   f'Correction package: `{relative.as_posix()}`.\n'
                   f'Manifest SHA-256: `{_hash(destination / "manifest.json")}`.\n'
                   'No rendering, release, model access or external transmission occurred. '
                   'Current eTable 4 binding registration is a subsequent documentation step; ET-04 stays unauthorized.\n')
    with erratum.open('a', encoding='utf-8') as stream:
        stream.write(result_text)
    _require(erratum.read_text() == specification + result_text, 'Erratum readback failure.')
    LOGGER.info('CA19-9 correction verified; independent review pending.')
    return relative
