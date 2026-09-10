# eTable 4 CA19-9 decimal correction — 2026-09-10

## Pre-execution specification

Classification: retrospective technical parsing correction to existing descriptive statistics; not a newly prespecified or exploratory analysis and not an ET-04 production session. Authority: the explicit bounded user task and reviewed eTable 4 baseline contract.

Original execution stripped a leading censoring sign and called float(); a decimal-comma CA19-9 entry consequently became missing. Correct only CA19-9 by replacing commas with decimal points, matching the existing Stage 7 convention, while retaining missing and censoring-bound semantics. CEA and CA72-4 remain unchanged. No source value is edited.

Before corrected result access: verify identities; minimally load eligibility, numeric alignment ID, endpoint, disease category and three markers; enforce unique one-to-one cohort alignment and locked splits. Reproduce all legacy CA19-9 cells, then recompute only its median, missing and elevated rows, including all seven groups and existing P values/SMDs. Preserve all other cells. Use pandas linear quantiles, strict >37, existing Kruskal-Wallis/chi-square and SMD helpers. No confidence intervals, fitting, bootstrap, threshold selection, imputation or randomness. Two isolated computations must have identical payload bytes. No participant-level persistence or network use.

Stop on identity, alignment, schema, privacy, collision, legacy reproduction or reconciliation failure. Completion requires aggregate readback, full-precision statistics, unchanged protected hashes, correct current bindings and independent review explicitly pending.

Pre-execution UTC timestamp: 2026-09-10T12:10:15.579680+00:00

Invocation: `python scripts/run_etable4_ca199_correction.py`

Target contract SHA-256: `0b81dc00dbd97c4202e2ece486196c3745d3189c3ad0a3497b1afdfdf6a2b7f1`

Actual Git provenance:
```json
{
  "branch": "main",
  "head": "b15f80533d7d2653790fecffb0c84563fd382606",
  "worktree": "M configs/visualization/EXPORT.md\n M configs/visualization/README.md\n M configs/visualization/figure2.json\n M configs/visualization/figure4.json\n M configs/visualization/main_figure_v1.json\n M src/analysis/cohort_description.py\n M src/visualization/artists.py\n M src/visualization/design.py\n M src/visualization/figure1_bundle.py\n M src/visualization/object_inputs.py\n M src/visualization/provenance.py\n M tests/test_cohort_description.py\n M tests/visualization/test_design.py\n M tests/visualization/test_object_isolation.py\n?? scripts/run_etable4_ca199_correction.py\n?? scripts/visualization/render_figure2.py\n?? src/analysis/etable4_ca199_correction.py\n?? src/visualization/figure2.py\n?? src/visualization/figure2_bundle.py\n?? tests/test_etable4_ca199_correction.py\n?? tests/visualization/test_figure2.py",
  "staged": "",
  "correction_commit": "pending_user_git_action"
}
```

Protected pre-execution identities:
```json
{
  "data/raw/clinical_data/metadata.csv": "58b7c08d9eea834e22d9aebe0e19049f5ca313c384262b3cba21ce9763523aa9",
  "data/processed/sample_index.csv": "3e31635c37c3cc142eade488c00a240ec5487175b531f9ed94837946ddcb403f",
  "data/processed/splits/train_idx.npz": "82f64e88f29228faa7dc93cf507013f14db32b80bf9d20e417c53f1ae6bb1131",
  "data/processed/splits/test_idx.npz": "7fc60acfc54f27e77f60a9813ca829fd61f174a8d3e4ab59aac18f5e05c981c3",
  "configs/modeling_global.yaml": "c4520df8040adfebdfd1f6ae89376be5484ae3dfeda0cfd2481103df323a2e54",
  "outputs/analysis/table_two_six_categories.csv": "46820f1f5c583e7dcca9c3d76037d5bf1bbd1feb43f689d0edddab83c960b546",
  "outputs/analysis/table_two_six_categories.md": "1f555f3bb918e3457719bf170916add23999a17af0c818ec995d97d6166aa34d",
  "outputs/analysis/stage7_part_c_audit.json": "84cddc888bb7260d42b5cbb09d24dea28a418685d46cf9c2762aadcaf4ebfc03",
  "outputs/analysis/tumor_marker_selection_bias.csv": "a41fbb08f93d28528919c2916dff5cf12c323a4a19b86863db98dc04819708a2",
  "outputs/analysis/tumor_marker_lr_comparison_minirocket_ridge.csv": "a2363085473a6e427e045e9dc19383c4f94d253a61f9a392be4d801b36ff7a51",
  "outputs/analysis/voc_marker_fusion_minirocket_ridge.csv": "bc10ce28b06c8d1f01dd5d70a14ee393fc6fb86a29101774c4b790a3d48083af",
  "src/analysis/stage7_part_c_common.py": "8dfb3800754d45bac8c5bf8d67223863c96dacc023a1307a40befb237463b237",
  "scripts/run_cohort_description.py": "55376ec044d8270b2560b753d73de130f0197e76d7a60cd7933d64ee443f2981",
  "docs/plans/visualization_v2/object_contracts/efigure16_voc_tumor_marker_comparison.contract.md": "099cc10e9d37ac82335448d242cea8cc1f691d804456c6a18847f2006ea277b4",
  "outputs/analysis/table_one_pos_vs_neg.csv": "8310da5b8d6db8cb7afc56d94c2706f14e2c135aff0406252e0a4d7654f2d22c",
  "outputs/analysis/table_one_pos_vs_neg.md": "4a7baffbab2caf82ba457dfa3739f37c6f7d6caeda03b8b84482e824fd874f73"
}
```

Method/environment identities:
```json
{
  "src/analysis/cohort_description.py": "f5a77af01b5c34460ff4f5c423595b26265d6953267fd196c9f261ecb17a8f12",
  "src/analysis/etable4_ca199_correction.py": "a200a87cdbee71b32a85b528c16d64a81d9fe299a54c226d43703739dce541c5",
  "scripts/run_etable4_ca199_correction.py": "4a839bbdef39ca1197cc3bdb1ee2fc50683c75944b407f561f92d57a6e2d4737",
  "tests/test_cohort_description.py": "926f2e0c5eba69aea6ab405cfffc8617234dd8aab85d55a330576b5fa96de4e3",
  "tests/test_etable4_ca199_correction.py": "05d8155aba7966c6ffec9fb79af80615e7e0e4aec7661f0d57d97037aacf9803",
  "src/analysis/data_split.py": "4c10e56a95d675f47c57922133bd48232893500d515ff50f3e215150261b2ada",
  "environment.yml": "4681d736febc0dbd182dda0627375d4a68591d110555b5450698eaba23e16803"
}
```

## Execution and verification

Implemented and self-checked; independent review pending. Correction commit remains pending the user’s Git action. Historical bytes and eFigure 16 contract/sources are unchanged. Both isolated real computations produced identical scientific payloads; all files were read back. The dangerous legacy real-data generator test was excluded.

Machine-derived reconciliation:
```json
{
  "legacy": {
    "availability": {
      "cea": 665,
      "ca199": 651,
      "ca724": 651
    },
    "complete_cases": 650,
    "train": {
      "complete_cases": 515,
      "positive": 292,
      "negative": 223
    },
    "test": {
      "complete_cases": 135,
      "positive": 76,
      "negative": 59
    }
  },
  "corrected": {
    "availability": {
      "cea": 665,
      "ca199": 652,
      "ca724": 651
    },
    "complete_cases": 651,
    "train": {
      "complete_cases": 516,
      "positive": 293,
      "negative": 223
    },
    "test": {
      "complete_cases": 135,
      "positive": 76,
      "negative": 59
    }
  }
}
```

CA19-9 availability is marker-specific; all-three-marker complete cases use the intersection. Frozen Stage 7 complete-case counts reconcile; eFigure 16 needs no statistical correction.

Machine-derived changed historical cells:
```json
[
  {
    "variable": "CA19-9, U/mL (median [IQR])",
    "column": "Total (N=972)",
    "before": "9.0 [5.5–16.2]",
    "after": "9.0 [5.5–16.1]"
  },
  {
    "variable": "CA19-9, U/mL (median [IQR])",
    "column": "EGC (n=122)",
    "before": "10.5 [5.3–16.6]",
    "after": "10.5 [5.4–16.5]"
  },
  {
    "variable": "CA19-9, U/mL (median [IQR])",
    "column": "p value",
    "before": "0.2770",
    "after": "0.2694"
  },
  {
    "variable": "CA19-9 missing, n (%)",
    "column": "Total (N=972)",
    "before": "321/972 (33.0%)",
    "after": "320/972 (32.9%)"
  },
  {
    "variable": "CA19-9 missing, n (%)",
    "column": "EGC (n=122)",
    "before": "15/122 (12.3%)",
    "after": "14/122 (11.5%)"
  },
  {
    "variable": "CA19-9 missing, n (%)",
    "column": "|SMD|",
    "before": "1.077085",
    "after": "1.080719"
  },
  {
    "variable": "CA19-9 elevated (>37 U/mL), n (%)",
    "column": "Total (N=972)",
    "before": "43/651 (6.6%)",
    "after": "43/652 (6.6%)"
  },
  {
    "variable": "CA19-9 elevated (>37 U/mL), n (%)",
    "column": "EGC (n=122)",
    "before": "4/107 (3.7%)",
    "after": "4/108 (3.7%)"
  },
  {
    "variable": "CA19-9 elevated (>37 U/mL), n (%)",
    "column": "|SMD|",
    "before": "0.440445",
    "after": "0.440753"
  }
]
```

Correction package: `outputs/analysis/etable4_ca199_decimal_correction_2026-09-10`.
Manifest SHA-256: `966258685fa25b534be3c071dda022b3efe830ae7237f52deaf021e20c96d6fa`.
No rendering, release, model access or external transmission occurred. Current eTable 4 binding registration is a subsequent documentation step; ET-04 stays unauthorized.

## Bounded verification and source registration

Synthetic regression: 33 passed, 1 deselected (the historical real-data generator test), using `python -m pytest tests/test_cohort_description.py tests/test_etable4_ca199_correction.py -k 'not test_pos_vs_neg_row_count_and_total_consistency' -q`.

The actual entry point completed two isolated executions with identical aggregate payload bytes. Full-precision numerical readback, category totals, all seven CA19-9 columns, strict thresholds, changed-cell inventory, immutable input hashes and eFigure 16 reconciliation passed. Privacy checks used aggregate schemas and an in-memory raw-value comparison; identities and recovered raw measurements were not persisted.

Future eTable 4 current source_01/source_02 bind the corrected CSV/Markdown package; source_06 supplies unrounded CA19-9 evidence. Historical source files remain separately identified. Source_04 is the correction entry point, not the bulk runner. The contract, target registry record, closure and snapshot current hashes are synchronized under a brief directory lock with fresh reads and compare-before-atomic-replace checks. Other object records and historical reviews are preserved. The old review covers baseline contract SHA-256 `0b81dc00dbd97c4202e2ece486196c3745d3189c3ad0a3497b1afdfdf6a2b7f1` only; corrected statistics and bindings await independent review. ET-04 remains planned_not_authorized.

Historical parser SHA-256: `5ad3ed73b6886b19411621e5ba852d4c7700038923087a3ce2e54d3e0e9f4992`; historical parser test SHA-256: `5cbe64b6155da3972e00e9848b4b77a8d1bc786099cbb77b88684e4926b79a77`. The package manifest records executed correction methods and actual Git state. The pre-existing visualization work was not edited by this task; no Git mutations occurred.

## Full-precision affected test evidence

Values below are read from ca199_correction_statistics.csv. The two small P values change even though their historical formatted cells both remain 0.0000.

| Existing statistic | Legacy | Corrected |
|---|---:|---:|
| continuous_p | 0.27698737419360325 | 0.2693928028940626 |
| continuous_smd | 0.31202146188965885 | 0.31202146188965885 |
| missing_p | 9.581510807857574e-50 | 3.092686927844434e-50 |
| missing_smd | 1.0770850214595658 | 1.080719114386907 |
| elevated_p | 1.3933704510202546e-06 | 1.3317150416662206e-06 |
| elevated_smd | 0.4404450754475899 | 0.44075257478022895 |
