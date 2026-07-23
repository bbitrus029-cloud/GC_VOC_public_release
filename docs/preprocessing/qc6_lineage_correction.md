# Additive QC6 reporting-lineage correction

- correction_date_utc: `2026-07-23T01:38:46Z`
- Scope: reporting closure only; this note does not edit the frozen protocol or run summary.

The operational plan used an any-channel rule: any gas-channel spike count above the configured threshold fails QC6. The executed implementation applied `not any(spike_counts > configured_threshold)` pass semantics in `src/data/qc.py::_evaluate_qc6` (commit `595d7db1525560e5e2e61936a222a71abbfe2762`). The frozen threshold was `211.0` in `configs/preprocessing.yaml` (provenance commit `421254714d7af7ba4466e1424bb443f26270d464`), so equality at 211.0 passes under the strict `>` rule.

The frozen protocol omitted the sample-level aggregation rule, while the historical frozen run summary used inaccurate six-channel quorum wording (commit `622cb3664774b3940f7bcf092004cfff62d06822`). Frozen QC artifacts from commit `650ce0e12c5f638f324ddad18f36812b14bc90ee` recorded zero QC6 failures. Consequently, this wording correction had no cohort or model impact and changes no analytical result. Future Methods and cohort-flow text must use the actually executed any-channel rule.
