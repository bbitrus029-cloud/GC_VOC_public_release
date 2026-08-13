# Stage 4 Numerical-Environment Build-Drift Correction

**Date:** 2026-08-13
**Classification:** additive execution-environment provenance and reproducibility correction
**Diagnostic classification:** `ROOT_CAUSE_CONFIRMED_NUMERICAL_ENVIRONMENT_BUILD_DRIFT`

## Purpose and historical status

This additive record closes the aggregate-safe root-cause diagnosis of the failed eFigure 4 Stage 4 reconciliation.
It preserves the failed executions, the original results-free plan, recovery-v2, the frozen Pipeline A artifact, and
all prior records unchanged as historical evidence. It does not rewrite or invalidate them.

The diagnosis was performed after the formal replay failed closed with
`FIRST_STAGE4_RECONCILIATION_FAILURE`. It used only aggregate-safe structural and numerical fingerprints and exposed
no participant identifier, raw filename, raw path, participant-level value, mismatch coordinate, or per-participant
diagnostic. It created no scientific output.

## Numerical-build identities

The current project environment had this numerical build:

| Package | Version | Build |
| --- | --- | --- |
| python | 3.11.15 | `hd63d673_0_cpython` |
| numpy | 2.3.5 | `py311h2e04523_1` |
| pandas | 2.3.3 | `py311hed34c8f_2` |
| pyyaml | 6.0.3 | `py311h3778330_1` |
| scipy | 1.17.1 | `py311hbe70eeb_1` |
| libblas | 3.11.0 | `8_h5875eb1_mkl` |
| libcblas | 3.11.0 | `8_hfef963f_mkl` |
| liblapack | 3.11.0 | `8_h5e43f62_mkl` |
| mkl | 2026.0.0 | `h0e700b2_915` |

The exact Pipeline A-compatible historical numerical build was:

| Package | Version | Build |
| --- | --- | --- |
| python | 3.11.15 | `hd63d673_0_cpython` |
| numpy | 2.3.5 | `py311h2e04523_1` |
| pandas | 2.3.3 | `py311hed34c8f_2` |
| pyyaml | 6.0.3 | `py311h3778330_1` |
| scipy | 1.17.1 | `py311hbe70eeb_0` |
| libblas | 3.11.0 | `6_h5875eb1_mkl` |
| libcblas | 3.11.0 | `6_hfef963f_mkl` |
| liblapack | 3.11.0 | `6_h5e43f62_mkl` |
| mkl | 2025.3.1 | `h0e700b2_11` |

Version-only equality is not build equality. Conda history records the exact transition from SciPy build `_0`,
BLAS/CBLAS/LAPACK build 6, and MKL 2025.3.1 to SciPy build `_1`, BLAS/CBLAS/LAPACK build 8, and MKL 2026.0.0 on
June 1, 2026. Frozen Pipeline A was generated on May 11, 2026 and added by commit
`650ce0e12c5f638f324ddad18f36812b14bc90ee`, dated May 12, 2026, before the June 1 transition.

## Aggregate-safe diagnostic evidence

Both the current-build and historical-build controls passed every structural check:

- shape was exactly `(972, 960, 11)`;
- sample order was equal;
- channel order was exactly S1-S11;
- clip length was 960;
- numeric dtypes were compatible;
- the dtype pair was `float64/float64`;
- both compared tensors were finite.

Under the current build, the replayed Stage 4 tensor and frozen Pipeline A had:

```text
bitwise_equal=false
allclose_rtol_0_atol_1e-12=false
maximum_absolute_difference=1.2150849215686321e-09
bitwise_differing_elements=9953790
elements_above_1e-12=5181
```

The loader-semantics control established:

```text
current_and_historical_loader_bitwise_equal=true
maximum_loader_difference=0
current_loader_passes_frozen_tolerance=false
historical_loader_under_current_build_passes=false
```

Under the exact historical numerical build, the reconstructed Stage 4 tensor and frozen Pipeline A had:

```text
bitwise_equal=true
allclose_rtol_0_atol_1e-12=true
maximum_absolute_difference=0
bitwise_differing_elements=0
elements_above_1e-12=0
```

## Correction and boundaries

The numerical-build change is causally confirmed because the exact historical numerical build reproduced frozen
Pipeline A bitwise while the current build failed the unchanged tolerance with all structural identities retained.
This conclusion is not inferred from chronology alone. Loader semantics are excluded as the cause by the
zero-difference loader control.

This correction changes no frozen data, participant alignment, preprocessing transformation, Stage 4 definition,
cohort, channel, frame, padding, aggregation, or tolerance. The reconciliation remains `rtol=0.0`,
`atol=1.0e-12`, with no NaN equality. Frozen Pipeline A remains immutable and authoritative.

The issue is an execution-environment provenance and reproducibility gap, not evidence of a changed scientific
result. Future eFigure 4 replay must restore the exact historical numerical build in an independently reviewed,
isolated, offline environment. The general project `gcvocs` environment and `environment.yml` must remain unchanged.

This record does not authorize implementation, environment creation, participant-capable access, replay,
aggregation, figure design, artifact release, or Git mutation.
