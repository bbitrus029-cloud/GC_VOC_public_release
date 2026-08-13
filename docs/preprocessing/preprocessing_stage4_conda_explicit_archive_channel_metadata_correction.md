# Stage 4 Conda Explicit-Archive Channel-Metadata Correction

**Date:** 2026-08-13
**Classification:** additive execution-environment metadata and provenance correction

## Purpose and historical status

This additive record corrects the environment-identity architecture used by the eFigure 4 recovery chain. It
preserves every failed execution, diagnostic, correction, and prior plan unchanged as historical evidence. It does
not revise the frozen preprocessing method, cohort, artifact, reconciliation tolerance, aggregation, privacy
contract, or output schema.

Recovery-v3 correctly required exact historical numerical package bytes, but it treated Conda's installed `channel`
label as if it were immutable computational identity. Conda 26.1.0 records an archive installed explicitly from an
absolute local path as a `file://` origin. Because that path is not a formal channel hierarchy, the installed record
uses channel `<unknown>`. For the six exact SHA-locked archives below, `<unknown>` is expected transport metadata; it
does not indicate different bytes, a different build, or a different numerical environment.

## Evidence retained without scientific access

The reviewed current `gcvocs` source ledger and the provisional Recovery-v3 clone each contained exactly 324 unique
package names and had canonical SHA-256
`e6a91ad784c41e2781c03947562f6bbf78d6cc33a8d0551245cb7eb9c54e04f8`.

The Recovery-v3 local explicit-install transaction installed the intended six historical packages with their exact
names, versions, builds, `linux-64` subdirectory, and archive hashes. All six installed records reported channel
`<unknown>`. The incomplete environment was removed before environment-lock creation, implementation,
participant-capable access, tests, replay, aggregation, or output creation.

That explicit local installation also rewrote the Conda-managed shared extracted-cache `repodata_record.json`
metadata for those six cached entries to channel `<unknown>` and local `file://` locators. The archive bytes remained
unchanged. This cache metadata is outside Git. It must not be repaired, deleted, rewritten, or manually edited, and
it is not a scientific input or a computational-identity authority.

A later environment-only URL-plus-SHA diagnostic reverified the source and clone ledgers, all six archive hashes,
the extracted cache entries, and retained official cached URLs. Every official URL-qualified MatchSpec nevertheless
had zero matching cache records and would have scheduled `CacheUrlAction` plus `ExtractPackageAction`, because the
shared cache records already carried the `<unknown>` local-origin metadata. The URL-form install therefore correctly
did not run.

The active `gcvocs` environment remains unchanged at 324 unique records with canonical SHA-256
`e6a91ad784c41e2781c03947562f6bbf78d6cc33a8d0551245cb7eb9c54e04f8`.

## Corrected identity model

Three provenance layers are distinct:

1. Computational package identity is the exact archive SHA-256 together with package name, version, build, and
   `linux-64` subdirectory.
2. The installed Conda `channel` field is mutable installation-transport metadata. `<unknown>` is accepted only for
   the six exact local-archive substitutions below and for no other package.
3. The official cached `conda-forge/linux-64` URLs are secondary historical provenance locators. They are not claims
   about the live installed channel and are not computational identity.

The six frozen computational identities are:

| Name | Version | Build | Subdirectory | Expected installed channel | Archive SHA-256 |
| --- | --- | --- | --- | --- | --- |
| scipy | 1.17.1 | `py311hbe70eeb_0` | `linux-64` | `<unknown>` | `0063a75e9406357ef8ecc2fb38808a2442b269bfc5a854cf2bcd368060d9a2d7` |
| libblas | 3.11.0 | `6_h5875eb1_mkl` | `linux-64` | `<unknown>` | `a73ec64c0f60a7733f82a679342bdad88e0230ba8243b12ece13a23aded431f4` |
| libcblas | 3.11.0 | `6_hfef963f_mkl` | `linux-64` | `<unknown>` | `d98a39a8e61af301bf67bf3fb946baff9686864886560cdd48d5259c080c58a5` |
| liblapack | 3.11.0 | `6_h5e43f62_mkl` | `linux-64` | `<unknown>` | `8715428e721a63880d4e548375a744f177200a5161aec3ebe533f33eaf7ec3a5` |
| mkl | 2025.3.1 | `h0e700b2_11` | `linux-64` | `<unknown>` | `e03877a111b613dff7b599fd7146def11eb3517f3813e80aedddeb00843f0bf2` |
| onemkl-license | 2025.3.1 | `hf2ce2f3_11` | `linux-64` | `<unknown>` | `857bfc5bfbb3956e56a1db3f73f8850c95ef2d962d78471bfdc5de2319149fdb` |

The other 318 final ledger records retain exact source-ledger identity, including channel and subdirectory. No other
`<unknown>` channel is permitted. The final 324-record historical ledger has canonical SHA-256
`7b11f59909a6e2f5b1bf1f8f49cba060b2e0b3d3e1b2b55f0da2eecfbcb30334`.

Local `file://` URLs and absolute cache or environment paths must never be persisted in a lock, scientific output,
audit, manifest, run summary, error, log, or handoff. Official cached URLs may be retained in this correction and a
reviewed plan only as secondary provenance locators; they are not required in runtime locks or scientific outputs.

## Scientific closure and boundaries

The exact historical numerical build previously reproduced frozen Pipeline A as:

```text
bitwise_equal=true
allclose_rtol_0_atol_1e-12=true
maximum_absolute_difference=0
bitwise_differing_elements=0
elements_above_1e-12=0
```

This end-to-end reconciliation, together with exact archive-byte identities, establishes computational equivalence.
Historical reproduction evidence already excludes loader semantics as the Stage 4 mismatch cause.

This correction changes no repository file other than this additive record and changes no `gcvocs` package,
frozen Pipeline A artifact, participant alignment, scientific method, transformation, stage, channel, frame,
tolerance, cohort, aggregation, privacy rule, or output schema. It does not authorize environment creation,
implementation, participant access, replay, aggregation, output creation, figure design, release, cache-metadata
repair, or Git mutation.
