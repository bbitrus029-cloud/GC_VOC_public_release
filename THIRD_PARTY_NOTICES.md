# Third-party notices

This file distinguishes redistributed or potentially adapted material, recipient-installed dependencies and
local-only preparation tools. LICENSING.md defines project grants; it does not replace third-party rights.
The review used static source inspection, declarations, runtime versions, installed distribution legal metadata
and version-specific upstream legal texts. No model was deserialized or executed.

## Notices retained for the DeLong routines

Affected path: `src/evaluation/statistical_tests.py`, specifically `_compute_midrank` and `_fast_delong`.
The tie-ranking loop, positive/negative/all-sample rank arrays and covariance sequence closely correspond
to the public implementations below. This is conservative notice treatment of potential adaptation, not a
claim that an undocumented historical copy came from the consulted revision. Frozen code is unchanged.
Local differences include descriptive names, `np.empty_like` allocations, one-based rank placement and
`np.atleast_2d` covariance handling.

- Yandex Data School `roc_comparison`, consulted commit `62da5b8f59fb9029dca8365d2b7098de735dc658`:
  [source](https://github.com/yandexdataschool/roc_comparison/blob/62da5b8f59fb9029dca8365d2b7098de735dc658/compare_auc_delong_xu.py) and
  [MIT license](https://github.com/yandexdataschool/roc_comparison/blob/62da5b8f59fb9029dca8365d2b7098de735dc658/LICENSE).
  Copyright (c) 2021 Nikita Kazeev. Complete unmodified text:
  [LICENSES/ROC-COMPARISON-MIT.txt](LICENSES/ROC-COMPARISON-MIT.txt).
- The Yandex source states adaptation from Netflix VMAF. VMAF v3.0.0
  [sigproc source](https://github.com/Netflix/vmaf/blob/v3.0.0/python/vmaf/tools/sigproc.py)
  declares “Copyright 2016-2020, Netflix, Inc.” and BSD+Patent; the
  [license](https://github.com/Netflix/vmaf/blob/v3.0.0/LICENSE) identifies BSD-2-Clause-Patent and
  Copyright (c) 2020 Netflix, Inc. Complete text:
  [LICENSES/VMAF-BSD-2-Clause-Patent.txt](LICENSES/VMAF-BSD-2-Clause-Patent.txt).
  The upstream `fastDeLong` attribution to Xu Sun, version 2014/12, and the Sun and Xu method are retained here.
  A paper citation alone was not treated as a software license.

Retain copyright, permission conditions and disclaimers with covered portions and source/binary redistributions
as required by those texts. No complete Yandex or VMAF distribution is bundled. Original project contributions
use Apache-2.0. No other copied upstream code distribution was identified in the targeted review; this is not
an exhaustive authorship investigation or a grant for material absent from the manifest.

## Separately installed dependencies

These are not bundled dependency distributions. Imports, class references and fitted state do not convey
ownership of the corresponding libraries. The 14 fitted project artifacts use Apache-2.0; required upstream
implementations retain their licenses. Versions were checked against installed distributions and the recorded
runtime/environment declarations. Links below are version-specific primary legal sources.

| Component | Version | Upstream license | Relationship | Primary legal source |
| --- | --- | --- | --- | --- |
| scikit-learn | 1.7.2 | BSD-3-Clause | Estimators, preprocessing and calibration; installed separately | [license](https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.7.2/COPYING) |
| sktime | 0.40.1 | BSD-3-Clause | MiniRocketMultivariate transformer; installed separately | [license](https://raw.githubusercontent.com/sktime/sktime/v0.40.1/LICENSE) |
| xgboost | 3.2.0 | Apache-2.0 | XGBoost estimator; installed separately | [license](https://raw.githubusercontent.com/dmlc/xgboost/v3.2.0/LICENSE) |
| pytorch-lightning | 2.6.1 | Apache-2.0 | Deep-model runtime; installed separately | [license](https://raw.githubusercontent.com/Lightning-AI/pytorch-lightning/2.6.1/LICENSE) |
| shap | 0.51.0 | MIT | Explanation implementation; installed separately | [license](https://raw.githubusercontent.com/shap/shap/v0.51.0/LICENSE) |
| torch | 2.10.0 | BSD-3-Clause | Tensor/deep-model runtime; upstream bundled notices also apply; installed separately | [license](https://raw.githubusercontent.com/pytorch/pytorch/v2.10.0/LICENSE) |
| numpy | 2.3.5 | BSD-3-Clause | Numerical arrays; installed separately | [license](https://raw.githubusercontent.com/numpy/numpy/v2.3.5/LICENSE.txt) |
| scipy | 1.17.1 | BSD-3-Clause | Numerical/statistical routines; installed separately | [license](https://raw.githubusercontent.com/scipy/scipy/v1.17.1/LICENSE.txt) |
| pandas | 2.3.3 | BSD-3-Clause | Tabular input/output; installed separately | [license](https://raw.githubusercontent.com/pandas-dev/pandas/v2.3.3/LICENSE) |
| numba | 0.65.1 | BSD-2-Clause | MiniRocket JIT dependency; installed separately | [license](https://raw.githubusercontent.com/numba/numba/0.65.1/LICENSE) |
| joblib | 1.5.3 | BSD-3-Clause | Estimator runtime support; installed separately | [license](https://raw.githubusercontent.com/joblib/joblib/1.5.3/LICENSE.txt) |
| PyYAML | 6.0.3 | MIT | Configuration parsing; installed separately | [license](https://raw.githubusercontent.com/yaml/pyyaml/6.0.3/LICENSE) |
| captum | 0.9.0 | BSD-3-Clause | Integrated gradients; installed separately | [license](https://raw.githubusercontent.com/pytorch/captum/v0.9.0/LICENSE) |
| statsmodels | 0.14.6 | BSD-3-Clause | Statistical analysis; installed separately | [license](https://raw.githubusercontent.com/statsmodels/statsmodels/v0.14.6/LICENSE.txt) |
| scikit-image | 0.26.0 | BSD-3-Clause; per-file BSD-2-Clause and other exceptions | Signal processing; installed separately | [license](https://raw.githubusercontent.com/scikit-image/scikit-image/v0.26.0/LICENSE.txt) |
| torchmetrics | 1.9.0 | Apache-2.0 | Deep-model metrics; installed separately | [license](https://raw.githubusercontent.com/Lightning-AI/torchmetrics/v1.9.0/LICENSE) |
| pytest | 9.0.3 | MIT | Historical test dependency; installed separately | [license](https://raw.githubusercontent.com/pytest-dev/pytest/9.0.3/LICENSE) |

BSD-2-Clause and BSD-3-Clause require retaining copyright/conditions/disclaimers with redistributed source
and accompanying binary documentation; BSD-3-Clause also restricts endorsement using upstream names. MIT
requires retaining its copyright/permission notice. Apache-2.0 requires the license, change notices and
applicable attribution/NOTICE material. These summaries do not replace the legal texts. Upstream distributions
can include separately licensed components; consult their complete notices if bundling them later.
This package does not bundle BLAS, CUDA, torch libraries, wheels, conda packages, fonts or dependency caches.

### Other entries in the preserved environment declaration

These tools are declared by environment.yml and installed separately. Their version-specific installed legal
texts or conda distribution metadata were checked. They are not additional redistributed project content.
Renderer/document tools do not imply inclusion of finished figures/tables or a rendering run in this update.

| Component | Version | Original terms | Official source |
| --- | --- | --- | --- |
| ipykernel | 7.2.0 | BSD-3-Clause | [upstream](https://github.com/ipython/ipykernel/tree/v7.2.0) |
| pip | 26.0.1 | MIT; bundled dependency notices | [upstream](https://github.com/pypa/pip/tree/26.0.1) |
| matplotlib | 3.10.9 | LicenseRef-Matplotlib (PSF-derived) | [upstream](https://matplotlib.org/3.10.9/project/license.html) |
| seaborn | 0.13.2 | BSD-3-Clause | [upstream](https://github.com/mwaskom/seaborn/tree/v0.13.2) |
| tqdm | 4.67.3 | MPL-2.0 AND MIT | [upstream](https://github.com/tqdm/tqdm/tree/v4.67.3) |
| CairoSVG | 2.9.0 | LGPL-3.0-or-later | [upstream](https://github.com/Kozea/CairoSVG/tree/2.9.0) |
| fonttools | 4.62.1 | MIT; external-font notices | [upstream](https://github.com/fonttools/fonttools/tree/4.62.1) |
| lxml | 6.1.1 | BSD-3-Clause; per-file exceptions | [upstream](https://github.com/lxml/lxml/tree/lxml-6.1.1) |
| Pillow | 12.2.0 | MIT-CMU | [upstream](https://github.com/python-pillow/Pillow/tree/12.2.0) |
| python-docx | 1.2.0 | MIT | [upstream](https://github.com/python-openxml/python-docx/tree/v1.2.0) |
| fontconfig | 2.17.1 | MIT | [upstream](https://gitlab.freedesktop.org/fontconfig/fontconfig/-/tree/2.17.1) |
| librsvg | 2.62.1 | LGPL-2.1-or-later | [upstream](https://gitlab.gnome.org/GNOME/librsvg/-/tree/2.62.1) |
| poppler | 26.02.0 | GPL-2.0-or-later | [upstream](https://poppler.freedesktop.org/) |
| python | 3.11.15 | Python-2.0 | [upstream](https://github.com/python/cpython/tree/v3.11.15) |

LicenseRef-Matplotlib describes Matplotlib's PSF-derived license, not a blanket PSF-2.0 relicensing.
Per-file and bundled-component exceptions remain with upstream distributions. LGPL/GPL/MPL components retain
applicable source-availability, change-notice and other obligations if those components are redistributed.
An environment declaration does not bundle their binaries or apply those licenses to unrelated project code/data.
No environment changes or installations were made.

## Historical references and local-only tools

- `tests/test_delong_aSAH.py` references pROC 1.19.0.1 and an aSAH fixture; its
  [version-specific DESCRIPTION](https://raw.githubusercontent.com/cran/pROC/1.19.0.1/DESCRIPTION)
  declares GPL-3.0-or-later.
- `tests/test_jt_npordtests.py` references npordtests 1.2 and jdata; its
  [version-specific DESCRIPTION](https://raw.githubusercontent.com/cran/npordtests/1.2/DESCRIPTION)
  declares GPL-2.0-or-later. Neither R package nor either fixture is distributed. Retained tests and reference
  assertions document historical validation; these missing-fixture tests are not claimed to run from this package.
  Matching a statistical formula or checking reference numbers is not evidence of copying package code.
- git-filter-repo v2.47.0, recorded upstream commit `6f79afc8c90c592a3052e6cc53c2ca8907515bca`, was a private
  preparation tool, not a public package file. Upstream [COPYING](https://github.com/newren/git-filter-repo/blob/v2.47.0/COPYING)
  and [COPYING.mit](https://github.com/newren/git-filter-repo/blob/v2.47.0/COPYING.mit) permit the standalone tool
  under MIT, with additional alternatives described upstream. Separate test-harness exceptions do not make
  that harness part of this release. Private tools and downloaded sources remain excluded.

LICENSE and LICENSE-DATA were obtained unchanged from the official English plain-text endpoints, including
the Apache example appendix; inclusion does not assert project ownership of legal texts.
No concrete incompatible term or missing redistribution permission was identified for the material treated here.
The unrecorded historical DeLong derivation revision remains a provenance limit, not silently replaced with
the consulted upstream revision. No publication, hosting configuration or upload has occurred.
