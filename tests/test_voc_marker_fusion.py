from __future__ import annotations

import importlib

import pandas as pd
import pytest


def _import_module_or_fail():
    """Import the Session 13 fusion module or fail clearly."""
    try:
        return importlib.import_module("src.analysis.vocs_marker_fusion")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Missing module src.analysis.vocs_marker_fusion: {exc}", pytrace=False)


def test_fusion_feature_matrix_has_locked_four_feature_order() -> None:
    """Fusion features must be exactly [voc_p_raw, cea, ca199, ca724]."""
    module = _import_module_or_fail()
    frame = pd.DataFrame(
        {
            "sample_id": [1, 2],
            "voc_p_raw": [0.2, 0.8],
            "cea": [1.0, 2.0],
            "ca199": [3.0, 4.0],
            "ca724": [5.0, 6.0],
        }
    )

    feature_frame = module._build_fusion_feature_frame(frame)

    assert feature_frame.columns.tolist() == ["voc_p_raw", "cea", "ca199", "ca724"]
    assert feature_frame.shape == (2, 4)


def test_conclusion_template_selects_voc_alone_preferable_for_negative_increment() -> None:
    """A sub-0.01 AUC gain with non-positive reclassification should choose template A."""
    module = _import_module_or_fail()

    branch, paragraph = module._select_fusion_conclusion(
        delta_auc=0.0099,
        delong_p_value=0.042,
        nri=-0.1000,
        idi=0.0200,
    )

    assert branch == "voc_alone_preferable"
    assert paragraph == "VOC alone is preferable; supplementation with markers is not supported by these data."


def test_conclusion_template_selects_voc_alone_preferable_for_e013_trigger_case() -> None:
    """Erratum E013 should route a just-above-threshold, non-significant, negative-NRI case to template A."""
    module = _import_module_or_fail()

    branch, paragraph = module._select_fusion_conclusion(
        delta_auc=0.0103,
        delong_p_value=0.3519,
        nri=-0.6044,
        idi=0.0085,
    )

    assert branch == "voc_alone_preferable"
    assert paragraph == "VOC alone is preferable; supplementation with markers is not supported by these data."


def test_conclusion_template_selects_fusion_improves_otherwise() -> None:
    """The positive branch should interpolate the locked wording and numeric fields."""
    module = _import_module_or_fail()

    branch, paragraph = module._select_fusion_conclusion(
        delta_auc=0.0234,
        delong_p_value=0.00321,
        nri=0.1456,
        idi=0.0189,
    )

    assert branch == "fusion_improves"
    assert paragraph == (
        "Fusion improves over VOC alone by ΔAUC = 0.0234 (p=0.0032), "
        "with NRI=0.1456, IDI=0.0189; the addition of tumor markers provides incremental information."
    )


def test_markdown_conclusion_paragraph_contains_prespecified_literal_phrasing() -> None:
    """Rendered markdown should place the conclusion paragraph up front with the exact template text."""
    module = _import_module_or_fail()

    markdown = module._render_fusion_markdown(
        best_model="minirocket_ridge",
        conclusion_paragraph="VOC alone is preferable; supplementation with markers is not supported by these data.",
        rationale_paragraph="Rationale: placeholder.",
        metric_lines=["- placeholder"],
    )

    assert markdown.splitlines()[0] == "# VOC + Marker Fusion - minirocket_ridge"
    assert "VOC alone is preferable; supplementation with markers is not supported by these data." in markdown
