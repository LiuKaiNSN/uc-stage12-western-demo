from __future__ import annotations

from typing import Any, Dict

import streamlit as st

from ..artifacts import StageArtifacts
from ..config_loader import config_profile
from ..feature_labels import get_stage_feature_meta
from ..figures import (
    get_enlarged_image,
    list_interpretability_panel,
    list_stage_publication_beeswarm,
    list_stage_publication_pdp,
    list_top_pdp_images,
)
from ..inference import PredictionResult, predict_stage
from .form_common import feature_input_form, render_variable_reference_bilingual


def _format_metric(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def render_metrics_sidebar(artifacts: StageArtifacts) -> None:
    m = artifacts.metrics
    st.subheader("External validation")
    st.markdown(f"**n (external):** {m.get('n_external', '—')}")
    st.markdown(f"**AUC:** {_format_metric(m.get('auc'))}")
    st.markdown(f"**AUPRC:** {_format_metric(m.get('auprc'))}")
    st.markdown(f"**F1 @ 0.5:** {_format_metric(m.get('f1'))}")


def render_prediction_result(result: PredictionResult) -> None:
    st.metric(result.probability_label, f"{result.probability:.4f}")
    st.progress(min(max(result.probability, 0.0), 1.0))
    if result.refer:
        st.warning(result.refer_message)
        st.markdown(f"**Flag / 标志:** Refer — {result.positive_label}")
    else:
        st.info(result.no_refer_message)
        st.markdown("**Flag / 标志:** No refer")


def render_skip_note_bilingual(cfg: dict) -> None:
    note = cfg.get("stage1_skip_note", {})
    st.markdown("#### 临床路径提示 / Clinical pathway note")
    col_zh, col_en = st.columns(2)
    with col_zh:
        st.markdown("**中文**")
        st.info(note.get("zh", ""))
    with col_en:
        st.markdown("**English**")
        st.info(note.get("en", ""))


def _uses_publication_figures(cfg: dict, stage_key: str) -> bool:
    block = cfg.get("figures", {}).get("stage_publication", {}).get(stage_key, {})
    return bool(isinstance(block, dict) and block.get("beeswarm"))


def render_figures_panel(cfg: dict, stage_key: str) -> None:
    top_k = int(cfg.get("figures", {}).get("pdp_top_k", 5))
    enlarge_stem = str(cfg.get("figures", {}).get("enlarge_stem", "shap_importance_correlation"))
    st.subheader("Global explainability (static)")
    st.caption("Publication figures (precomputed; not updated per input).")

    if _uses_publication_figures(cfg, stage_key):
        beeswarm = list_stage_publication_beeswarm(cfg, stage_key)
        pdp_images = list_stage_publication_pdp(cfg, stage_key, top_k=top_k)
        if not beeswarm and not pdp_images:
            st.warning("Publication interpretability figures not found.")
            return
        for path, caption in beeswarm:
            st.image(str(path), caption=caption, use_container_width=True)
        if pdp_images:
            st.markdown("**Partial dependence (top 5 by SHAP rank)**")
            for path, caption in pdp_images:
                st.image(str(path), caption=caption, use_container_width=True)
        return

    st.caption("External validation SHAP beeswarm + top PDP plots (precomputed).")
    panels = list_interpretability_panel(cfg, stage_key)
    if not panels and not list_top_pdp_images(cfg, stage_key, top_k):
        st.warning("Interpretability figures not found.")
        return

    for path, caption in panels:
        if path.stem == enlarge_stem:
            continue
        st.image(str(path), caption=caption, use_container_width=True)

    pdp_images = list_top_pdp_images(cfg, stage_key, top_k=top_k)
    if pdp_images:
        st.markdown("**Partial dependence (top 5 by SHAP rank)**")
        for path, caption in pdp_images:
            st.image(str(path), caption=caption, use_container_width=True)


def render_enlarged_correlation(cfg: dict, stage_key: str) -> None:
    enlarged = get_enlarged_image(cfg, stage_key=stage_key)
    if enlarged:
        path, caption = enlarged
        st.markdown("---")
        st.markdown(f"### {caption}")
        st.caption("Full-width view / 全宽显示便于阅读")
        st.image(str(path), use_container_width=True)


def render_stage_tab(
    cfg: dict,
    artifacts: StageArtifacts,
    stage_key: str,
    feature_meta: Dict[str, Dict[str, Any]],
) -> None:
    left, right = st.columns([1.2, 0.85], gap="large")
    with left:
        st.subheader(artifacts.stage_cfg["display_name"])
        values, errors = feature_input_form(
            artifacts.feature_names,
            feature_meta,
            form_key=f"form_{stage_key}",
        )
        if errors:
            for err in errors:
                st.error(err)
        elif values is not None:
            result = predict_stage(artifacts, values)
            render_prediction_result(result)

        if stage_key == "stage1":
            render_skip_note_bilingual(cfg)

        render_variable_reference_bilingual(
            artifacts.feature_names,
            feature_meta,
            footer_zh=cfg.get("variable_reference_footer", {}).get("zh"),
            footer_en=cfg.get("variable_reference_footer", {}).get("en"),
        )

    with right:
        render_metrics_sidebar(artifacts)
        st.divider()
        render_figures_panel(cfg, stage_key)

    render_enlarged_correlation(cfg, stage_key)


def render_western_stage12(cfg: dict, artifacts_map: dict) -> None:
    profile = config_profile(cfg)
    meta_s1 = get_stage_feature_meta(profile, "stage1")
    meta_s2 = get_stage_feature_meta(profile, "stage2")

    tab1, tab2 = st.tabs([
        cfg["stages"]["stage1"]["tab_title"],
        cfg["stages"]["stage2"]["tab_title"],
    ])
    with tab1:
        render_stage_tab(cfg, artifacts_map["stage1"], "stage1", meta_s1)
    with tab2:
        render_stage_tab(cfg, artifacts_map["stage2"], "stage2", meta_s2)
