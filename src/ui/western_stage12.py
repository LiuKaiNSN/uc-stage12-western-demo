from __future__ import annotations

from typing import Any, Callable, Dict

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


def _fig_toggle_key(cfg: dict, stage_key: str) -> str:
    return f"show_figs_{config_profile(cfg)}_{stage_key}"


def render_figures_panel(cfg: dict, stage_key: str) -> None:
    """Explainability images load only after the user turns the toggle on (saves RAM)."""
    top_k = int(cfg.get("figures", {}).get("pdp_top_k", 5))
    enlarge_stem = str(cfg.get("figures", {}).get("enlarge_stem", "shap_importance_correlation"))
    st.subheader("Global explainability (static)")
    st.caption(
        "Publication figures (precomputed; not updated per input).  "
        "Hidden by default to reduce Cloud memory use."
    )

    show = st.toggle(
        "Show SHAP / beeswarm / PDP figures  |  显示解释性图（蜂群图 / PDP 等）",
        value=False,
        key=_fig_toggle_key(cfg, stage_key),
    )
    if not show:
        st.info(
            "Figures are collapsed to save memory. Turn the switch on to view SHAP beeswarm, "
            "PDP, and related panels.  \n"
            "默认折叠以节省内存；打开开关后可查看 SHAP 蜂群图、PDP 等。"
        )
        return

    if _uses_publication_figures(cfg, stage_key):
        beeswarm = list_stage_publication_beeswarm(cfg, stage_key)
        pdp_images = list_stage_publication_pdp(cfg, stage_key, top_k=top_k)
        if not beeswarm and not pdp_images:
            st.warning("Publication interpretability figures not found.")
            return
        with st.expander("SHAP beeswarm / 蜂群图", expanded=True):
            if not beeswarm:
                st.caption("No beeswarm figures found.")
            for path, caption in beeswarm:
                st.image(str(path), caption=caption, use_container_width=True)
        with st.expander("Partial dependence (PDP) / 偏依赖图", expanded=False):
            if not pdp_images:
                st.caption("No PDP figures found.")
            for path, caption in pdp_images:
                st.image(str(path), caption=caption, use_container_width=True)
        return

    panels = list_interpretability_panel(cfg, stage_key)
    pdp_images = list_top_pdp_images(cfg, stage_key, top_k=top_k)
    if not panels and not pdp_images:
        st.warning("Interpretability figures not found.")
        return

    with st.expander("SHAP / interpretability panels  |  SHAP 解释面板", expanded=True):
        shown = False
        for path, caption in panels:
            if path.stem == enlarge_stem:
                continue
            shown = True
            st.image(str(path), caption=caption, use_container_width=True)
        if not shown:
            st.caption("No panel figures found.")

    with st.expander("Partial dependence (PDP) / 偏依赖图", expanded=False):
        if not pdp_images:
            st.caption("No PDP figures found.")
        for path, caption in pdp_images:
            st.image(str(path), caption=caption, use_container_width=True)


def render_enlarged_correlation(cfg: dict, stage_key: str) -> None:
    # Only decode the large correlation image when explainability is enabled.
    if not st.session_state.get(_fig_toggle_key(cfg, stage_key), False):
        return
    enlarged = get_enlarged_image(cfg, stage_key=stage_key)
    if not enlarged:
        return
    path, caption = enlarged
    with st.expander(f"{caption} (full width) / 全宽相关图", expanded=False):
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


def render_western_stage12(
    cfg: dict,
    load_stage: Callable[[str], StageArtifacts],
) -> None:
    """Render one stage at a time (radio) so only that model is loaded.

    Streamlit tabs execute all tab bodies, which would defeat lazy loading.
    """
    profile = config_profile(cfg)
    stage_keys = ["stage1", "stage2"]
    labels = {k: cfg["stages"][k]["tab_title"] for k in stage_keys}

    stage_key = st.radio(
        "Select stage / 选择阶段",
        stage_keys,
        format_func=lambda k: labels[k],
        horizontal=True,
        key=f"stage_select_{profile}",
    )
    st.caption(
        "Only the selected stage model is loaded (memory-saving).  "
        "仅加载当前所选阶段模型，以降低 Cloud 内存占用。"
    )

    with st.spinner(f"Loading {labels[stage_key]} model… / 正在加载模型…"):
        artifacts = load_stage(stage_key)
    meta = get_stage_feature_meta(profile, stage_key)
    render_stage_tab(cfg, artifacts, stage_key, meta)
