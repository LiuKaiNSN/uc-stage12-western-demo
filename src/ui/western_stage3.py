from __future__ import annotations

from typing import Any, Dict

import streamlit as st

from ..artifacts import MulticlassStageArtifacts
from ..config_loader import config_profile
from ..feature_labels import get_stage_feature_meta
from ..figures import (
    get_enlarged_image,
    list_publication_images,
    list_publication_pdp_rank_images,
)
from ..inference import MulticlassPredictionResult, predict_multiclass
from .form_common import feature_input_form, render_variable_reference_bilingual


def _format_metric(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def render_metrics_sidebar(artifacts: MulticlassStageArtifacts) -> None:
    m = artifacts.metrics
    st.subheader("External validation")
    st.markdown(f"**n (external):** {m.get('n_external', '—')}")
    st.markdown(f"**Macro AUC-OVR:** {_format_metric(m.get('macro_auc_ovr'))}")
    st.markdown(f"**Balanced accuracy:** {_format_metric(m.get('balanced_acc'))}")
    st.markdown(f"**Macro F1:** {_format_metric(m.get('macro_f1'))}")


def render_multiclass_result(
    result: MulticlassPredictionResult,
    artifacts: MulticlassStageArtifacts,
) -> None:
    st.markdown("#### Montreal classification probabilities / Montreal 分类概率")
    class_display = artifacts.model_cfg.get("class_display", {})
    cols = st.columns(len(result.probabilities))
    for i, (cls, prob) in enumerate(result.probabilities.items()):
        disp = class_display.get(cls, {})
        with cols[i]:
            st.metric(
                f"{disp.get('zh', cls)}",
                f"{prob:.4f}",
                help=str(disp.get("en", "")),
            )
            st.caption(str(disp.get("en", "")))

    st.markdown(
        f"**Predicted class / 预测类别:** {result.predicted_label_zh}  "
        f"({result.predicted_label_en})"
    )

    st.markdown("---")
    st.markdown("#### Extensive disease (E3) pathway / 广泛型（E3）路径")
    st.caption(
        f"E3 operating threshold / E3 工作点阈值: **{artifacts.e3_threshold:.2f}** "
        "(prespecified on external DCA; separate from argmax class prediction)."
    )
    st.metric("P(E3)", f"{result.e3_probability:.4f}")
    st.progress(min(max(result.e3_probability, 0.0), 1.0))
    if result.e3_refer:
        st.warning(result.e3_refer_message)
        st.markdown("**Flag / 标志:** Refer — Extensive (E3)")
    else:
        st.info(result.e3_no_refer_message)
        st.markdown("**Flag / 标志:** No refer")


def _fig_toggle_key(cfg: dict) -> str:
    return f"show_figs_stage3_{config_profile(cfg)}"


def render_figures_panel(cfg: dict) -> None:
    top_k = int(cfg.get("figures", {}).get("pdp_top_k", 5))
    st.subheader("Global explainability (static)")
    st.caption(
        "Publication figures (precomputed; not updated per input).  "
        "Hidden by default to reduce Cloud memory use."
    )

    show = st.toggle(
        "Show SHAP / beeswarm / PDP figures  |  显示解释性图（蜂群图 / PDP 等）",
        value=False,
        key=_fig_toggle_key(cfg),
    )
    if not show:
        st.info(
            "Figures are collapsed to save memory. Turn the switch on to view.  \n"
            "默认折叠以节省内存；打开开关后可查看。"
        )
        return

    pub = list_publication_images(cfg)
    with st.expander("SHAP beeswarm / 蜂群图", expanded=True):
        if not pub:
            st.caption("No beeswarm figures found.")
        for path, caption in pub:
            st.image(str(path), caption=caption, use_container_width=True)

    pdp_images = list_publication_pdp_rank_images(cfg, top_k=top_k)
    with st.expander("Partial dependence (PDP) / 偏依赖图", expanded=False):
        if not pdp_images:
            st.caption("Top-5 PDP images not found.")
        for path, caption in pdp_images:
            st.image(str(path), caption=caption, use_container_width=True)


def render_enlarged_correlation(cfg: dict) -> None:
    if not st.session_state.get(_fig_toggle_key(cfg), False):
        return
    enlarged = get_enlarged_image(cfg)
    if not enlarged:
        return
    path, caption = enlarged
    with st.expander(f"{caption} (full width) / 全宽相关图", expanded=False):
        st.caption("Full-width view / 全宽显示便于阅读")
        st.image(str(path), use_container_width=True)


def render_western_stage3(cfg: dict, artifacts: MulticlassStageArtifacts) -> None:
    profile = config_profile(cfg)
    feature_meta = get_stage_feature_meta(profile, "stage3")

    left, right = st.columns([1.2, 0.85], gap="large")
    with left:
        st.subheader(artifacts.model_cfg["display_name"])
        values, errors = feature_input_form(
            artifacts.feature_names,
            feature_meta,
            form_key=f"form_stage3_{profile}",
        )
        if errors:
            for err in errors:
                st.error(err)
        elif values is not None:
            result = predict_multiclass(artifacts, values)
            render_multiclass_result(result, artifacts)

        render_variable_reference_bilingual(
            artifacts.feature_names,
            feature_meta,
            footer_zh=cfg.get("variable_reference_footer", {}).get("zh"),
            footer_en=cfg.get("variable_reference_footer", {}).get("en"),
        )

    with right:
        render_metrics_sidebar(artifacts)
        st.divider()
        render_figures_panel(cfg)

    render_enlarged_correlation(cfg)
