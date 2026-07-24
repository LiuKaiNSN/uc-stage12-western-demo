from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from ..feature_labels import (
    continuous_min,
    label_en,
    label_zh,
    option_choices,
    option_value_map,
    scoring_en,
    scoring_zh,
)

_PLACEHOLDER = "— 请选择 / Please select —"


def render_bilingual_field_heading(meta: Dict[str, Any], feat: str) -> None:
    st.markdown(f"**{label_zh(meta, feat)}**")
    st.caption(label_en(meta, feat))


def validate_continuous(name: str, value: Optional[float], min_v: float) -> Optional[str]:
    if value is None:
        return f"{name}: 请填写大于 0 的数值 / Enter a value > 0."
    try:
        v = float(value)
    except (TypeError, ValueError):
        return f"{name}: 无效数值 / Invalid number."
    if v <= 0:
        return f"{name}: 须为大于 0 的数值 / Must be > 0."
    if v < min_v:
        return f"{name}: 须 ≥ {min_v:g} / Must be ≥ {min_v:g}."
    return None


def feature_input_form(
    feature_names: List[str],
    feature_meta: Dict[str, Dict[str, Any]],
    form_key: str,
) -> Tuple[Optional[Dict[str, float]], List[str]]:
    errors: List[str] = []
    values: Dict[str, float] = {}

    with st.form(form_key, clear_on_submit=False):
        st.caption(
            "连续变量须 > 0；分类变量请从下拉列表选择（编码与建模一致）。"
            " / Continuous inputs must be > 0; categorical codes match training."
        )
        cols = st.columns(2)
        raw_inputs: Dict[str, Any] = {}

        for i, feat in enumerate(feature_names):
            meta = feature_meta.get(feat, {})
            input_type = meta.get("input_type", "continuous")
            with cols[i % 2]:
                render_bilingual_field_heading(meta, feat)
                if input_type in ("ordinal", "binary"):
                    options = [_PLACEHOLDER] + option_choices(meta)
                    raw_inputs[feat] = st.selectbox(
                        "选项 / Option",
                        options,
                        key=f"{form_key}_{feat}",
                        label_visibility="collapsed",
                    )
                else:
                    min_v = continuous_min(meta)
                    raw_inputs[feat] = st.number_input(
                        "数值 / Value",
                        min_value=min_v,
                        value=None,
                        format="%.6f",
                        key=f"{form_key}_{feat}",
                        label_visibility="collapsed",
                    )

        submitted = st.form_submit_button("Calculate probability / 计算概率", type="primary")

    if not submitted:
        return None, errors

    for feat in feature_names:
        meta = feature_meta.get(feat, {})
        input_type = meta.get("input_type", "continuous")
        raw = raw_inputs.get(feat)
        display_name = label_zh(meta, feat)

        if input_type in ("ordinal", "binary"):
            if raw is None or raw == _PLACEHOLDER:
                errors.append(f"{display_name}: 请选择 / Please select an option.")
                continue
            mapping = option_value_map(meta)
            if raw not in mapping:
                errors.append(f"{display_name}: 无效选项 / Invalid selection.")
                continue
            values[feat] = mapping[raw]
        else:
            err = validate_continuous(display_name, raw, continuous_min(meta))
            if err:
                errors.append(err)
            else:
                values[feat] = float(raw)

    if errors:
        return None, errors
    return values, errors


def render_variable_reference_bilingual(
    feature_names: List[str],
    feature_meta: Dict[str, Dict[str, Any]],
    footer_zh: Optional[str] = None,
    footer_en: Optional[str] = None,
) -> None:
    st.markdown("#### 变量赋分说明 / Variable scoring reference")

    rows_zh = []
    rows_en = []
    for feat in feature_names:
        meta = feature_meta.get(feat, {})
        input_type = meta.get("input_type", "continuous")
        type_zh = {"continuous": "连续", "ordinal": "等级", "binary": "二分类"}.get(input_type, input_type)
        rows_zh.append({
            "变量": label_zh(meta, feat),
            "类型": type_zh,
            "赋分说明": scoring_zh(meta),
        })
        rows_en.append({
            "Variable": label_en(meta, feat),
            "Type": input_type,
            "Scoring": scoring_en(meta),
        })

    col_zh, col_en = st.columns(2)
    with col_zh:
        st.markdown("**中文表**")
        st.table(rows_zh)
    with col_en:
        st.markdown("**English table**")
        st.table(rows_en)

    if footer_zh or footer_en:
        st.markdown("---")
        if footer_zh:
            st.caption(footer_zh)
        if footer_en:
            st.caption(footer_en)
