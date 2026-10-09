"""Thesis Chapter-4 full system — Stage 1/2/3 × Baseline | TCM Integrated.

THIRD Streamlit Cloud entry. Does NOT replace:
  - streamlit_app.py      (Paper 1 SCI URL)
  - streamlit_app_hub.py  (Paper 2 SCI Hub URL)

Cloud: create a *new* app with Main file = this file.
Loads only the currently selected system × stage (memory-safe).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from src.artifacts import load_stage_artifacts, validate_config
from src.config_loader import config_profile, load_config
from src.feature_labels import get_stage_feature_meta
from src.inference import load_stage3_model
from src.ui.western_stage12 import render_stage_tab
from src.ui.western_stage3 import render_western_stage3

SYSTEMS = {
    "Baseline Model / 基线模型（西医客观指标）": {
        "key": "baseline",
        "stage12": {"profile": "stage12_baseline", "config": "baseline_stage12.yaml"},
        "stage3": {"profile": "stage3_western", "config": "western_stage3.yaml"},
        "title_en": "Baseline Model",
        "title_zh": "基线模型",
    },
    "TCM Integrated Model / 中西医融合模型": {
        "key": "tcm",
        "stage12": {"profile": "stage12_tcm", "config": "tcm_stage12.yaml"},
        "stage3": {"profile": "stage3_tcm", "config": "tcm_stage3.yaml"},
        "title_en": "TCM Integrated Model",
        "title_zh": "中西医融合模型",
    },
}

STAGE_OPTIONS = {
    "stage1": "Stage 1 — IE/FDIBS vs Organic / 有机病分流",
    "stage2": "Stage 2 — UC vs Other organic / UC 定向鉴别",
    "stage3": "Stage 3 — Montreal E1/E2/E3 / 病变范围",
}

st.set_page_config(
    page_title="UC Full Assist System (Stage 1–3 | Baseline | TCM)",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.title("全模型辅助诊断系统")
    st.caption("Thesis Ch.4 / 博士论文第四章主演示入口")
    system_label = st.radio(
        "Feature system / 特征体系",
        list(SYSTEMS.keys()),
        index=0,
    )
    stage_key = st.radio(
        "Clinical stage / 临床阶段",
        list(STAGE_OPTIONS.keys()),
        format_func=lambda k: STAGE_OPTIONS[k],
        index=0,
    )
    st.divider()
    st.markdown(
        "**Note:** This full app is separate from Paper 1 / Paper 2 SCI demo URLs.  \n"
        "**说明：** 本全模型入口与 SCI 论文中的两个演示链接分离，不替换旧链接。"
    )
    with st.expander("Disclaimer / 免责声明", expanded=False):
        st.markdown(
            "Research / thesis demonstration only. Not for standalone clinical diagnosis. "
            "Endoscopy and histopathology retain priority. Inputs are not stored.  \n"
            "仅供科研与学位论文演示；不得用于独立临床诊断；内镜与病理优先；输入不落盘。"
        )

system = SYSTEMS[system_label]

# Keep at most ONE model in RAM: clear cache whenever system or stage changes.
# Switching back still works; the model reloads (a few–teen seconds).
active_slot = f"{system['key']}:{stage_key}"
prev_slot = st.session_state.get("full_active_slot")
if prev_slot is not None and prev_slot != active_slot:
    st.cache_resource.clear()
st.session_state.full_active_slot = active_slot


@st.cache_resource(show_spinner=False, max_entries=1)
def _load_binary(profile: str, config_name: str, stage: str):
    cfg = load_config(name=config_name, profile=profile)
    return load_stage_artifacts(cfg, stage), cfg


@st.cache_resource(show_spinner=False, max_entries=1)
def _load_stage3(profile: str, config_name: str):
    cfg = load_config(name=config_name, profile=profile)
    return load_stage3_model(cfg), cfg


st.header(f"{system['title_en']} — Full Stage 1–3 System")
st.header(f"{system['title_zh']} — 全阶段辅助诊断系统")
st.markdown(
    "**Stage 1:** IE–FDIBS vs organic (UC/CD/IC/CRC).  "
    "**Stage 2:** UC vs other organic.  "
    "**Stage 3:** Montreal extent E1 / E2 / E3 (among UC).  "
    "Stages validated separately; **not** a probability cascade.  "
    "**各阶段分段外验，非概率级联。**"
)
st.caption(
    f"Active: **{STAGE_OPTIONS[stage_key]}** — only this model is kept in memory.  "
    "Switching modules reloads the model (may take several seconds).  "
    "仅常驻当前模块；切换阶段/体系会重新加载（可能需数秒至十几秒）。"
)

if stage_key in ("stage1", "stage2"):
    entry = system["stage12"]
    cfg_probe = load_config(name=entry["config"], profile=entry["profile"])
    errors = validate_config(cfg_probe)
    if errors:
        st.error("Startup validation failed:\n\n" + "\n".join(f"- {e}" for e in errors))
        st.stop()
    with st.spinner(f"Loading {STAGE_OPTIONS[stage_key]} …"):
        artifacts, active_cfg = _load_binary(entry["profile"], entry["config"], stage_key)
    profile = config_profile(active_cfg)
    meta = get_stage_feature_meta(profile, stage_key)
    render_stage_tab(active_cfg, artifacts, stage_key, meta)
else:
    entry = system["stage3"]
    cfg_probe = load_config(name=entry["config"], profile=entry["profile"])
    errors = validate_config(cfg_probe)
    if errors:
        st.error("Startup validation failed:\n\n" + "\n".join(f"- {e}" for e in errors))
        st.stop()
    with st.spinner("Loading Stage 3 model… / 正在加载 Stage 3 模型…"):
        artifacts, active_cfg = _load_stage3(entry["profile"], entry["config"])
    render_western_stage3(active_cfg, artifacts)
