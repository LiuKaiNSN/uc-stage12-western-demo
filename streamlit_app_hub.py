"""Paper 2 Hub — Baseline Model + TCM Integrated Model (Stage 1–2).

Does NOT replace Paper 1 entry ``streamlit_app.py``.
Streamlit Cloud: create a *second* app with Main file = this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from src.artifacts import validate_config
from src.config_loader import load_config
from src.inference import load_both_stages
from src.ui.western_stage12 import render_western_stage12

MODULES = {
    "Baseline Model (Stage 1–2) / 基线模型": {
        "profile": "stage12_baseline",
        "config": "baseline_stage12.yaml",
        "title_en": "Baseline Model — Pre-endoscopic Stage 1 & 2",
        "title_zh": "基线模型 — 内镜前 Stage 1 & 2",
    },
    "TCM Integrated Model (Stage 1–2) / 中西医融合模型": {
        "profile": "stage12_tcm",
        "config": "tcm_stage12.yaml",
        "title_en": "TCM Integrated Model — Pre-endoscopic Stage 1 & 2",
        "title_zh": "中西医融合模型 — 内镜前 Stage 1 & 2",
    },
}

st.set_page_config(
    page_title="UC Stage 1–2 Hub (Baseline | TCM Integrated)",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.title("Stage 1–2 Hub")
    st.caption("Paper 2 interactive demo / 第二篇论文演示平台")
    choice = st.radio(
        "Select model / 选择模型",
        list(MODULES.keys()),
        index=0,
    )
    st.divider()
    st.markdown(
        "**Note:** This hub is separate from the Paper 1 Baseline-only URL.  \n"
        "**说明：** 本 Hub 与第一篇论文的独立演示链接分离；第一篇入口未改动。"
    )

entry = MODULES[choice]
cfg = load_config(name=entry["config"], profile=entry["profile"])

with st.sidebar:
    st.markdown(f"**Active:** {entry['title_en']}")
    with st.expander("Disclaimer / 免责声明", expanded=True):
        st.markdown(cfg["disclaimer"]["en"])
        st.markdown(cfg["disclaimer"]["zh"])

errors = validate_config(cfg)
if errors:
    st.error("Startup validation failed:\n\n" + "\n".join(f"- {e}" for e in errors))
    st.stop()


@st.cache_resource(show_spinner="Loading models… / 正在加载模型…")
def _load(profile: str, config_name: str):
    loaded = load_config(name=config_name, profile=profile)
    return load_both_stages(loaded), loaded


artifacts_map, active_cfg = _load(entry["profile"], entry["config"])

st.header(entry["title_en"])
st.header(entry["title_zh"])
st.markdown(
    "**Stage 1:** IE–FDIBS spectrum vs organic colorectal disease (UC/CD/IC/CRC).  "
    "**Stage 2:** UC vs other organic diseases (CD/IC/CRC).  "
    "Stages validated separately (W2); not a probability cascade.  "
    "**阶段分段外验（W2），非概率级联。**"
)

render_western_stage12(active_cfg, artifacts_map)
