"""Streamlit Cloud entry — Western Stage 1+2 pre-endoscopic UC triage demo."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from src.artifacts import load_stage_artifacts, validate_config
from src.config_loader import load_config
from src.ui.western_stage12 import render_western_stage12

PROFILE = "stage12_western"

st.set_page_config(
    page_title="UC Pre-endoscopic Triage (Stage 1–2)",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

cfg = load_config(profile=PROFILE)

with st.sidebar:
    st.title("Western Stage 1+2")
    st.caption(cfg.get("display_name", ""))
    st.markdown("**Pre-endoscopic decision support / 内镜前辅助决策**")
    st.divider()
    with st.expander("Disclaimer / 免责声明", expanded=False):
        st.markdown(cfg["disclaimer"]["en"])
        st.markdown(cfg["disclaimer"]["zh"])

errors = validate_config(cfg)
if errors:
    st.error("Startup validation failed:\n\n" + "\n".join(f"- {e}" for e in errors))
    st.stop()


@st.cache_resource(show_spinner=False)
def _load_stage(stage_key: str):
    """Load one stage pipeline; Stage 2 stays unloaded until selected."""
    return load_stage_artifacts(cfg, stage_key)


st.header("UC Multistage Pre-endoscopic Triage — Western Models")
st.header("溃疡性结肠炎序贯内镜前分流 — 西医客观指标模型")
st.markdown(
    "**Stage 1:** IE–FDIBS spectrum vs organic colorectal disease (UC/CD/IC/CRC).  "
    "**Stage 2:** UC vs other organic diseases (CD/IC/CRC).  "
    "Stages validated separately (W2); not a probability cascade.  "
    "**阶段分段外验（W2），非概率级联。**"
)

render_western_stage12(cfg, _load_stage)
