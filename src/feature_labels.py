from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from .paths import SCRIPT_WEB_ROOT


def load_feature_labels(profile: str = "stage12_western") -> Dict[str, Dict[str, Any]]:
    path = SCRIPT_WEB_ROOT / profile / "configs" / "feature_labels.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Feature labels not found: {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid feature labels: {path}")
    return data


def get_stage_feature_meta(profile: str, stage_key: str) -> Dict[str, Dict[str, Any]]:
    all_labels = load_feature_labels(profile)
    stage = all_labels.get(stage_key, {})
    if not isinstance(stage, dict):
        return {}
    return stage


def label_zh(meta: Dict[str, Any], fallback: str = "") -> str:
    return str(meta.get("label_zh") or meta.get("label") or fallback)


def label_en(meta: Dict[str, Any], fallback: str = "") -> str:
    return str(meta.get("label_en") or meta.get("scoring") or fallback)


def scoring_zh(meta: Dict[str, Any]) -> str:
    return str(meta.get("scoring_zh") or meta.get("scoring") or "—")


def scoring_en(meta: Dict[str, Any]) -> str:
    return str(meta.get("scoring_en") or meta.get("scoring") or "—")


def option_display(opt: Dict[str, Any]) -> str:
    if "short_zh" in opt and "short_en" in opt:
        return f"{opt['short_zh']}  |  {opt['short_en']}"
    return str(opt.get("short") or opt.get("short_zh") or "")


def option_value_map(meta: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for opt in meta.get("options") or []:
        out[option_display(opt)] = float(opt["value"])
    return out


def option_choices(meta: Dict[str, Any]) -> list[str]:
    return [option_display(o) for o in meta.get("options") or []]


def continuous_min(meta: Dict[str, Any]) -> float:
    return float(meta.get("min_value", 0.0001))
