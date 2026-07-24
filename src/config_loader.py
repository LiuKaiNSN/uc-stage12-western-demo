from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .paths import SCRIPT_WEB_ROOT


def profile_config_dir(profile: str = "stage12_western") -> Path:
    return SCRIPT_WEB_ROOT / profile / "configs"


def load_config(
    name: str = "western_stage12.yaml",
    profile: str = "stage12_western",
) -> Dict[str, Any]:
    path = profile_config_dir(profile) / name
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config format: {path}")
    data["_profile"] = profile
    return data


def config_profile(cfg: dict) -> str:
    return str(cfg.get("_profile") or cfg.get("profile_id") or "stage12_western")
