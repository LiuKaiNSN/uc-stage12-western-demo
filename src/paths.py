from __future__ import annotations

from pathlib import Path

# Repo root (parent of `src/`) — self-contained Streamlit Cloud bundle
SCRIPT_WEB_ROOT = Path(__file__).resolve().parents[1]


def project_root_from_config(cfg: dict) -> Path:
    root = Path(cfg["project_root"])
    if root.is_absolute():
        return root.resolve()
    return (SCRIPT_WEB_ROOT / root).resolve()


def resolve_project_path(cfg: dict, rel_path: str) -> Path:
    return (project_root_from_config(cfg) / rel_path).resolve()


def web_output_dir(cfg: dict) -> Path:
    rel = cfg["paths"]["web_output"]
    root = project_root_from_config(cfg)
    return (root / rel).resolve()
