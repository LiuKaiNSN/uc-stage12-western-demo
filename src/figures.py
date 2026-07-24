from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .paths import project_root_from_config, resolve_project_path


def _safe_feature_token(feature: str) -> str:
    token = "".join(c if c.isalnum() or c in "._-" else "_" for c in feature.strip())
    return token or "feature"


def _ranked_pdp_filename(rank: int, feature: str, ext: str) -> str:
    return f"rank{int(rank):02d}_{_safe_feature_token(feature)}{ext}"


def _resolve_image_path(path: Path) -> Optional[Path]:
    if path.is_file():
        return path
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = Path(f"{path}{ext}") if path.suffix == "" else None
        if candidate and candidate.is_file():
            return candidate
    if path.suffix == "":
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = path.parent / f"{path.name}{ext}"
            if candidate.is_file():
                return candidate
    stem = path.stem if path.suffix else path.name
    parent = path.parent if path.suffix else path
    if parent.is_dir():
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = parent / f"{stem}{ext}"
            if candidate.is_file():
                return candidate
    return None


def _resolve_image(base: Path, stem: str) -> Optional[Path]:
    return _resolve_image_path(base / stem)


def _resolve_pdp_image(pdp_dir: Path, rank: int, feature: str) -> Optional[Path]:
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = pdp_dir / _ranked_pdp_filename(rank, feature, ext)
        if candidate.is_file():
            return candidate
    return None


def _resolve_config_path(cfg: dict, rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute():
        return p
    return (project_root_from_config(cfg) / rel).resolve()


def list_interpretability_panel(cfg: dict, stage_key: str) -> List[Tuple[Path, str]]:
    """SHAP panel images from model run interpretability folder."""
    stage_cfg = cfg["stages"][stage_key]
    interp_dir = resolve_project_path(cfg, stage_cfg["interpretability_dir"])
    panels: List[Tuple[Path, str]] = []

    for item in cfg.get("figures", {}).get("panel_images", []):
        sub = interp_dir / str(item["subdir"])
        stem = str(item["stem"])
        caption = str(item.get("caption", stem))
        found = _resolve_image(sub, stem)
        if found is not None:
            panels.append((found, caption))

    return panels


def list_publication_images(cfg: dict) -> List[Tuple[Path, str]]:
    """Publication figure paths (Stage3 Figure folder layout)."""
    fig_cfg = cfg.get("figures", {})
    out: List[Tuple[Path, str]] = []

    for item in fig_cfg.get("publication_images", []):
        rel = str(item["path"])
        caption = str(item.get("caption", rel))
        found = _resolve_image_path(_resolve_config_path(cfg, rel))
        if found is not None:
            out.append((found, caption))
    return out


def list_top_pdp_images(cfg: dict, stage_key: str, top_k: int = 5) -> List[Tuple[Path, str]]:
    """Top-K PDP from run interpretability (Stage12)."""
    stage_cfg = cfg["stages"][stage_key]
    interp_dir = resolve_project_path(cfg, stage_cfg["interpretability_dir"])
    pdp_dir = interp_dir / "marginal_effects" / "pdp"

    top_file_name = stage_cfg.get("top_features_file", "top_features_for_pdp.csv")
    top_path = interp_dir / top_file_name
    if not top_path.is_file():
        alt = interp_dir / "top_features_for_marginal_effects.csv"
        top_path = alt if alt.is_file() else top_path

    if not top_path.is_file() or not pdp_dir.is_dir():
        return []

    top = pd.read_csv(top_path).head(top_k)
    result: List[Tuple[Path, str]] = []
    for _, row in top.iterrows():
        rank = int(row["rank"])
        feat = str(row["feature"])
        img = _resolve_pdp_image(pdp_dir, rank, feat)
        if img is not None:
            result.append((img, f"PDP rank {rank}: {feat}"))
    return result


def list_publication_pdp_rank_images(cfg: dict, top_k: int = 5) -> List[Tuple[Path, str]]:
    """Glob rank01–rank0K in publication pdp_dir (Stage3 figure3)."""
    fig_cfg = cfg.get("figures", {})
    pdp_rel = fig_cfg.get("pdp_dir")
    if not pdp_rel:
        return []
    pdp_dir = _resolve_config_path(cfg, str(pdp_rel))
    if not pdp_dir.is_dir():
        return []

    found: List[Tuple[int, Path]] = []
    for rank in range(1, top_k + 1):
        prefix = f"rank{rank:02d}_"
        for ext in (".jpg", ".jpeg", ".png"):
            for path in sorted(pdp_dir.glob(f"{prefix}*{ext}")):
                found.append((rank, path))
                break

    found.sort(key=lambda x: x[0])
    return [(path, f"PDP rank {rank:02d}: {path.stem}") for rank, path in found]


def _stage_publication_cfg(cfg: dict, stage_key: str) -> Dict[str, Any]:
    pub = cfg.get("figures", {}).get("stage_publication", {})
    block = pub.get(stage_key, {})
    return block if isinstance(block, dict) else {}


def list_stage_publication_beeswarm(cfg: dict, stage_key: str) -> List[Tuple[Path, str]]:
    """Beeswarm from Figure/Stage12_TCM figure5/stage* (TCM Stage12)."""
    block = _stage_publication_cfg(cfg, stage_key)
    rel = block.get("beeswarm")
    if not rel:
        return []
    caption = str(block.get("beeswarm_caption", "SHAP beeswarm (external validation)"))
    found = _resolve_image_path(_resolve_config_path(cfg, str(rel)))
    return [(found, caption)] if found else []


def list_stage_publication_pdp(cfg: dict, stage_key: str, top_k: int = 5) -> List[Tuple[Path, str]]:
    block = _stage_publication_cfg(cfg, stage_key)
    pdp_rel = block.get("pdp_dir")
    if not pdp_rel:
        return []
    pdp_dir = _resolve_config_path(cfg, str(pdp_rel))
    if not pdp_dir.is_dir():
        return []

    found: List[Tuple[int, Path]] = []
    for rank in range(1, top_k + 1):
        prefix = f"rank{rank:02d}_"
        for ext in (".jpg", ".jpeg", ".png"):
            for path in sorted(pdp_dir.glob(f"{prefix}*{ext}")):
                found.append((rank, path))
                break

    found.sort(key=lambda x: x[0])
    return [(path, f"PDP rank {rank:02d}: {path.stem}") for rank, path in found]


def get_stage_publication_enlarge(cfg: dict, stage_key: str) -> Optional[Tuple[Path, str]]:
    block = _stage_publication_cfg(cfg, stage_key)
    rel = block.get("enlarge") or block.get("enlarge_path")
    if not rel:
        return None
    caption = str(
        block.get("enlarge_caption", "SHAP importance & feature correlation (external validation)")
    )
    found = _resolve_image_path(_resolve_config_path(cfg, str(rel)))
    return (found, caption) if found else None


def get_enlarged_image(cfg: dict, stage_key: Optional[str] = None) -> Optional[Tuple[Path, str]]:
    fig_cfg = cfg.get("figures", {})

    if stage_key:
        pub_enlarge = get_stage_publication_enlarge(cfg, stage_key)
        if pub_enlarge:
            return pub_enlarge

    if "enlarge_path" in fig_cfg:
        rel = str(fig_cfg["enlarge_path"])
        caption = str(fig_cfg.get("enlarge_caption", "SHAP importance & correlation"))
        found = _resolve_image_path(_resolve_config_path(cfg, rel))
        if found:
            return found, caption
        return None

    enlarge_stem = str(fig_cfg.get("enlarge_stem", "shap_importance_correlation"))
    if stage_key and "stages" in cfg:
        panels = list_interpretability_panel(cfg, stage_key)
        for path, caption in panels:
            if path.stem == enlarge_stem:
                return path, caption
    return None
