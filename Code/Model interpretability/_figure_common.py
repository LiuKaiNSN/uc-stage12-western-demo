# -*- coding: utf-8 -*-
"""Shared helpers for publication figure scripts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from _pub_plot_style import MODEL_INDEX_MAP

RUN_DIR_PATTERN = re.compile(r"^(\d+)", re.IGNORECASE)


def detect_model_dirs(stage_root: Path) -> Dict[int, Path]:
    stage_root = Path(stage_root)
    candidates: Dict[int, List[Path]] = {}
    for p in stage_root.iterdir():
        if not p.is_dir():
            continue
        m = RUN_DIR_PATTERN.match(p.name)
        if not m:
            continue
        idx = int(m.group(1))
        if idx in MODEL_INDEX_MAP:
            candidates.setdefault(idx, []).append(p)
    resolved: Dict[int, Path] = {}
    for idx, paths in candidates.items():
        paths_sorted = sorted(paths, key=lambda x: x.stat().st_mtime, reverse=True)
        resolved[idx] = paths_sorted[0]
    return resolved


def model_name(idx: int) -> str:
    return MODEL_INDEX_MAP.get(idx, f"Model {idx}")


def load_external_report(run_dir: Path) -> dict:
    path = run_dir / "external_validation_binary" / "external_validation_report.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_external_predictions(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "external_validation_binary" / "external_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2:
        auc = float("nan")
        return auc, auc, auc
    auc = float(roc_auc_score(y_true, y_score))
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boots: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        boots.append(float(roc_auc_score(yt, y_score[idx])))
    if not boots:
        return auc, auc, auc
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return auc, float(lo), float(hi)


def metrics_table_from_reports(
    stage_root: Path,
    metric_keys: Optional[List[str]] = None,
) -> pd.DataFrame:
    metric_keys = metric_keys or ["auc", "auprc", "acc", "f1", "sensitivity", "specificity"]
    rows: List[dict] = []
    for idx, run_dir in sorted(detect_model_dirs(stage_root).items()):
        report = load_external_report(run_dir)
        overall = report.get("overall", {})
        m = dict(overall.get("metrics", {}))
        pair = overall.get("pairwise_binary", [])
        if pair:
            for k in ("sensitivity", "specificity"):
                if k not in m and k in pair[0]:
                    m[k] = pair[0][k]
        row = {"model_index": idx, "model": model_name(idx), "run_dir": str(run_dir)}
        for k in metric_keys:
            row[k] = m.get(k, pair[0].get(k) if pair else np.nan)
        row["n"] = overall.get("n_samples")
        rows.append(row)
    return pd.DataFrame(rows)


def subgroup_auc_rows(
    df: pd.DataFrame,
    group_col: str,
    y_col: str = "y_true",
    score_col: str = "prob_1",
    min_n: int = 20,
    n_boot: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    rows: List[dict] = []
    for g, sub in df.groupby(group_col, dropna=False):
        if len(sub) < min_n:
            continue
        y = sub[y_col].astype(int).to_numpy()
        s = sub[score_col].astype(float).to_numpy()
        if len(np.unique(y)) < 2:
            continue
        auc, lo, hi = bootstrap_auc_ci(y, s, n_boot=n_boot, seed=seed)
        rows.append(
            {
                "subgroup": str(g),
                "n": int(len(sub)),
                "auc": auc,
                "auc_lo": lo,
                "auc_hi": hi,
            }
        )
    return pd.DataFrame(rows)
