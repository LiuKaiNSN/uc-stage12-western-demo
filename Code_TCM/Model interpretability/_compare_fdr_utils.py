# -*- coding: utf-8 -*-
"""TCM vs Western 成对比较：确认性 ML 模型（1–6）AUC 的 BH-FDR 工具。"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

DEFAULT_CONFIRMATORY_ML_INDICES: tuple[int, ...] = (1, 2, 3, 4, 5, 6)


def parse_confirmatory_indices(text: str) -> List[int]:
    out: List[int] = []
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def benjamini_hochberg_fdr(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini–Hochberg FDR q-values; NaN p 保持 NaN。"""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    q = np.full(n, np.nan, dtype=float)
    valid = np.isfinite(p)
    if valid.sum() == 0:
        return q
    pv = p[valid]
    m = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    adj = np.empty(m, dtype=float)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * m / rank
        prev = min(prev, val)
        adj[i] = prev
    adj = np.clip(adj, 0.0, 1.0)
    q_valid = np.empty(m, dtype=float)
    q_valid[order] = adj
    q[np.where(valid)[0]] = q_valid
    return q


def add_confirmatory_auc_fdr(
    df: pd.DataFrame,
    p_raw_col: str,
    *,
    confirmatory_indices: Iterable[int] = DEFAULT_CONFIRMATORY_ML_INDICES,
    p_fdr_col: str = "auc_p_fdr_bh",
    p_raw_out_col: str = "auc_p_raw",
) -> pd.DataFrame:
    """
    在完整结果表上为确认性模型追加 ``auc_p_raw`` / ``auc_p_fdr_bh``。
    FDR 仅在同一 ``dataset`` 内、指定 model_index 子集上计算（通常 6 个 ML）。
    非确认性模型 ``auc_p_fdr_bh`` 为 NaN。
    """
    out = df.copy()
    if p_raw_out_col not in out.columns:
        out[p_raw_out_col] = out[p_raw_col]
    out[p_fdr_col] = np.nan

    idx_set = set(int(i) for i in confirmatory_indices)
    for dataset, g in out.groupby("dataset", sort=False):
        sub = g[g["model_index"].isin(idx_set)].sort_values("model_index")
        if sub.empty:
            continue
        q = benjamini_hochberg_fdr(sub[p_raw_col].to_numpy(dtype=float))
        out.loc[sub.index, p_fdr_col] = q
    return out


def build_confirmatory_auc_fdr_summary(
    df: pd.DataFrame,
    p_raw_col: str,
    *,
    confirmatory_indices: Iterable[int] = DEFAULT_CONFIRMATORY_ML_INDICES,
    delta_col: str,
    metric_label: str,
) -> pd.DataFrame:
    """6 行确认性 AUC FDR 汇总表（含原始 p 与 BH-FDR q）。"""
    idx_set = set(int(i) for i in confirmatory_indices)
    rows: List[pd.DataFrame] = []
    base_cols = ["model_index", "model_name", "dataset", "n_samples", delta_col, p_raw_col]
    for dataset, g in df.groupby("dataset", sort=False):
        sub = g[g["model_index"].isin(idx_set)].sort_values("model_index").copy()
        if sub.empty:
            continue
        sub["auc_p_raw"] = sub[p_raw_col]
        sub["auc_p_fdr_bh"] = benjamini_hochberg_fdr(sub[p_raw_col].to_numpy(dtype=float))
        sub["confirmatory_metric"] = metric_label
        sub["fdr_family"] = f"ml_1-6_{metric_label}_{dataset}"
        keep = [c for c in base_cols if c in sub.columns] + [
            "auc_p_raw",
            "auc_p_fdr_bh",
            "confirmatory_metric",
            "fdr_family",
        ]
        rows.append(sub[keep])
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)
