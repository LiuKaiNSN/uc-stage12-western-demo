# -*- coding: utf-8 -*-
"""
Stage2 二分类：中医融合 vs 西医基础 —— 10 算法成对比较（OOF + 外部验证）。

主分析：Bootstrap 配对 ΔAUC（AUC_TCM − AUC_Western，2000 次，seed=42，95% CI，双侧 p）。
确认性 FDR：模型 1–6、仅 AUC，Benjamini–Hochberg（OOF / 外部各 6 次检验为一族）。
补充：DeLong 检验（同一批样本上两模型 AUC 是否不同，双侧 p）。

数据对齐：
- OOF：``oof_predictions.csv``，按 ``row_index`` 合并，概率列 ``y_prob``（P(UC)）。
- 外部：``external_validation_binary/external_predictions.csv``，按 ``No`` 合并，``prob_1``。

输出目录（默认）::
  F:\\KeTi\\Project\\Stage2_TCM_vs_Western_comparison\\comparison_YYYYMMDD_HHMMSS\\

启动示例::

  python "F:\\KeTi\\Project\\Script\\compare_stage2_tcm_vs_western.py"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from _compare_fdr_utils import (
    add_confirmatory_auc_fdr,
    build_confirmatory_auc_fdr_summary,
    parse_confirmatory_indices,
)

MODEL_INDEX_MAP: Dict[int, str] = {
    1: "XGBoost",
    2: "LR",
    3: "LightGBM",
    4: "CatBoost",
    5: "SVM",
    6: "RF",
    7: "MLP",
    8: "DCNV2",
    9: "FT-Transformer",
    10: "TabTransformer",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage2 TCM-integrated vs Western baseline pairwise comparison")
    p.add_argument(
        "--western-root",
        type=str,
        default=r"F:\KeTi\Project\outputs\Stage2",
        help="西医 Stage2 产出根目录",
    )
    p.add_argument(
        "--tcm-root",
        type=str,
        default=r"F:\KeTi\Project\TCM\output\Stage2",
        help="中医融合 Stage2 产出根目录",
    )
    p.add_argument(
        "--output-root",
        type=str,
        default=r"F:\KeTi\Project\Stage2_TCM_vs_Western_comparison",
        help="比较结果根目录（其下新建 comparison_时间戳 子文件夹）",
    )
    p.add_argument("--bootstrap-iter", type=int, default=2000)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    p.add_argument("--id-col-external", type=str, default="No")
    p.add_argument(
        "--confirmatory-indices",
        type=str,
        default="1,2,3,4,5,6",
        help="确认性 ML 模型序号（仅对这些模型的 AUC p 做 BH-FDR）",
    )
    return p.parse_args()


def _detect_model_dirs(root: Path) -> Dict[int, Path]:
    candidates: Dict[int, List[Path]] = {}
    for p in root.iterdir():
        if not p.is_dir():
            continue
        m = re.match(r"^(\d+)", p.name)
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


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j2 = i
        while j2 < n and z[j2] == z[i]:
            j2 += 1
        t[i:j2] = 0.5 * (i + j2 - 1)
        i = j2
    t2 = np.empty(n, dtype=float)
    t2[j] = t + 1
    return t2


def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int) -> Tuple[np.ndarray, np.ndarray]:
    """Two-row prediction matrix (2 x n), samples sorted: positives first."""
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    pos = predictions_sorted_transposed[:, :m]
    neg = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(pos[r, :])
        ty[r, :] = _compute_midrank(neg[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    if k == 1:
        sx = np.array([[float(sx)]])
        sy = np.array([[float(sy)]])
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_paired_test(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
) -> Dict[str, float]:
    """DeLong test: score_a vs score_b on same subjects (A=Western, B=TCM in our convention)."""
    y_true = np.asarray(y_true, dtype=int)
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)
    label_1_count = int(y_true.sum())
    n = len(y_true)
    if label_1_count == 0 or label_1_count == n:
        return {
            "auc_a": float("nan"),
            "auc_b": float("nan"),
            "delta_auc": float("nan"),
            "delong_z": float("nan"),
            "delong_p_two_sided": float("nan"),
        }

    order = np.argsort(-y_true)
    preds = np.vstack([score_a, score_b])[:, order]
    aucs, cov = _fast_delong(preds, label_1_count)
    delta = float(aucs[0] - aucs[1])
    var_delta = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    if var_delta <= 0 or not np.isfinite(var_delta):
        z = float("nan")
        p = float("nan")
    else:
        z = delta / np.sqrt(var_delta)
        p = float(2.0 * stats.norm.sf(abs(z)))
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "delta_auc": delta,
        "delong_z": float(z),
        "delong_p_two_sided": p,
    }


def bootstrap_delta_auc(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    n_boot: int,
    seed: int,
) -> Dict[str, float]:
    """Paired bootstrap for ΔAUC = AUC_B − AUC_A (TCM − Western when B=TCM).

    Absolute AUC 95% CIs for arms A/B are taken from the *same* resampled
    replicates (same seed, same n_boot, same valid draws) as ΔAUC.
    """
    y_true = np.asarray(y_true, dtype=int)
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)
    n = len(y_true)

    auc_a = _safe_auc(y_true, score_a)
    auc_b = _safe_auc(y_true, score_b)
    delta_obs = auc_b - auc_a

    rng = np.random.default_rng(seed)
    boot_deltas: List[float] = []
    boot_auc_a: List[float] = []
    boot_auc_b: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ys = y_true[idx]
        if len(np.unique(ys)) < 2:
            continue
        try:
            da = float(roc_auc_score(ys, score_a[idx]))
            db = float(roc_auc_score(ys, score_b[idx]))
            boot_auc_a.append(da)
            boot_auc_b.append(db)
            boot_deltas.append(db - da)
        except ValueError:
            continue

    nan_block = {
        "auc_western": auc_a,
        "auc_tcm": auc_b,
        "delta_auc": delta_obs,
        "bootstrap_auc_western_ci_low": float("nan"),
        "bootstrap_auc_western_ci_high": float("nan"),
        "bootstrap_auc_tcm_ci_low": float("nan"),
        "bootstrap_auc_tcm_ci_high": float("nan"),
        "bootstrap_delta_ci_low": float("nan"),
        "bootstrap_delta_ci_high": float("nan"),
        "bootstrap_p_two_sided": float("nan"),
        "bootstrap_n_valid": 0,
    }
    if not boot_deltas:
        return nan_block

    arr = np.asarray(boot_deltas, dtype=float)
    arr_a = np.asarray(boot_auc_a, dtype=float)
    arr_b = np.asarray(boot_auc_b, dtype=float)
    ci_low, ci_high = float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
    a_lo, a_hi = float(np.percentile(arr_a, 2.5)), float(np.percentile(arr_a, 97.5))
    b_lo, b_hi = float(np.percentile(arr_b, 2.5)), float(np.percentile(arr_b, 97.5))
    if delta_obs >= 0:
        p_two = 2.0 * float(np.mean(arr <= 0))
    else:
        p_two = 2.0 * float(np.mean(arr >= 0))
    p_two = min(1.0, p_two)

    return {
        "auc_western": auc_a,
        "auc_tcm": auc_b,
        "delta_auc": delta_obs,
        "bootstrap_auc_western_ci_low": a_lo,
        "bootstrap_auc_western_ci_high": a_hi,
        "bootstrap_auc_tcm_ci_low": b_lo,
        "bootstrap_auc_tcm_ci_high": b_hi,
        "bootstrap_delta_ci_low": ci_low,
        "bootstrap_delta_ci_high": ci_high,
        "bootstrap_p_two_sided": p_two,
        "bootstrap_n_valid": int(len(arr)),
    }


def _load_oof_pair(west_dir: Path, tcm_dir: Path) -> pd.DataFrame:
    w = pd.read_csv(west_dir / "oof_predictions.csv")
    t = pd.read_csv(tcm_dir / "oof_predictions.csv")
    merged = w.merge(t, on="row_index", suffixes=("_western", "_tcm"), how="inner")
    if not (merged["y_true_western"] == merged["y_true_tcm"]).all():
        raise ValueError(f"OOF y_true 不一致: {west_dir.name} vs {tcm_dir.name}")
    merged = merged.rename(columns={"y_true_western": "y_true"})
    return merged


def _load_external_pair(west_dir: Path, tcm_dir: Path, id_col: str) -> pd.DataFrame:
    w = pd.read_csv(west_dir / "external_validation_binary" / "external_predictions.csv")
    t = pd.read_csv(tcm_dir / "external_validation_binary" / "external_predictions.csv")
    if id_col not in w.columns or id_col not in t.columns:
        raise ValueError(f"外部预测缺少 {id_col!r}")
    merged = w.merge(t, on=id_col, suffixes=("_western", "_tcm"), how="inner")
    if not (merged["y_true_western"] == merged["y_true_tcm"]).all():
        raise ValueError(f"外部 y_true 不一致: {west_dir.name} vs {tcm_dir.name}")
    merged = merged.rename(columns={"y_true_western": "y_true"})
    return merged


def _compare_one(
    y_true: np.ndarray,
    score_western: np.ndarray,
    score_tcm: np.ndarray,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    boot = bootstrap_delta_auc(y_true, score_western, score_tcm, n_boot, seed)
    delong = delong_paired_test(y_true, score_western, score_tcm)
    out = {**boot}
    out["delong_z"] = delong["delong_z"]
    out["delong_p_two_sided"] = delong["delong_p_two_sided"]
    out["delong_auc_western"] = delong["auc_a"]
    out["delong_auc_tcm"] = delong["auc_b"]
    return out


def _preflight(west_dirs: Dict[int, Path], tcm_dirs: Dict[int, Path]) -> List[str]:
    errors: List[str] = []
    missing_w = sorted(set(MODEL_INDEX_MAP) - set(west_dirs))
    missing_t = sorted(set(MODEL_INDEX_MAP) - set(tcm_dirs))
    if missing_w:
        errors.append(f"西医目录缺失模型序号: {missing_w}")
    if missing_t:
        errors.append(f"中医目录缺失模型序号: {missing_t}")

    for idx in sorted(MODEL_INDEX_MAP):
        if idx not in west_dirs or idx not in tcm_dirs:
            continue
        for side, d in [("western", west_dirs[idx]), ("tcm", tcm_dirs[idx])]:
            oof = d / "oof_predictions.csv"
            ext = d / "external_validation_binary" / "external_predictions.csv"
            if not oof.exists():
                errors.append(f"[{idx}] {side} 缺少 {oof}")
            if not ext.exists():
                errors.append(f"[{idx}] {side} 缺少 {ext}")
    return errors


def main() -> None:
    args = parse_args()
    western_root = Path(args.western_root).resolve()
    tcm_root = Path(args.tcm_root).resolve()
    output_root = Path(args.output_root).resolve()

    west_dirs = _detect_model_dirs(western_root)
    tcm_dirs = _detect_model_dirs(tcm_root)
    errors = _preflight(west_dirs, tcm_dirs)
    if errors:
        raise FileNotFoundError("预检未通过:\n  " + "\n  ".join(errors))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = output_root / f"comparison_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    oof_rows: List[Dict[str, Any]] = []
    ext_rows: List[Dict[str, Any]] = []
    manifest: Dict[str, Any] = {
        "created_at": ts,
        "western_root": str(western_root),
        "tcm_root": str(tcm_root),
        "bootstrap_iter": int(args.bootstrap_iter),
        "bootstrap_seed": int(args.bootstrap_seed),
        "delta_auc_definition": "AUC_TCM - AUC_Western (positive favors TCM)",
        "pairs": {},
    }

    for idx in sorted(MODEL_INDEX_MAP):
        wdir = west_dirs[idx]
        tdir = tcm_dirs[idx]
        model_name = MODEL_INDEX_MAP[idx]
        pair_key = str(idx)
        manifest["pairs"][pair_key] = {
            "model_name": model_name,
            "western_run_dir": str(wdir),
            "tcm_run_dir": str(tdir),
        }

        oof = _load_oof_pair(wdir, tdir)
        y_oof = oof["y_true"].to_numpy(dtype=int)
        sw = oof["y_prob_western"].to_numpy(dtype=float)
        st = oof["y_prob_tcm"].to_numpy(dtype=float)
        oof_cmp = _compare_one(y_oof, sw, st, args.bootstrap_iter, args.bootstrap_seed)
        oof_rows.append(
            {
                "model_index": idx,
                "model_name": model_name,
                "dataset": "oof",
                "n_samples": len(oof),
                "n_pos_uc": int(y_oof.sum()),
                "western_run_dir": str(wdir),
                "tcm_run_dir": str(tdir),
                **oof_cmp,
            }
        )

        ext = _load_external_pair(wdir, tdir, args.id_col_external)
        y_ext = ext["y_true"].to_numpy(dtype=int)
        sw_e = ext["prob_1_western"].to_numpy(dtype=float)
        st_e = ext["prob_1_tcm"].to_numpy(dtype=float)
        ext_cmp = _compare_one(y_ext, sw_e, st_e, args.bootstrap_iter, args.bootstrap_seed)
        ext_rows.append(
            {
                "model_index": idx,
                "model_name": model_name,
                "dataset": "external",
                "n_samples": len(ext),
                "n_pos_uc": int(y_ext.sum()),
                "western_run_dir": str(wdir),
                "tcm_run_dir": str(tdir),
                **ext_cmp,
            }
        )

    oof_df = pd.DataFrame(oof_rows).sort_values("model_index")
    ext_df = pd.DataFrame(ext_rows).sort_values("model_index")
    combined_df = pd.concat([oof_df, ext_df], ignore_index=True)

    confirmatory_indices = parse_confirmatory_indices(args.confirmatory_indices)
    combined_df = add_confirmatory_auc_fdr(
        combined_df,
        "bootstrap_p_two_sided",
        confirmatory_indices=confirmatory_indices,
    )
    oof_df = combined_df[combined_df["dataset"] == "oof"].copy()
    ext_df = combined_df[combined_df["dataset"] == "external"].copy()

    fdr_summary_df = build_confirmatory_auc_fdr_summary(
        combined_df,
        "bootstrap_p_two_sided",
        confirmatory_indices=confirmatory_indices,
        delta_col="delta_auc",
        metric_label="auc",
    )

    col_order = [
        "model_index",
        "model_name",
        "dataset",
        "n_samples",
        "n_pos_uc",
        "auc_western",
        "bootstrap_auc_western_ci_low",
        "bootstrap_auc_western_ci_high",
        "auc_tcm",
        "bootstrap_auc_tcm_ci_low",
        "bootstrap_auc_tcm_ci_high",
        "delta_auc",
        "bootstrap_delta_ci_low",
        "bootstrap_delta_ci_high",
        "bootstrap_p_two_sided",
        "auc_p_raw",
        "auc_p_fdr_bh",
        "bootstrap_n_valid",
        "delong_z",
        "delong_p_two_sided",
        "western_run_dir",
        "tcm_run_dir",
    ]
    for df in (oof_df, ext_df, combined_df):
        for c in col_order:
            if c not in df.columns:
                df[c] = np.nan
        df.sort_values(["model_index"], inplace=True)

    oof_path = out_dir / "stage2_oof_tcm_vs_western.csv"
    ext_path = out_dir / "stage2_external_tcm_vs_western.csv"
    combined_path = out_dir / "stage2_combined_tcm_vs_western.csv"
    fdr_path = out_dir / "stage2_fdr_confirmatory_auc_ml1-6.csv"
    oof_df[col_order].to_csv(oof_path, index=False, encoding="utf-8-sig")
    ext_df[col_order].to_csv(ext_path, index=False, encoding="utf-8-sig")
    combined_df[col_order].to_csv(combined_path, index=False, encoding="utf-8-sig")
    fdr_summary_df.to_csv(fdr_path, index=False, encoding="utf-8-sig")

    manifest_path = out_dir / "run_manifest.json"
    manifest["confirmatory_fdr"] = {
        "method": "benjamini_hochberg",
        "metric": "auc",
        "model_indices": confirmatory_indices,
        "families": [
            f"ml_{confirmatory_indices[0]}-{confirmatory_indices[-1]}_auc_oof",
            f"ml_{confirmatory_indices[0]}-{confirmatory_indices[-1]}_auc_external",
        ],
        "note": "macro_f1 / delong / models 7-10 not FDR-adjusted",
        "summary_csv": str(fdr_path.resolve()),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    readme = out_dir / "README_comparison.txt"
    readme.write_text(
        "\n".join(
            [
                "Stage2 binary: TCM-integrated vs Western baseline",
                f"Generated: {ts}",
                "",
                "delta_auc = AUC_TCM - AUC_Western (positive => TCM better)",
                f"Bootstrap: {args.bootstrap_iter} paired resamples, seed={args.bootstrap_seed}",
                "DeLong: supplementary paired AUC test (two-sided p, not FDR-adjusted)",
                f"Confirmatory FDR (BH): ML models {confirmatory_indices}, AUC only; "
                "auc_p_raw + auc_p_fdr_bh in outputs; separate families for oof vs external.",
                "",
                "Files:",
                f"  {oof_path.name}  - internal OOF comparison (10 models)",
                f"  {ext_path.name}  - external validation comparison",
                f"  {combined_path.name} - stacked long format",
                f"  {fdr_path.name} - confirmatory AUC FDR summary (models 1-6)",
                f"  {manifest_path.name} - run directory mapping",
            ]
        ),
        encoding="utf-8",
    )

    def _print_table(df: pd.DataFrame, title: str) -> None:
        print(f"\n=== {title} ===")
        show = df[
            [
                "model_index",
                "model_name",
                "n_samples",
                "auc_western",
                "auc_tcm",
                "delta_auc",
                "bootstrap_delta_ci_low",
                "bootstrap_delta_ci_high",
                "bootstrap_p_two_sided",
                "auc_p_fdr_bh",
                "delong_p_two_sided",
            ]
        ].copy()
        for c in ["auc_western", "auc_tcm", "delta_auc", "bootstrap_delta_ci_low", "bootstrap_delta_ci_high"]:
            show[c] = show[c].map(lambda x: f"{x:.4f}" if np.isfinite(x) else "nan")
        for c in ["bootstrap_p_two_sided", "auc_p_fdr_bh", "delong_p_two_sided"]:
            show[c] = show[c].map(lambda x: f"{x:.4g}" if np.isfinite(x) else "nan")
        print(show.to_string(index=False))

    print(">>> Stage2 中医融合 vs 西医基础 成对比较完成")
    print(f"    输出目录: {out_dir}")
    _print_table(oof_df, "内部 OOF（Bootstrap ΔAUC + DeLong）")
    _print_table(ext_df, "外部验证（Bootstrap ΔAUC + DeLong）")


if __name__ == "__main__":
    main()
