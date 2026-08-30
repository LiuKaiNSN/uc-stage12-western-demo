# -*- coding: utf-8 -*-
"""
Figure 2: external-validation model comparison — heatmap + AUC forest plot.

Output: F:\\KeTi\\Project\\Figure\\Stage12_TCM\\figure2\\
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from _figure_common import bootstrap_auc_ci, detect_model_dirs, load_external_predictions, metrics_table_from_reports, model_name
from _pub_plot_style import apply_pub_style, save_pub_figure, style_axes

# Light blue sequential (YlGnBu-style); vmin=0.5 keeps cells pale for readable labels.
_HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "metrics_light_blue",
    ["#ffffff", "#f7fbff", "#e9f2fa", "#d4e8f5", "#b9d9eb", "#8fc1de", "#6baed6", "#4292c6"],
    N=256,
)
_HEATMAP_VMIN = 0.5
_HEATMAP_VMAX = 1.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TCM Figure 2: heatmap + forest (external validation)")
    p.add_argument("--stage1-root", type=str, default=r"F:\KeTi\Project\TCM\output\Stage1")
    p.add_argument("--stage2-root", type=str, default=r"F:\KeTi\Project\TCM\output\Stage2")
    p.add_argument("--out-dir", type=str, default=r"F:\KeTi\Project\Figure\Stage12_TCM\figure2")
    p.add_argument("--bootstrap-iter", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _auc_forest_df(stage_root: Path, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for idx, run_dir in sorted(detect_model_dirs(stage_root).items()):
        df = load_external_predictions(run_dir)
        y = df["y_true"].astype(int).to_numpy()
        s = df["prob_1"].astype(float).to_numpy()
        auc, lo, hi = bootstrap_auc_ci(y, s, n_boot=n_boot, seed=seed + idx)
        rows.append({"model": model_name(idx), "auc": auc, "auc_lo": lo, "auc_hi": hi, "model_index": idx})
    return pd.DataFrame(rows)


def plot_heatmap(df: pd.DataFrame, stage_tag: str, out_path: Path) -> None:
    apply_pub_style()
    metrics = ["auc", "auprc", "acc", "f1", "sensitivity", "specificity"]
    labels = ["AUC", "AUPRC", "Accuracy", "F1", "Sensitivity", "Specificity"]
    mat = df.set_index("model")[metrics].to_numpy(dtype=float)
    models = df["model"].tolist()

    fig, ax = plt.subplots(figsize=(8, max(5, 0.45 * len(models))))
    im = ax.imshow(
        mat,
        aspect="auto",
        cmap=_HEATMAP_CMAP,
        vmin=_HEATMAP_VMIN,
        vmax=_HEATMAP_VMAX,
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    span = _HEATMAP_VMAX - _HEATMAP_VMIN
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isfinite(val):
                norm_pos = (float(val) - _HEATMAP_VMIN) / span
                txt_color = "#ffffff" if norm_pos > 0.78 else "#222222"
                ax.text(
                    j,
                    i,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=txt_color,
                )
    ax.set_title(f"External validation metrics ({stage_tag})")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Score")
    style_axes(ax)
    fig.tight_layout()
    save_pub_figure(fig, out_path)


def plot_forest(df: pd.DataFrame, stage_tag: str, out_path: Path) -> None:
    apply_pub_style()
    df = df.sort_values("auc", ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.42 * len(df))))
    for i, row in df.iterrows():
        ax.plot([row["auc_lo"], row["auc_hi"]], [y_pos[i], y_pos[i]], color="#2166AC", linewidth=2)
        ax.scatter(row["auc"], y_pos[i], color="#2166AC", s=40, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["model"])
    ax.set_xlabel("AUC (95% CI, bootstrap)")
    ax.set_title(f"External validation AUC ({stage_tag})")
    ax.set_xlim(0.5, 1.0)
    ax.axvline(df["auc"].max(), color="#CCCCCC", linestyle=":", linewidth=1)
    style_axes(ax)
    fig.tight_layout()
    save_pub_figure(fig, out_path)


def run_stage(stage_root: Path, stage_tag: str, out_dir: Path, n_boot: int, seed: int) -> None:
    metrics_df = metrics_table_from_reports(stage_root)
    metrics_df.to_csv(out_dir / f"{stage_tag}_metrics_external.csv", index=False, encoding="utf-8-sig")
    plot_heatmap(metrics_df, stage_tag, out_dir / f"figure2_{stage_tag}_heatmap.jpg")
    forest_df = _auc_forest_df(stage_root, n_boot, seed)
    forest_df.to_csv(out_dir / f"{stage_tag}_auc_forest_data.csv", index=False, encoding="utf-8-sig")
    plot_forest(forest_df, stage_tag, out_dir / f"figure2_{stage_tag}_forest_auc.jpg")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_stage(Path(args.stage1_root), "stage1", out_dir, args.bootstrap_iter, args.seed)
    run_stage(Path(args.stage2_root), "stage2", out_dir, args.bootstrap_iter, args.seed + 100)
    print(f"[OK] Figure 2 -> {out_dir}")


if __name__ == "__main__":
    main()
