# -*- coding: utf-8 -*-
"""
Figure 3: external validation — 2×2 panel (ROC + DCA, Stage1 & Stage2).

Main output: figure3_external_validation_roc_dca.jpg
Optional supplement plots + subgroup CSV (--write-supplement) for tables / SI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, roc_curve

from _figure_common import bootstrap_auc_ci, load_external_predictions, subgroup_auc_rows
from _pub_plot_style import (
    COLOR_TREAT_ALL,
    COLOR_TREAT_NONE,
    COLOR_WESTERN,
    apply_pub_style,
    save_pub_figure,
    style_axes,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TCM Figure 3: external validation ROC+DCA panel")
    p.add_argument(
        "--s1-run-dir",
        type=str,
        default=r"F:\KeTi\Project\TCM\output\Stage1\6_run_20260708_052605",
    )
    p.add_argument(
        "--s2-run-dir",
        type=str,
        default=r"F:\KeTi\Project\TCM\output\Stage2\6_run_20260707_111716",
    )
    p.add_argument(
        "--workflow-json",
        type=str,
        default=r"F:\KeTi\Project\TCM\output\workflow_batch\tcm\06_6_best_cross\workflow_stage12.json",
    )
    p.add_argument(
        "--s1-dca-dir",
        type=str,
        default=r"F:\KeTi\Project\TCM\output\Stage1\dca_batch\6_run_20260708_052605\overall",
    )
    p.add_argument(
        "--s2-dca-dir",
        type=str,
        default=r"F:\KeTi\Project\TCM\output\Stage2\dca_batch\6_run_20260707_111716\overall",
    )
    p.add_argument("--s1-model-label", type=str, default="RF")
    p.add_argument("--s2-model-label", type=str, default="CatBoost")
    p.add_argument("--out-dir", type=str, default=r"F:\KeTi\Project\Figure\Stage12_TCM\figure3")
    p.add_argument("--bootstrap-iter", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--write-supplement",
        action="store_true",
        help="Also write calibration / confusion / subgroup forest to supplement/",
    )
    return p.parse_args()


def _thresholds_from_workflow(path: Path) -> Tuple[float, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data["pt1"]), float(data["pt2"])


def _stage_labels(stage: str) -> Tuple[str, str]:
    if stage == "stage1":
        return "Functional (IE+IBS)", "Organic (UC+CD+IC+CRC)"
    return "Non-UC (CD+IC+CRC)", "UC"


def _read_operating_pt(dca_dir: Path) -> Optional[float]:
    op = dca_dir / "operating_point.csv"
    if not op.is_file():
        return None
    df = pd.read_csv(op)
    if "threshold" not in df.columns or df.empty:
        return None
    return float(df.iloc[0]["threshold"])


def _draw_roc_ax(
    ax: plt.Axes,
    df: pd.DataFrame,
    stage: str,
    n_boot: int,
    seed: int,
) -> None:
    y = df["y_true"].astype(int).to_numpy()
    s = df["prob_1"].astype(float).to_numpy()
    auc, lo, hi = bootstrap_auc_ci(y, s, n_boot=n_boot, seed=seed)
    fpr, tpr, _ = roc_curve(y, s)
    ax.plot(fpr, tpr, color="#2166AC", linewidth=2.2, label=f"AUC = {auc:.3f} ({lo:.3f}–{hi:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#969696", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    _, pos = _stage_labels(stage)
    ax.set_title(f"{stage.upper()} ROC — positive: {pos}", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    style_axes(ax)


def _draw_dca_ax(
    ax: plt.Axes,
    dca_dir: Path,
    model_label: str,
    stage: str,
    show_legend: bool = False,
) -> None:
    curve_path = dca_dir / "dca_curve.csv"
    if not curve_path.is_file():
        raise FileNotFoundError(curve_path)
    dca = pd.read_csv(curve_path)
    op_pt = _read_operating_pt(dca_dir)

    ax.plot(dca["threshold"], dca["model"], label=model_label, color=COLOR_WESTERN, linewidth=2.2)
    ax.plot(
        dca["threshold"],
        dca["treat_all"],
        label="Treat all",
        color=COLOR_TREAT_ALL,
        linestyle="--",
        linewidth=1.4,
    )
    ax.plot(
        dca["threshold"],
        dca["treat_none"],
        label="Treat none",
        color=COLOR_TREAT_NONE,
        linestyle=":",
        linewidth=1.4,
    )
    if op_pt is not None:
        ax.axvline(
            op_pt,
            color="#555555",
            linestyle="-.",
            linewidth=1.1,
            label=f"Operating pt = {op_pt:.2f}",
        )
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title(f"{stage.upper()} decision curve analysis", fontsize=10)
    if show_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=True, fontsize=8)
    ax.set_xlim(float(dca["threshold"].min()), float(dca["threshold"].max()))
    style_axes(ax)


def plot_combined_roc_dca_panel(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s1_dca_dir: Path,
    s2_dca_dir: Path,
    s1_model_label: str,
    s2_model_label: str,
    out_path: Path,
    n_boot: int,
    seed: int,
) -> None:
    apply_pub_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    _draw_roc_ax(axes[0, 0], s1_df, "stage1", n_boot, seed)
    _draw_roc_ax(axes[0, 1], s2_df, "stage2", n_boot, seed + 50)
    _draw_dca_ax(axes[1, 0], s1_dca_dir, s1_model_label, "stage1")
    _draw_dca_ax(axes[1, 1], s2_dca_dir, s2_model_label, "stage2")
    legend_handles = [
        Line2D([0], [0], color=COLOR_WESTERN, linewidth=2.2, label=f"Stage1 ({s1_model_label})"),
        Line2D([0], [0], color=COLOR_WESTERN, linewidth=2.2, label=f"Stage2 ({s2_model_label})"),
        Line2D([0], [0], color=COLOR_TREAT_ALL, linewidth=1.4, linestyle="--", label="Treat all"),
        Line2D([0], [0], color=COLOR_TREAT_NONE, linewidth=1.4, linestyle=":", label="Treat none"),
        Line2D([0], [0], color="#555555", linewidth=1.1, linestyle="-.", label="Operating point"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=True,
        fontsize=8,
    )
    fig.suptitle("External validation: discrimination and clinical utility", fontsize=13, y=0.98)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.14, hspace=0.32, wspace=0.28)
    save_pub_figure(fig, out_path)


def _export_subgroup_csv(
    df: pd.DataFrame,
    stage: str,
    out_path: Path,
    n_boot: int,
    seed: int,
) -> None:
    parts: List[pd.DataFrame] = []
    if "Source" in df.columns:
        parts.append(subgroup_auc_rows(df, "Source", n_boot=n_boot, seed=seed))
        parts[-1]["subgroup"] = "Source: " + parts[-1]["subgroup"]
    if "Agerange" in df.columns:
        parts.append(subgroup_auc_rows(df, "Agerange", n_boot=n_boot, seed=seed + 7))
        parts[-1]["subgroup"] = "Age: " + parts[-1]["subgroup"].astype(str)
    if not parts:
        return
    sub = pd.concat(parts, ignore_index=True)
    y_all = df["y_true"].astype(int).to_numpy()
    s_all = df["prob_1"].astype(float).to_numpy()
    overall, lo, hi = bootstrap_auc_ci(y_all, s_all, n_boot=n_boot, seed=seed + 99)
    overall_row = pd.DataFrame(
        [{"subgroup": "Overall", "n": len(df), "auc": overall, "auc_lo": lo, "auc_hi": hi}]
    )
    out = pd.concat([overall_row, sub], ignore_index=True)
    out.insert(0, "stage", stage)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")


def plot_calibration(df: pd.DataFrame, stage: str, out_path: Path) -> None:
    apply_pub_style()
    y = df["y_true"].astype(int).to_numpy()
    s = df["prob_1"].astype(float).to_numpy()
    prob_true, prob_pred = calibration_curve(y, s, n_bins=8, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#969696", label="Perfect calibration")
    ax.plot(prob_pred, prob_true, marker="o", color="#2166AC", linewidth=2, label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction positive")
    ax.set_title(f"Calibration plot — external validation ({stage})")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    style_axes(ax)
    save_pub_figure(fig, out_path)


def plot_confusion(df: pd.DataFrame, threshold: float, stage: str, out_path: Path) -> None:
    apply_pub_style()
    y = df["y_true"].astype(int).to_numpy()
    s = df["prob_1"].astype(float).to_numpy()
    pred = (s >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    neg, pos = _stage_labels(stage)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([f"Pred {neg}", f"Pred {pos}"], rotation=20, ha="right")
    ax.set_yticklabels([f"True {neg}", f"True {pos}"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="#111", fontsize=14)
    ax.set_title(f"Confusion matrix ({stage}, threshold = {threshold:.2f})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_pub_figure(fig, out_path)


def plot_subgroup_forest(
    df: pd.DataFrame,
    stage: str,
    out_path: Path,
    n_boot: int,
    seed: int,
) -> None:
    apply_pub_style()
    parts: List[pd.DataFrame] = []
    if "Source" in df.columns:
        parts.append(subgroup_auc_rows(df, "Source", n_boot=n_boot, seed=seed))
        parts[-1]["subgroup"] = "Source: " + parts[-1]["subgroup"]
    if "Agerange" in df.columns:
        parts.append(subgroup_auc_rows(df, "Agerange", n_boot=n_boot, seed=seed + 7))
        parts[-1]["subgroup"] = "Age: " + parts[-1]["subgroup"].astype(str)
    if not parts:
        return
    sub = pd.concat(parts, ignore_index=True).sort_values("auc", ascending=True).reset_index(drop=True)
    y_all = df["y_true"].astype(int).to_numpy()
    s_all = df["prob_1"].astype(float).to_numpy()
    overall, _, _ = bootstrap_auc_ci(y_all, s_all, n_boot=n_boot, seed=seed + 99)

    y_pos = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.38 * len(sub))))
    for i, row in sub.iterrows():
        ax.plot([row["auc_lo"], row["auc_hi"]], [y_pos[i], y_pos[i]], color="#2166AC", linewidth=2)
        ax.scatter(row["auc"], y_pos[i], color="#2166AC", s=36)
    ax.axvline(overall, color="#C44E52", linestyle="--", linewidth=1.2, label=f"Overall AUC = {overall:.3f}")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{row['subgroup']} (n={row['n']})" for _, row in sub.iterrows()], fontsize=8)
    ax.set_xlabel("AUC (95% CI)")
    ax.set_title(f"Subgroup AUC — external validation ({stage})")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0.5, 1.0)
    style_axes(ax)
    fig.tight_layout()
    save_pub_figure(fig, out_path)


def _write_supplement(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    pt1: float,
    pt2: float,
    supp_dir: Path,
    n_boot: int,
    seed: int,
) -> None:
    supp_dir.mkdir(parents=True, exist_ok=True)
    plot_calibration(s1_df, "stage1", supp_dir / "stage1_calibration.jpg")
    plot_calibration(s2_df, "stage2", supp_dir / "stage2_calibration.jpg")
    plot_confusion(s1_df, pt1, "stage1", supp_dir / "stage1_confusion_matrix.jpg")
    plot_confusion(s2_df, pt2, "stage2", supp_dir / "stage2_confusion_matrix.jpg")
    plot_subgroup_forest(s1_df, "stage1", supp_dir / "stage1_subgroup_forest_auc.jpg", n_boot, seed)
    plot_subgroup_forest(s2_df, "stage2", supp_dir / "stage2_subgroup_forest_auc.jpg", n_boot, seed + 50)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    data_dir = out_dir / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    pt1, pt2 = _thresholds_from_workflow(Path(args.workflow_json))
    s1_df = load_external_predictions(Path(args.s1_run_dir))
    s2_df = load_external_predictions(Path(args.s2_run_dir))
    s1_dca = Path(args.s1_dca_dir)
    s2_dca = Path(args.s2_dca_dir)

    plot_combined_roc_dca_panel(
        s1_df,
        s2_df,
        s1_dca,
        s2_dca,
        args.s1_model_label,
        args.s2_model_label,
        out_dir / "figure3_external_validation_roc_dca.jpg",
        args.bootstrap_iter,
        args.seed,
    )

    _export_subgroup_csv(s1_df, "stage1", data_dir / "stage1_subgroup_auc.csv", args.bootstrap_iter, args.seed)
    _export_subgroup_csv(
        s2_df, "stage2", data_dir / "stage2_subgroup_auc.csv", args.bootstrap_iter, args.seed + 50
    )

    if args.write_supplement:
        _write_supplement(s1_df, s2_df, pt1, pt2, out_dir / "supplement", args.bootstrap_iter, args.seed)

    print(f"[OK] Figure 3 -> {out_dir / 'figure3_external_validation_roc_dca.jpg'}")
    print(f"     Subgroup CSV -> {data_dir}")


if __name__ == "__main__":
    main()
