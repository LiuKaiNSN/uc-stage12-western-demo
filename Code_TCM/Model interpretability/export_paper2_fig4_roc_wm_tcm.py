# -*- coding: utf-8 -*-
"""
Paper2 Figure 4: external ROC — left column Baseline Model (best), right TCM-integrated (best).

Layout (option B):
  Left:  WM Stage1 ROC + WM Stage2 ROC
  Right: TCM Stage1 ROC + TCM Stage2 ROC
No DCA.

Default best models:
  WM  S1 LightGBM #3, S2 CatBoost #4
  TCM S1 RF #6, S2 RF #6
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from _figure_common import bootstrap_auc_ci, load_external_predictions
from _pub_plot_style import COLOR_TCM, COLOR_WESTERN, apply_pub_style, save_pub_figure, style_axes

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parent
DEFAULT_OUT = PROJECT / "Figure" / "比较" / "For submission only"

WM_S1 = PROJECT / "outputs" / "Stage1" / "3_run_20260707_142952"
WM_S2 = PROJECT / "outputs" / "Stage2" / "4_run_20260707_030540"
TCM_S1 = PROJECT / "TCM" / "output" / "Stage1" / "6_run_20260708_052605"
TCM_S2 = PROJECT / "TCM" / "output" / "Stage2" / "6_run_20260707_111716"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper2 Fig4 WM|TCM two-stage external ROC")
    p.add_argument("--wm-s1-run", type=str, default=str(WM_S1))
    p.add_argument("--wm-s2-run", type=str, default=str(WM_S2))
    p.add_argument("--tcm-s1-run", type=str, default=str(TCM_S1))
    p.add_argument("--tcm-s2-run", type=str, default=str(TCM_S2))
    p.add_argument("--wm-s1-label", type=str, default="Baseline Stage1 (LightGBM)")
    p.add_argument("--wm-s2-label", type=str, default="Baseline Stage2 (CatBoost)")
    p.add_argument("--tcm-s1-label", type=str, default="TCM-integrated Stage1 (RF)")
    p.add_argument("--tcm-s2-label", type=str, default="TCM-integrated Stage2 (RF)")
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--bootstrap-iter", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _stage_pos_label(stage: str) -> str:
    if stage == "stage1":
        return "positive: Organic (UC+CD+IC+CRC)"
    return "positive: UC"


def _draw_roc(
    ax,
    df,
    *,
    color: str,
    model_label: str,
    stage: str,
    n_boot: int,
    seed: int,
) -> None:
    y = df["y_true"].astype(int).to_numpy()
    s = df["prob_1"].astype(float).to_numpy()
    auc, lo, hi = bootstrap_auc_ci(y, s, n_boot=n_boot, seed=seed)
    fpr, tpr, _ = roc_curve(y, s)
    ax.plot(fpr, tpr, color=color, linewidth=2.2, label=f"AUC = {auc:.3f} ({lo:.3f}–{hi:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#969696", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{model_label}\n{_stage_pos_label(stage)}", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    style_axes(ax)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_pub_style()

    packs = [
        (Path(args.wm_s1_run), "stage1", COLOR_WESTERN, args.wm_s1_label, 0),
        (Path(args.wm_s2_run), "stage2", COLOR_WESTERN, args.wm_s2_label, 1),
        (Path(args.tcm_s1_run), "stage1", COLOR_TCM, args.tcm_s1_label, 0),
        (Path(args.tcm_s2_run), "stage2", COLOR_TCM, args.tcm_s2_label, 1),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.2))
    # Leave headroom so suptitle does not collide with column headers.
    fig.subplots_adjust(hspace=0.38, wspace=0.28, left=0.08, right=0.98, top=0.82, bottom=0.08)

    # columns: WM | TCM ; rows: Stage1 | Stage2
    layout = [
        (0, 0, packs[0], args.seed),
        (1, 0, packs[1], args.seed + 50),
        (0, 1, packs[2], args.seed + 100),
        (1, 1, packs[3], args.seed + 150),
    ]
    for row, col, (run_dir, stage, color, label, _), seed in layout:
        df = load_external_predictions(run_dir)
        _draw_roc(
            axes[row, col],
            df,
            color=color,
            model_label=label,
            stage=stage,
            n_boot=args.bootstrap_iter,
            seed=seed,
        )

    # Figure-level column headers (between suptitle and top axes), not axes.annotate.
    fig.text(
        0.29,
        0.90,
        "Baseline (best models)",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=COLOR_WESTERN,
    )
    fig.text(
        0.71,
        0.90,
        "TCM-integrated (best models)",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=COLOR_TCM,
    )
    fig.suptitle(
        "External validation ROC — two-stage best models (Baseline vs TCM-integrated)",
        y=0.98,
        fontsize=12,
    )

    out_jpg = out_dir / "Fig4_external_roc_baseline_tcm_twostage_best.jpg"
    save_pub_figure(fig, out_jpg, dpi=300)

    manifest = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "figure": str(out_jpg),
        "wm_s1": str(Path(args.wm_s1_run).resolve()),
        "wm_s2": str(Path(args.wm_s2_run).resolve()),
        "tcm_s1": str(Path(args.tcm_s1_run).resolve()),
        "tcm_s2": str(Path(args.tcm_s2_run).resolve()),
        "note": "Best-model strategy (not same-algorithm pairing); no DCA",
    }
    (out_dir / "manifest_fig4_roc.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK] Fig4 -> {out_jpg}")


if __name__ == "__main__":
    main()
