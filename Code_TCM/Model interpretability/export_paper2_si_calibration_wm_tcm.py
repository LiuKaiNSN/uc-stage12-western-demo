# -*- coding: utf-8 -*-
"""
Paper2 SI: external calibration — best models, 2×2 panel.

  Left column: Baseline Stage1 / Stage2
  Right column: TCM-integrated Stage1 / Stage2
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

from _figure_common import load_external_predictions
from _pub_plot_style import COLOR_TCM, COLOR_WESTERN, apply_pub_style, save_pub_figure, style_axes

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parent
DEFAULT_OUT = PROJECT / "Figure" / "比较" / "For submission only"

WM_S1 = PROJECT / "outputs" / "Stage1" / "3_run_20260707_142952"
WM_S2 = PROJECT / "outputs" / "Stage2" / "4_run_20260707_030540"
TCM_S1 = PROJECT / "TCM" / "output" / "Stage1" / "6_run_20260708_052605"
TCM_S2 = PROJECT / "TCM" / "output" / "Stage2" / "6_run_20260707_111716"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper2 SI calibration WM|TCM best models")
    p.add_argument("--wm-s1-run", type=str, default=str(WM_S1))
    p.add_argument("--wm-s2-run", type=str, default=str(WM_S2))
    p.add_argument("--tcm-s1-run", type=str, default=str(TCM_S1))
    p.add_argument("--tcm-s2-run", type=str, default=str(TCM_S2))
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    return p.parse_args()


def _draw_cal(ax, df, *, color: str, title: str) -> None:
    y = df["y_true"].astype(int).to_numpy()
    s = df["prob_1"].astype(float).to_numpy()
    prob_true, prob_pred = calibration_curve(y, s, n_bins=8, strategy="quantile")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#969696", linewidth=1, label="Perfect")
    ax.plot(prob_pred, prob_true, marker="o", color=color, linewidth=2, label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction positive")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
    style_axes(ax)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_pub_style()

    cells = [
        (0, 0, Path(args.wm_s1_run), COLOR_WESTERN, "Baseline Stage1 (LightGBM)"),
        (1, 0, Path(args.wm_s2_run), COLOR_WESTERN, "Baseline Stage2 (CatBoost)"),
        (0, 1, Path(args.tcm_s1_run), COLOR_TCM, "TCM-integrated Stage1 (RF)"),
        (1, 1, Path(args.tcm_s2_run), COLOR_TCM, "TCM-integrated Stage2 (RF)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.0))
    fig.subplots_adjust(hspace=0.35, wspace=0.28, left=0.08, right=0.98, top=0.90, bottom=0.08)
    for row, col, run_dir, color, title in cells:
        _draw_cal(axes[row, col], load_external_predictions(run_dir), color=color, title=title)

    fig.suptitle(
        "External validation calibration — best models (Baseline vs TCM-integrated)",
        y=0.97,
        fontsize=12,
    )
    out_jpg = out_dir / "SI_calibration_baseline_tcm_twostage_best.jpg"
    save_pub_figure(fig, out_jpg, dpi=300)

    manifest = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "figure": str(out_jpg),
        "note": "Best-model strategy; SI only",
    }
    (out_dir / "manifest_si_calibration.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK] SI calibration -> {out_jpg}")


if __name__ == "__main__":
    main()
