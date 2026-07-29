# -*- coding: utf-8 -*-
"""Figure 6: simple W2 workflow panels from 03_4_best_cross workflow JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _pub_plot_style import apply_pub_style, COLOR_NEG, COLOR_POS, COLOR_WESTERN, save_pub_figure, style_axes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Figure 6: W2 workflow summary panels")
    p.add_argument(
        "--workflow-json",
        type=str,
        default=r"F:\KeTi\Project\outputs\workflow_batch\western\03_4_best_cross\workflow_stage12.json",
    )
    p.add_argument("--out-dir", type=str, default=r"F:\KeTi\Project\Figure\Stage12\figure6")
    p.add_argument("--s1-model", type=str, default="LightGBM (run 3)")
    p.add_argument("--s2-model", type=str, default="CatBoost (run 4)")
    return p.parse_args()


def _draw_panel(ax, panel: dict, model_name: str, stage_title: str) -> None:
    m = panel["metrics_at_threshold"]
    tp, tn, fp, fn = m["tp"], m["tn"], m["fp"], m["fn"]
    referred = panel["referred_or_uc_path_n"]
    not_ref = panel["not_referred_n"]
    pt = panel["threshold"]
    n = panel["n"]

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(5, 9.2, stage_title, ha="center", fontsize=13, fontweight="bold")
    ax.text(5, 8.5, f"Model: {model_name}  |  n = {n}  |  threshold = {pt:.2f}", ha="center", fontsize=9)

    # flow boxes
    ax.add_patch(plt.Rectangle((0.8, 5.8), 3.6, 1.6, fill=True, facecolor="#E8F4FC", edgecolor=COLOR_WESTERN, linewidth=1.5))
    ax.text(2.6, 6.6, f"Positive path\n{referred}", ha="center", va="center", fontsize=10)

    ax.add_patch(plt.Rectangle((5.8, 5.8), 3.6, 1.6, fill=True, facecolor="#F5F5F5", edgecolor="#888888", linewidth=1.5))
    ax.text(7.6, 6.6, f"Negative path\n{not_ref}", ha="center", va="center", fontsize=10)

    ax.text(5, 5.2, panel.get("positive_class", ""), ha="center", fontsize=8, color="#444444")

    labels = ["TP", "FP", "TN", "FN"]
    vals = [tp, fp, tn, fn]
    colors = [COLOR_POS, "#F4A582", COLOR_NEG, "#D6604D"]
    xs = [1.5, 3.5, 5.5, 7.5]
    for x, lab, val, col in zip(xs, labels, vals, colors):
        ax.add_patch(plt.Rectangle((x - 0.7, 2.0), 1.4, 1.4, facecolor=col, alpha=0.35, edgecolor="#333333"))
        ax.text(x, 2.9, lab, ha="center", fontsize=10, fontweight="bold")
        ax.text(x, 2.35, str(int(val)), ha="center", fontsize=11)

    sens = m.get("sensitivity", 0)
    spec = m.get("specificity", 0)
    ax.text(
        5,
        0.8,
        f"Sensitivity = {sens:.3f}   Specificity = {spec:.3f}   PPV = {m.get('ppv', 0):.3f}   NPV = {m.get('npv', 0):.3f}",
        ha="center",
        fontsize=9,
    )
    ax.text(5, 0.2, "W2: stages on respective external cohorts (illustrative, not probability cascade)", ha="center", fontsize=7, color="#666666")


def main() -> None:
    args = parse_args()
    data = json.loads(Path(args.workflow_json).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    apply_pub_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    _draw_panel(
        axes[0],
        data["panel_stage1"],
        args.s1_model,
        "Stage 1 — Organic vs functional triage",
    )
    _draw_panel(
        axes[1],
        data["panel_stage2"],
        args.s2_model,
        "Stage 2 — UC vs non-UC discrimination",
    )
    fig.suptitle("Sequential screening workflow (Western, best cross 1×4)", fontsize=14, y=1.02)
    fig.tight_layout()
    save_pub_figure(fig, out_dir / "workflow_w2_panels.jpg")
    print(f"[OK] Figure 6 -> {out_dir / 'workflow_w2_panels.jpg'}")


if __name__ == "__main__":
    main()
