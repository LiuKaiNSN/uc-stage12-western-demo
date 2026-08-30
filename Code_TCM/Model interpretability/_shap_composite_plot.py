# -*- coding: utf-8 -*-
"""
SHAP composite (nested, separate axes):
  upper — upper-triangular Pearson r (top feature labels only)
  lower — mean |SHAP| bars (own left y-axis + bottom labels)
  columns aligned; upper grid only on j>i cells; full-height colorbar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

from _pub_plot_style import apply_pub_style, style_axes

_CORR_CMAP = LinearSegmentedColormap.from_list(
    "corr_light_prgn",
    ["#7b3294", "#c2a5cf", "#f7f7f7", "#a6dba0", "#008837"],
    N=256,
)

_GRID_COLOR = "#cccccc"
_GRID_LW = 0.65


def _transparent_axes(ax: plt.Axes) -> None:
    """Let overlapping axes show through (no opaque white panel)."""
    ax.set_facecolor("none")
    ax.patch.set_alpha(0.0)


def _align_mat_with_bar(ax_mat: plt.Axes, ax_bar: plt.Axes) -> None:
    """Square data cells; keep matrix top/footprint, align columns with bar axis."""
    ax_mat.set_aspect("equal", adjustable="box", anchor="N")
    fig = ax_mat.figure
    fig.canvas.draw()
    p_bar = ax_bar.get_position()
    p_mat = ax_mat.get_position()
    ax_mat.set_position([p_bar.x0, p_mat.y0, p_bar.width, p_mat.height])
    fig.canvas.draw()


def _circle_size_for_square_cells(ax_mat: plt.Axes) -> float:
    o0 = ax_mat.transData.transform((0.0, 0.0))
    ox = ax_mat.transData.transform((1.0, 0.0))
    oy = ax_mat.transData.transform((0.0, 1.0))
    cell_px = min(float(np.hypot(*(ox - o0))), float(np.hypot(*(oy - o0))))
    return float(np.clip((0.68 * cell_px) ** 2, 320.0, 2200.0))


def _align_xlim(ax_mat: plt.Axes, ax_bar: plt.Axes, n: int) -> None:
    x0, x1 = -0.5, n - 0.5
    ax_mat.set_xlim(x0, x1)
    ax_bar.set_xlim(x0, x1)


def _draw_upper_triangle_grid(ax: plt.Axes, n: int) -> None:
    """Grid lines only for upper-triangle cells (j > i); skip empty lower-left."""
    for i in range(n):
        for j in range(i + 1, n):
            ax.add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5),
                    1.0,
                    1.0,
                    facecolor="none",
                    edgecolor=_GRID_COLOR,
                    linewidth=_GRID_LW,
                    zorder=1,
                )
            )


def plot_shap_importance_correlation(
    importance: pd.DataFrame,
    corr: np.ndarray,
    out_path: Path,
    title: str,
    feature_label_map: Dict[str, str],
    bar_ymax: float | None = None,
) -> None:
    apply_pub_style()
    if "feature" not in importance.columns:
        raise ValueError("importance DataFrame must contain a 'feature' column")

    features_orig = importance["feature"].astype(str).tolist()
    imp = importance.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    features_sorted = imp["feature"].astype(str).tolist()
    values = imp["mean_abs_shap"].to_numpy(dtype=float)
    n = len(values)
    if corr.shape != (len(features_orig), len(features_orig)):
        raise ValueError(f"corr shape {corr.shape} != ({len(features_orig)}, {len(features_orig)})")

    reorder_idx = [features_orig.index(f) for f in features_sorted]
    corr = corr[np.ix_(reorder_idx, reorder_idx)]
    display_labels = [feature_label_map.get(f, f) for f in features_sorted]

    side = max(10.7, n * 0.79)
    # Mild extra height only (~1/3 of prior +1.35 overshoot).
    fig_h = side + 0.45
    fig = plt.figure(figsize=(side, fig_h))

    left = 0.11
    width = 0.66
    cbar_gap = 0.018
    cbar_w = 0.022

    # Keep ~1/3 of the previous margin expansion (avoid overlap without large gaps).
    content_bottom = 0.133
    content_top = 0.82
    content_height = content_top - content_bottom

    # Upper matrix: square cells at full bar width; layout independent of bar height.
    xlim_span = float(n)
    ylim_span = float(n) - 0.5  # y in [-0.5, n-1]
    mat_height = width * (ylim_span / xlim_span)
    mat_bottom = content_top - mat_height

    bar_bottom = content_bottom
    bar_height = content_height * 1.10

    cbar_ax = fig.add_axes([left + width + cbar_gap, content_bottom, cbar_w, content_height])
    ax_mat = fig.add_axes([left, mat_bottom, width, mat_height])
    ax_bar = fig.add_axes([left, bar_bottom, width, bar_height])
    fig.patch.set_facecolor("white")
    _transparent_axes(ax_mat)
    _transparent_axes(ax_bar)
    ax_mat.set_zorder(1)
    ax_bar.set_zorder(2)

    vmax_r = float(np.nanmax(np.abs(corr))) if np.isfinite(corr).any() else 0.1
    vmax_r = max(vmax_r, 0.05)
    norm_r = Normalize(vmin=-vmax_r, vmax=vmax_r)
    cmap_r = _CORR_CMAP

    vmax_imp = float(np.max(values)) if values.size else 1.0
    vmax_imp = max(vmax_imp, 1e-9)
    imp_min = float(np.min(values))
    norm_imp = Normalize(vmin=imp_min, vmax=vmax_imp)

    _align_xlim(ax_mat, ax_bar, n)

    # Drop the empty last matrix row (upper triangle has no cells on row n-1).
    y_top = -0.5
    y_bottom = float(n) - 1.0 if n > 1 else 0.5
    ax_mat.set_ylim(y_bottom, y_top)
    ax_mat.set_clip_on(False)

    _align_mat_with_bar(ax_mat, ax_bar)
    circle_size = _circle_size_for_square_cells(ax_mat)
    label_fs = max(6, min(8, int(0.24 * np.sqrt(circle_size))))
    for i in range(n):
        for j in range(i + 1, n):
            r = float(corr[i, j])
            color = cmap_r(norm_r(r))
            ax_mat.scatter(
                j,
                i,
                s=circle_size,
                c=[color],
                edgecolors="#888888",
                linewidths=0.45,
                zorder=3,
            )
            ax_mat.text(
                j, i, f"{r:.2f}", ha="center", va="center", fontsize=label_fs,
                color="#222222", zorder=4,
            )

    _draw_upper_triangle_grid(ax_mat, n)

    ax_mat.set_xticks(range(n))
    ax_mat.set_xticklabels(display_labels, rotation=45, ha="left", fontsize=7.5)
    ax_mat.xaxis.tick_top()
    ax_mat.xaxis.set_label_position("top")
    # Small pad: labels stay close to matrix and clear of the title band above.
    ax_mat.tick_params(axis="x", bottom=False, labelbottom=False, top=True, labeltop=True, pad=2)
    ax_mat.set_yticks([])
    ax_mat.tick_params(axis="y", left=False, labelleft=False, right=False, labelright=False)
    for spine in ("left", "bottom", "right"):
        ax_mat.spines[spine].set_visible(False)
    style_axes(ax_mat)
    _transparent_axes(ax_mat)

    sm = plt.cm.ScalarMappable(cmap=cmap_r, norm=norm_r)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Pearson r", fontsize=9)

    x = np.arange(n)
    bar_colors = [
        cmap_r(norm_r(vmax_r * (1.0 - 2.0 * float(norm_imp(v)))))
        for v in values
    ]
    bars = ax_bar.bar(
        x,
        values,
        width=0.72,
        color=bar_colors,
        edgecolor="#666666",
        linewidth=0.45,
        align="center",
    )
    for rect, val in zip(bars, values):
        ax_bar.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + vmax_imp * 0.03,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#222222",
        )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=7.5)
    ax_bar.set_ylabel("Mean |SHAP|")
    # Low-SHAP panels (e.g. TCM Stage1 RF): fixed y-axis cap instead of default 1.0 floor.
    _LOW_SHAP_BAR_YMAX = 0.14
    if bar_ymax is not None:
        y_max = float(bar_ymax)
    elif vmax_imp < _LOW_SHAP_BAR_YMAX:
        y_max = _LOW_SHAP_BAR_YMAX
    else:
        y_max = max(1.0, vmax_imp * 1.22)
    ax_bar.set_ylim(0, y_max)
    ax_bar.tick_params(axis="x", top=False, labeltop=False, pad=2)
    ax_bar.tick_params(axis="y", right=False, labelright=False)
    for spine in ("top", "right"):
        ax_bar.spines[spine].set_visible(False)
    style_axes(ax_bar)
    _transparent_axes(ax_bar)

    # Title sits in the upper band of the taller figure, above matrix top labels.
    fig.suptitle(title, fontsize=12, y=0.97)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Do NOT crop to a square Bbox — that was clipping title / bottom labels.
    # pad_inches kept mild (~1/3 of prior 0.45).
    fig.savefig(
        out_path,
        format="jpeg",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.15,
        facecolor="white",
        edgecolor="none",
        pil_kwargs={"quality": 95},
    )
    plt.close(fig)
