# -*- coding: utf-8 -*-
"""Shared helpers for TCM vs Baseline Model comparison figures (statistics + DCA composites)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from _dca_common import (
    STAGE1_DISEASE_SLICES,
    STAGE2_DISEASE_SLICES,
    STAGE3_E3_PAIR_ID,
    disease_slices_for_stage,
    stage_defaults,
)
from _pub_plot_style import (
    COLOR_TCM,
    COLOR_TREAT_ALL,
    COLOR_TREAT_NONE,
    COLOR_WESTERN,
    MODEL_INDEX_MAP,
    apply_pub_style,
    save_pub_figure,
)

LABEL_WESTERN = "Objective Factor Baseline Model"
LABEL_BASELINE = LABEL_WESTERN
LABEL_TCM = "TCM-integrated model"
LABEL_TREAT_ALL = "Treat all"
LABEL_TREAT_NONE = "Treat none"

SLICE_DISPLAY: Dict[str, str] = {
    "uc_vs_ie_ibs_pool": "UC vs IE+IBS",
    "cd_vs_ie_ibs_pool": "CD vs IE+IBS",
    "ic_vs_ie_ibs_pool": "IC vs IE+IBS",
    "crc_vs_ie_ibs_pool": "CRC vs IE+IBS",
    "uc_vs_cd": "UC vs CD",
    "uc_vs_ic": "UC vs IC",
    "uc_vs_crc": "UC vs CRC",
    STAGE3_E3_PAIR_ID: "E3 vs E1+E2",
    "overall": "Overall",
}


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_stage12_overlay_dir(batch_root: Path, model_index: int) -> Path:
    batch_root = Path(batch_root)
    candidates = [
        batch_root / f"{model_index:02d}_run",
        batch_root / f"{model_index}_run",
    ]
    for p in candidates:
        if p.is_dir():
            return p
    raise FileNotFoundError(f"No overlay run folder for index {model_index} under {batch_root}")


def resolve_stage3_overlay_dir(batch_root: Path, model_index: int) -> Path:
    batch_root = Path(batch_root)
    matches = sorted(batch_root.glob(f"overlay_{model_index:02d}_*"))
    if not matches:
        raise FileNotFoundError(f"No Stage3 overlay folder for index {model_index} under {batch_root}")
    return matches[0]


def resolve_overlay_dir(batch_root: Path, stage: str, model_index: int) -> Path:
    if stage.lower() == "stage3":
        return resolve_stage3_overlay_dir(batch_root, model_index)
    return resolve_stage12_overlay_dir(batch_root, model_index)


def load_overlay_curves(run_dir: Path, slice_id: str = "overall") -> Tuple[pd.DataFrame, pd.DataFrame, float, float]:
    base = Path(run_dir)
    if slice_id == "overall":
        sub = base
    else:
        sub = base / "slices" / slice_id
    w_path = sub / "dca_curve_western.csv"
    t_path = sub / "dca_curve_tcm.csv"
    if not w_path.is_file() or not t_path.is_file():
        raise FileNotFoundError(f"Missing DCA curves in {sub}")
    meta_path = sub / "dca_overlay_meta.json"
    op_w, op_t = _operating_points_from_meta(meta_path)
    return pd.read_csv(w_path), pd.read_csv(t_path), op_w, op_t


def _operating_points_from_meta(meta_path: Path) -> Tuple[float, float]:
    if not meta_path.is_file():
        return float("nan"), float("nan")
    meta = _read_json(meta_path)
    op_w = float(meta.get("operating_point_western", {}).get("threshold", np.nan))
    op_t = float(meta.get("operating_point_tcm", {}).get("threshold", np.nan))
    return op_w, op_t


def _clip_dca(df: pd.DataFrame, plot_min: float, plot_max: float, op_pts: Sequence[float]) -> pd.DataFrame:
    x_max = plot_max
    for pt in op_pts:
        if np.isfinite(pt):
            x_max = max(x_max, float(pt) + 0.02)
    return df[(df["threshold"] >= plot_min) & (df["threshold"] <= x_max)].copy()


FOOTER_Y_DEFAULT = -0.14
FOOTER_Y_SUPPLEMENT = -0.19  # default + 1/3 of (-0.28 → -0.14) adjustment
FOOTER_Y_STAGE1_SMALL = -0.36
FOOTER_Y_STAGE2_MAIN_MULTI = -0.20
FOOTER_Y_STAGE2_OVERALL_MULTI = -0.38  # overall panel: Op. pt clear of xlabel
X_LABELPAD_STAGE2_MAIN_MULTI = 6.0
X_LABELPAD_STAGE2_OVERALL_MULTI = 4.0
# Stage2 multi-panel typography (match single-panel Stage2 DCA)
FS_STAGE2_SECTION = 12
FS_STAGE2_SUPTITLE = 13
FS_STAGE2_SUPTITLE_MULTI = 11  # subordinate to section headers in catboost4+rf6 composite
Y_STAGE2_SECTION_TAG = 1.10
Y_STAGE2_OVERALL_SUBTITLE = 0.96
# CatBoost tags stay fixed in figure coords (do not move with plots)
Y_STAGE2_CB_SECTION_FIG = 0.955
Y_STAGE2_CB_OVERALL_FIG = 0.933
# Plot bands (figure coords). Slice rows are shifted down by SLICE_DOWN vs prior nested layout.
Y_STAGE2_SLICE_DOWN = 0.02
# Extra clearance under Overall for "Threshold probability" without moving Overall panels.
Y_STAGE2_OVERALL_XLABEL_GAP = 0.040
Y_STAGE2_CB_OVERALL_TOP = 0.918
Y_STAGE2_CB_OVERALL_BOTTOM = 0.792
Y_STAGE2_CB_SLICE_TOP = 0.750 - Y_STAGE2_SLICE_DOWN - Y_STAGE2_OVERALL_XLABEL_GAP
Y_STAGE2_CB_SLICE_BOTTOM = 0.615 - Y_STAGE2_SLICE_DOWN - Y_STAGE2_OVERALL_XLABEL_GAP
# RF rows 3+4: tied down by RF_BLOCK_DOWN; then gap shrunk by RF_GAP_SHRINK (row4 rises toward row3).
Y_STAGE2_RF_BLOCK_DOWN = 0.04
Y_STAGE2_RF_GAP_SHRINK = 0.02
Y_STAGE2_RF_OVERALL_TOP = 0.485 - Y_STAGE2_RF_BLOCK_DOWN
Y_STAGE2_RF_OVERALL_BOTTOM = 0.329 - Y_STAGE2_RF_BLOCK_DOWN
Y_STAGE2_RF_SLICE_TOP = (
    0.277 - Y_STAGE2_SLICE_DOWN - Y_STAGE2_OVERALL_XLABEL_GAP - Y_STAGE2_RF_BLOCK_DOWN + Y_STAGE2_RF_GAP_SHRINK
)
Y_STAGE2_RF_SLICE_BOTTOM = (
    0.110 - Y_STAGE2_SLICE_DOWN - Y_STAGE2_OVERALL_XLABEL_GAP - Y_STAGE2_RF_BLOCK_DOWN + Y_STAGE2_RF_GAP_SHRINK
)
Y_STAGE2_LEGEND_Y = 0.014 - 0.08
Y_STAGE2_FIGSIZE = (10, 9.7)


def build_stage2_main_figure_multi(
    batch_root: Path,
    out_path: Path,
    indices: Tuple[int, ...] = (4, 6),
) -> None:
    """Stage2 main DCA: one vertical block per model index (overall + disease slices)."""
    apply_pub_style()
    defaults = stage_defaults("stage2")
    plot_min = defaults["plot_min"]
    plot_max = defaults["plot_max"]
    if len(indices) != 2:
        raise ValueError("build_stage2_main_figure_multi expects exactly two model indices")

    fig = plt.figure(figsize=Y_STAGE2_FIGSIZE)
    # Overall and disease-slice rows are independent bands so slice rows can shift alone.
    overall_specs = [
        GridSpec(
            1,
            1,
            figure=fig,
            left=0.10,
            right=0.98,
            top=Y_STAGE2_CB_OVERALL_TOP,
            bottom=Y_STAGE2_CB_OVERALL_BOTTOM,
        ),
        GridSpec(
            1,
            1,
            figure=fig,
            left=0.10,
            right=0.98,
            top=Y_STAGE2_RF_OVERALL_TOP,
            bottom=Y_STAGE2_RF_OVERALL_BOTTOM,
        ),
    ]
    slice_specs = [
        GridSpec(
            1,
            3,
            figure=fig,
            wspace=0.35,
            left=0.10,
            right=0.98,
            top=Y_STAGE2_CB_SLICE_TOP,
            bottom=Y_STAGE2_CB_SLICE_BOTTOM,
        ),
        GridSpec(
            1,
            3,
            figure=fig,
            wspace=0.35,
            left=0.10,
            right=0.98,
            top=Y_STAGE2_RF_SLICE_TOP,
            bottom=Y_STAGE2_RF_SLICE_BOTTOM,
        ),
    ]
    first_block_labels: Optional[Tuple[str, str]] = None

    for row, idx in enumerate(indices):
        run_dir = resolve_overlay_dir(batch_root, "stage2", idx)
        name = model_label(idx)

        dca_w, dca_t, op_w, op_t = load_overlay_curves(run_dir, "overall")
        if np.isfinite(op_w) and np.isfinite(op_t):
            op_suffix = f" (Op. pt W={op_w:.2f}, TCM={op_t:.2f})"
        elif np.isfinite(op_w):
            op_suffix = f" (Op. pt W={op_w:.2f})"
        elif np.isfinite(op_t):
            op_suffix = f" (Op. pt TCM={op_t:.2f})"
        else:
            op_suffix = ""
        ax_o = fig.add_subplot(overall_specs[row][0])
        _draw_overlay_axes(
            ax_o,
            dca_w,
            dca_t,
            op_w,
            op_t,
            "",
            plot_min,
            plot_max,
            footer_y=FOOTER_Y_STAGE2_OVERALL_MULTI,
            xlabel_pad=X_LABELPAD_STAGE2_OVERALL_MULTI,
            show_footer=False,
        )
        section_label = f"{name} (index {idx})"
        overall_label = f"Overall{op_suffix}"
        if row == 0:
            first_block_labels = (section_label, overall_label)
        else:
            ax_o.text(
                0.5,
                Y_STAGE2_SECTION_TAG,
                section_label,
                transform=ax_o.transAxes,
                ha="center",
                va="bottom",
                fontsize=FS_STAGE2_SECTION,
                fontweight="bold",
            )
            ax_o.text(
                0.5,
                Y_STAGE2_OVERALL_SUBTITLE,
                overall_label,
                transform=ax_o.transAxes,
                ha="center",
                va="bottom",
                fontsize=10,
            )

        for s_i, sid in enumerate(STAGE2_DISEASE_SLICES):
            sw, st, ow, ot = load_overlay_curves(run_dir, sid)
            slice_title = SLICE_DISPLAY.get(sid, sid)
            if np.isfinite(ow) and np.isfinite(ot):
                slice_title += f"  (Op. pt W={ow:.2f}, TCM={ot:.2f})"
            elif np.isfinite(ow):
                slice_title += f"  (Op. pt W={ow:.2f})"
            elif np.isfinite(ot):
                slice_title += f"  (Op. pt TCM={ot:.2f})"
            ax_s = fig.add_subplot(slice_specs[row][0, s_i])
            _draw_overlay_axes(
                ax_s,
                sw,
                st,
                ow,
                ot,
                slice_title,
                plot_min,
                plot_max,
                footer_y=FOOTER_Y_STAGE2_MAIN_MULTI,
                xlabel_pad=X_LABELPAD_STAGE2_MAIN_MULTI,
                show_footer=False,
            )

    fig.suptitle(
        "Overlay DCA — Stage2 (paired Baseline Model vs TCM-integrated, best cross)",
        y=0.995,
        fontsize=FS_STAGE2_SUPTITLE_MULTI,
    )
    if first_block_labels is not None:
        section_label, overall_label = first_block_labels
        fig.text(
            0.54,
            Y_STAGE2_CB_SECTION_FIG,
            section_label,
            ha="center",
            va="bottom",
            fontsize=FS_STAGE2_SECTION,
            fontweight="bold",
        )
        fig.text(
            0.54,
            Y_STAGE2_CB_OVERALL_FIG,
            overall_label,
            ha="center",
            va="bottom",
            fontsize=10,
        )
    _add_shared_overlay_legend(fig, y=Y_STAGE2_LEGEND_Y)
    save_pub_figure(fig, out_path, dpi=300)


def _draw_overlay_axes(
    ax: plt.Axes,
    dca_w: pd.DataFrame,
    dca_t: pd.DataFrame,
    op_w: float,
    op_t: float,
    title: str,
    plot_min: float,
    plot_max: float,
    footer_y: float = FOOTER_Y_DEFAULT,
    xlabel_pad: float = 4.0,
    show_footer: bool = True,
) -> None:
    pw = _clip_dca(dca_w, plot_min, plot_max, [op_w, op_t])
    pt = _clip_dca(dca_t, plot_min, plot_max, [op_w, op_t])
    ax.plot(pw["threshold"], pw["model"], color=COLOR_WESTERN, linewidth=2)
    ax.plot(pt["threshold"], pt["model"], color=COLOR_TCM, linewidth=2)
    ax.plot(pw["threshold"], pw["treat_all"], color=COLOR_TREAT_ALL, linestyle="--", linewidth=1.5)
    ax.plot(pw["threshold"], pw["treat_none"], color=COLOR_TREAT_NONE, linestyle=":", linewidth=1.5)
    if np.isfinite(op_w):
        ax.axvline(op_w, color=COLOR_WESTERN, linestyle="-.", linewidth=1, alpha=0.75)
    if np.isfinite(op_t):
        ax.axvline(op_t, color=COLOR_TCM, linestyle="-.", linewidth=1, alpha=0.75)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Threshold probability", fontsize=9, labelpad=xlabel_pad)
    ax.set_ylabel("Net benefit", fontsize=9)
    x_lo = float(min(pw["threshold"].min(), pt["threshold"].min()))
    x_hi = float(max(pw["threshold"].max(), pt["threshold"].max()))
    ax.set_xlim(x_lo, x_hi)
    if np.isfinite(op_w) and np.isfinite(op_t):
        footer = f"Op. pt W={op_w:.2f}   Op. pt TCM={op_t:.2f}"
    elif np.isfinite(op_w):
        footer = f"Op. pt W={op_w:.2f}"
    elif np.isfinite(op_t):
        footer = f"Op. pt TCM={op_t:.2f}"
    else:
        footer = ""
    if footer and show_footer:
        ax.text(0.5, footer_y, footer, transform=ax.transAxes, ha="center", va="top", fontsize=8)
    ax.tick_params(axis="x", pad=2)


def _add_shared_overlay_legend(fig: plt.Figure, y: float = 0.015) -> None:
    handles = [
        Line2D([0], [0], color=COLOR_WESTERN, linewidth=2, label=LABEL_WESTERN),
        Line2D([0], [0], color=COLOR_TCM, linewidth=2, label=LABEL_TCM),
        Line2D([0], [0], color=COLOR_TREAT_ALL, linestyle="--", linewidth=1.5, label=LABEL_TREAT_ALL),
        Line2D([0], [0], color=COLOR_TREAT_NONE, linestyle=":", linewidth=1.5, label=LABEL_TREAT_NONE),
    ]
    fig.legend(
        handles=handles,
        labels=[LABEL_WESTERN, LABEL_TCM, LABEL_TREAT_ALL, LABEL_TREAT_NONE],
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        bbox_transform=fig.transFigure,
        ncol=4,
        frameon=True,
        fontsize=9,
    )


def _save_dca_composite_figure(fig: plt.Figure, out_path: Path, bottom: float = 0.12, legend_y: float = 0.015) -> None:
    fig.subplots_adjust(bottom=bottom)
    _add_shared_overlay_legend(fig, y=legend_y)
    save_pub_figure(fig, out_path, dpi=300)


def model_label(index: int, stage: str = "") -> str:
    name = MODEL_INDEX_MAP.get(index, f"Model {index}")
    if stage.lower() == "stage3":
        return f"{name}3" if not name.endswith("3") else name
    return name


def build_stage1_main_figure(
    batch_root: Path,
    out_path: Path,
    indices: Tuple[int, int] = (1, 6),
) -> None:
    apply_pub_style()
    defaults = stage_defaults("stage1")
    plot_min = defaults["plot_min"]
    plot_max = defaults["plot_max"]
    fig = plt.figure(figsize=(17, 11))
    gs = GridSpec(2, 2, figure=fig, hspace=0.52, wspace=0.38, left=0.07, right=0.98, top=0.90, bottom=0.16)

    for row, idx in enumerate(indices):
        run_dir = resolve_overlay_dir(batch_root, "stage1", idx)
        name = model_label(idx)
        dca_w, dca_t, op_w, op_t = load_overlay_curves(run_dir, "overall")
        ax_o = fig.add_subplot(gs[row, 0])
        _draw_overlay_axes(
            ax_o,
            dca_w,
            dca_t,
            op_w,
            op_t,
            f"{name} — Overall",
            plot_min,
            plot_max,
        )
        sub_gs = gs[row, 1].subgridspec(2, 2, hspace=0.82, wspace=0.38)
        for s_i, sid in enumerate(STAGE1_DISEASE_SLICES):
            sw, st, ow, ot = load_overlay_curves(run_dir, sid)
            ax_s = fig.add_subplot(sub_gs[s_i // 2, s_i % 2])
            _draw_overlay_axes(
                ax_s,
                sw,
                st,
                ow,
                ot,
                SLICE_DISPLAY.get(sid, sid),
                plot_min,
                plot_max,
                footer_y=FOOTER_Y_STAGE1_SMALL,
            )
        fig.text(
            0.5,
            0.94 - row * 0.48,
            f"{name} (index {idx})",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

    fig.suptitle("Overlay DCA — Stage1 (paired Baseline Model vs TCM-integrated)", y=0.97, fontsize=13)
    _save_dca_composite_figure(fig, out_path, bottom=0.16, legend_y=0.012)


def build_stage2_main_figure(
    batch_root: Path,
    out_path: Path,
    model_index: int = 4,
) -> None:
    apply_pub_style()
    defaults = stage_defaults("stage2")
    plot_min = defaults["plot_min"]
    plot_max = defaults["plot_max"]
    run_dir = resolve_overlay_dir(batch_root, "stage2", model_index)
    name = model_label(model_index)

    fig = plt.figure(figsize=(10, 11))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1.2], hspace=0.42, left=0.10, right=0.98, top=0.92, bottom=0.14)

    dca_w, dca_t, op_w, op_t = load_overlay_curves(run_dir, "overall")
    ax_o = fig.add_subplot(gs[0])
    _draw_overlay_axes(ax_o, dca_w, dca_t, op_w, op_t, f"{name} — Overall", plot_min, plot_max)

    sub_gs = gs[1].subgridspec(1, 3, wspace=0.35)
    for s_i, sid in enumerate(STAGE2_DISEASE_SLICES):
        sw, st, ow, ot = load_overlay_curves(run_dir, sid)
        ax_s = fig.add_subplot(sub_gs[0, s_i])
        _draw_overlay_axes(ax_s, sw, st, ow, ot, SLICE_DISPLAY.get(sid, sid), plot_min, plot_max)

    fig.suptitle(
        f"Overlay DCA — Stage2 ({name}, index {model_index})",
        y=0.98,
        fontsize=13,
    )
    _save_dca_composite_figure(fig, out_path, bottom=0.14, legend_y=0.012)


def build_stage3_main_figure(
    batch_root: Path,
    out_path: Path,
    model_index: int = 3,
) -> None:
    apply_pub_style()
    defaults = stage_defaults("stage3")
    plot_min = defaults["plot_min"]
    plot_max = defaults["plot_max"]
    run_dir = resolve_stage3_overlay_dir(batch_root, model_index)
    name = model_label(model_index, "stage3")
    dca_w, dca_t, op_w, op_t = load_overlay_curves(run_dir, "overall")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    fig.subplots_adjust(bottom=0.22)
    _draw_overlay_axes(
        ax,
        dca_w,
        dca_t,
        op_w,
        op_t,
        SLICE_DISPLAY[STAGE3_E3_PAIR_ID],
        plot_min,
        plot_max,
    )
    fig.suptitle(
        f"Overlay DCA — Stage3 E3 vs (E1+E2) ({name}, index {model_index})",
        y=0.98,
        fontsize=12,
    )
    _save_dca_composite_figure(fig, out_path, bottom=0.22, legend_y=0.012)


def build_overall_grid_figure(
    batch_root: Path,
    stage: str,
    out_path: Path,
    model_indices: Sequence[int],
    ncol: int = 3,
) -> None:
    apply_pub_style()
    defaults = stage_defaults(stage)
    plot_min = defaults["plot_min"]
    plot_max = defaults["plot_max"]
    n = len(model_indices)
    nrows = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(5 * ncol, 4.2 * nrows))
    gs = GridSpec(nrows, ncol, figure=fig, hspace=0.58, wspace=0.32, bottom=0.14, top=0.92)

    for i, idx in enumerate(model_indices):
        run_dir = resolve_overlay_dir(batch_root, stage, idx)
        name = model_label(idx, stage)
        dca_w, dca_t, op_w, op_t = load_overlay_curves(run_dir, "overall")
        ax = fig.add_subplot(gs[i // ncol, i % ncol])
        _draw_overlay_axes(
            ax,
            dca_w,
            dca_t,
            op_w,
            op_t,
            f"{name} (index {idx})",
            plot_min,
            plot_max,
            footer_y=FOOTER_Y_SUPPLEMENT,
        )

    for j in range(n, nrows * ncol):
        fig.add_subplot(gs[j // ncol, j % ncol]).axis("off")

    stage_title = stage.replace("stage", "Stage").upper()
    fig.suptitle(f"Overlay DCA supplement — {stage_title} overall (paired models)", y=0.98, fontsize=13)
    _save_dca_composite_figure(fig, out_path, bottom=0.14, legend_y=0.01)


def _delta_table_paths(stage_key: str, dataset: str) -> Tuple[Path, str, str, str, str, str]:
    if stage_key == "stage1":
        path_name = (
            "stage2_external_tcm_vs_western.csv"
            if dataset == "external"
            else "stage2_oof_tcm_vs_western.csv"
        )
        delta_col = "delta_auc"
        lo_col = "bootstrap_delta_ci_low"
        hi_col = "bootstrap_delta_ci_high"
        fdr_col = "auc_p_fdr_bh"
        metric_label = "ΔAUC (TCM − Baseline)"
    elif stage_key == "stage2":
        path_name = (
            "stage2_external_tcm_vs_western.csv"
            if dataset == "external"
            else "stage2_oof_tcm_vs_western.csv"
        )
        delta_col = "delta_auc"
        lo_col = "bootstrap_delta_ci_low"
        hi_col = "bootstrap_delta_ci_high"
        fdr_col = "auc_p_fdr_bh"
        metric_label = "ΔAUC (TCM − Baseline)"
    elif stage_key == "stage3":
        path_name = (
            "stage3_primary_external_all.csv"
            if dataset == "external"
            else "stage3_primary_oof_all.csv"
        )
        delta_col = "delta_macro_auc_ovr"
        lo_col = "bootstrap_delta_macro_auc_ovr_ci_low"
        hi_col = "bootstrap_delta_macro_auc_ovr_ci_high"
        fdr_col = "auc_p_fdr_bh"
        metric_label = "Δ Macro AUC-OVR (TCM − Baseline)"
    else:
        raise ValueError(stage_key)
    return Path(path_name), delta_col, lo_col, hi_col, fdr_col, metric_label


def load_delta_table(
    comparison_dir: Path,
    stage_key: str,
    dataset: str,
) -> pd.DataFrame:
    comparison_dir = Path(comparison_dir)
    path_name, delta_col, lo_col, hi_col, fdr_col, metric_label = _delta_table_paths(stage_key, dataset)
    path = comparison_dir / path_name
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if stage_key == "stage3" and "model_group" in df.columns:
        df = df[df["model_group"] == "multiclass"].copy()
    df = df[df["model_index"].isin(range(1, 7))].copy()
    df["stage_key"] = stage_key
    df["dataset"] = dataset
    df["delta"] = df[delta_col]
    df["ci_low"] = df[lo_col]
    df["ci_high"] = df[hi_col]
    df["fdr"] = df[fdr_col] if fdr_col in df.columns else np.nan
    df["metric_label"] = metric_label
    df["model_name"] = df["model_index"].map(lambda x: model_label(int(x)))
    return df


def _draw_forest_panel(ax: plt.Axes, sub: pd.DataFrame, title: str) -> None:
    sub = sub.sort_values("model_index")
    y_pos = np.arange(len(sub))
    colors = [
        "#C44E52" if (np.isfinite(r.fdr) and r.fdr < 0.05) else "#4C72B0"
        for r in sub.itertuples()
    ]
    for i, row in enumerate(sub.itertuples(index=False)):
        ax.plot([row.ci_low, row.ci_high], [y_pos[i], y_pos[i]], color="#333333", linewidth=1.5)
        ax.scatter(row.delta, y_pos[i], color=colors[i], s=45, zorder=3)
    ax.axvline(0, color="#999999", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([row.model_name for row in sub.itertuples(index=False)], fontsize=9)
    ax.set_xlabel(sub["metric_label"].iloc[0], fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def build_statistics_forest_figure(
    stage1_comparison_dir: Path,
    stage2_comparison_dir: Path,
    stage3_comparison_dir: Path,
    out_path: Path,
) -> pd.DataFrame:
    apply_pub_style()
    dirs = {
        "stage1": Path(stage1_comparison_dir),
        "stage2": Path(stage2_comparison_dir),
        "stage3": Path(stage3_comparison_dir),
    }
    parts: List[pd.DataFrame] = []
    for sk, comp_dir in dirs.items():
        parts.append(load_delta_table(comp_dir, sk, "external"))
        parts.append(load_delta_table(comp_dir, sk, "oof"))
    plot_df = pd.concat(parts, ignore_index=True)

    stage_order = ["stage1", "stage2", "stage3"]
    stage_titles = {
        "stage1": "Stage1 (ΔAUC)",
        "stage2": "Stage2 (ΔAUC)",
        "stage3": "Stage3 (Δ Macro AUC-OVR)",
    }
    row_titles = ["Internal validation (OOF)", "External validation"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9.5), sharey=False)
    fig.subplots_adjust(hspace=0.38, wspace=0.32, bottom=0.10, top=0.90)

    for col, sk in enumerate(stage_order):
        for row, dataset in enumerate(["oof", "external"]):
            sub = plot_df[(plot_df["stage_key"] == sk) & (plot_df["dataset"] == dataset)]
            title = f"{stage_titles[sk]} — {row_titles[row]}"
            _draw_forest_panel(axes[row, col], sub, title)

    fig.suptitle(
        "TCM-integrated vs Baseline Model (models 1–6, bootstrap 95% CI)",
        y=0.96,
        fontsize=13,
    )
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4C72B0", markersize=8, label="FDR ≥ 0.05"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#C44E52", markersize=8, label="FDR < 0.05"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        bbox_transform=fig.transFigure,
        ncol=2,
        frameon=True,
        fontsize=9,
    )
    save_pub_figure(fig, out_path, dpi=300)
    return plot_df


def build_statistics_forest_figure_stage12(
    stage1_comparison_dir: Path,
    stage2_comparison_dir: Path,
    out_path: Path,
    *,
    dataset: str,
    title: str,
) -> pd.DataFrame:
    """Stage1+Stage2 only (no Stage3). dataset is 'oof' or 'external'."""
    if dataset not in {"oof", "external"}:
        raise ValueError(f"dataset must be oof|external, got {dataset!r}")

    apply_pub_style()
    dirs = {
        "stage1": Path(stage1_comparison_dir),
        "stage2": Path(stage2_comparison_dir),
    }
    parts: List[pd.DataFrame] = []
    for sk, comp_dir in dirs.items():
        parts.append(load_delta_table(comp_dir, sk, dataset))
    plot_df = pd.concat(parts, ignore_index=True)

    stage_titles = {
        "stage1": "Stage1 (ΔAUC)",
        "stage2": "Stage2 (ΔAUC)",
    }
    row_label = "Internal validation (OOF)" if dataset == "oof" else "External validation"

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), sharey=False)
    fig.subplots_adjust(wspace=0.32, bottom=0.18, top=0.84, left=0.08, right=0.98)

    for col, sk in enumerate(["stage1", "stage2"]):
        sub = plot_df[plot_df["stage_key"] == sk]
        panel_title = f"{stage_titles[sk]} — {row_label}"
        _draw_forest_panel(axes[col], sub, panel_title)

    fig.suptitle(title, y=0.96, fontsize=12)
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4C72B0", markersize=8, label="FDR ≥ 0.05"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#C44E52", markersize=8, label="FDR < 0.05"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        bbox_transform=fig.transFigure,
        ncol=2,
        frameon=True,
        fontsize=9,
    )
    save_pub_figure(fig, out_path, dpi=300)
    return plot_df
