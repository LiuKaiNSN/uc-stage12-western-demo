# -*- coding: utf-8 -*-
"""Shared decision curve analysis utilities for Stage1/Stage2/Stage3 external validation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Disease-focused slice IDs (binary, both classes present).
STAGE1_DISEASE_SLICES: Tuple[str, ...] = (
    "uc_vs_ie_ibs_pool",
    "cd_vs_ie_ibs_pool",
    "ic_vs_ie_ibs_pool",
    "crc_vs_ie_ibs_pool",
)
STAGE2_DISEASE_SLICES: Tuple[str, ...] = (
    "uc_vs_cd",
    "uc_vs_ic",
    "uc_vs_crc",
)

STAGE_DEFAULTS: Dict[str, Dict[str, float]] = {
    "stage1": {"min_sens": 0.95, "plot_min": 0.05, "plot_max": 0.50},
    "stage2": {"min_sens": 0.90, "plot_min": 0.10, "plot_max": 0.60},
    "stage3": {"min_sens": 0.85, "plot_min": 0.10, "plot_max": 0.70},
}

STAGE3_POSITIVE_LABEL = "E3 (extensive)"
STAGE3_NEGATIVE_LABEL = "E1+E2 (limited)"
STAGE3_E3_PAIR_ID = "e3_vs_e12"

RUN_DIR_SKIP_NAMES = {"旧", "旧2", "dca_batch", "dca_final", "dca_batch_stage3", "summary_reports"}
RUN_DIR_PATTERN = re.compile(r"^(\d+)_run", re.IGNORECASE)

EXTERNAL_PRED_REL = Path("external_validation_binary") / "external_predictions.csv"
STAGE3_EXTERNAL_PRED_REL = Path("external_validation") / "external_predictions.csv"
STAGE1_SLICE_DIR = Path("external_validation_binary") / "type_slices"
STAGE2_SLICE_DIR = Path("external_validation_binary") / "type_montreal_slices"


def stage_defaults(stage: str) -> Dict[str, float]:
    key = stage.lower()
    if key not in STAGE_DEFAULTS:
        raise ValueError(f"Unknown stage: {stage!r}; expected stage1, stage2, or stage3.")
    return STAGE_DEFAULTS[key]


def disease_slices_for_stage(stage: str) -> Tuple[str, ...]:
    key = stage.lower()
    if key == "stage1":
        return STAGE1_DISEASE_SLICES
    if key == "stage2":
        return STAGE2_DISEASE_SLICES
    raise ValueError(f"Unknown stage: {stage!r}")


def slice_dir_for_stage(stage: str) -> Path:
    key = stage.lower()
    if key == "stage1":
        return STAGE1_SLICE_DIR
    if key == "stage2":
        return STAGE2_SLICE_DIR
    raise ValueError(f"Unknown stage: {stage!r}")


def resolve_pred_csv(model_run_dir: Path) -> Path:
    path = Path(model_run_dir) / EXTERNAL_PRED_REL
    if not path.is_file():
        raise FileNotFoundError(f"Missing external predictions: {path}")
    return path


def resolve_stage3_pred_csv(model_run_dir: Path) -> Path:
    path = Path(model_run_dir) / STAGE3_EXTERNAL_PRED_REL
    if not path.is_file():
        raise FileNotFoundError(f"Missing Stage3 external predictions: {path}")
    return path


def load_stage3_predictions(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ("y_true", "prob_1", "prob_2", "prob_3"):
        if col not in df.columns:
            raise ValueError(f"{csv_path} missing column {col!r}")
    out = df.copy()
    out["y_true"] = out["y_true"].astype(int)
    for col in ("prob_1", "prob_2", "prob_3"):
        out[col] = out[col].astype(float)
    if "No" in out.columns:
        out["No"] = out["No"].astype(str)
    return out


def binary_e3_vs_e12(df: pd.DataFrame) -> pd.DataFrame:
    """Montreal E3 vs (E1+E2): y_true 0/1/2 -> binary E3=1; score = P(E3)=prob_3."""
    valid = df["y_true"].isin([0, 1, 2])
    if not valid.all():
        bad = sorted(set(df.loc[~valid, "y_true"].astype(int).tolist()))
        raise ValueError(f"Unexpected y_true values (expected 0/1/2 for E1/E2/E3): {bad}")
    sub = df.loc[valid].copy()
    out = sub.copy()
    out["y_true_multiclass"] = out["y_true"].astype(int)
    out["y_true"] = (out["y_true_multiclass"] == 2).astype(int)
    out["prob_1"] = out["prob_3"].astype(float)
    return out


def discover_stage3_run_dirs(stage_root: Path) -> List[Path]:
    stage_root = Path(stage_root)
    if not stage_root.is_dir():
        raise NotADirectoryError(stage_root)

    runs: List[Tuple[int, Path]] = []
    for child in stage_root.iterdir():
        if not child.is_dir():
            continue
        if child.name in RUN_DIR_SKIP_NAMES or child.name.startswith("summary_reports"):
            continue
        m = RUN_DIR_PATTERN.match(child.name)
        if not m:
            continue
        if not (child / STAGE3_EXTERNAL_PRED_REL).is_file():
            continue
        runs.append((int(m.group(1)), child))
    runs.sort(key=lambda x: x[0])
    return [p for _, p in runs]


def run_stage3_e3_dca(
    df_stage3: pd.DataFrame,
    out_dir: Path,
    label_model: str,
    pt_min: float = 0.01,
    pt_max: float = 0.80,
    pt_step: float = 0.01,
    plot_min: Optional[float] = None,
    plot_max: Optional[float] = None,
    min_sens: Optional[float] = None,
) -> Dict[str, object]:
    """DCA for E3 vs (E1+E2) using P(E3) on Stage3 triclass external predictions."""
    defaults = stage_defaults("stage3")
    if plot_min is None:
        plot_min = defaults["plot_min"]
    if plot_max is None:
        plot_max = defaults["plot_max"]
    if min_sens is None:
        min_sens = defaults["min_sens"]

    binary_df = binary_e3_vs_e12(df_stage3)
    meta = run_single_dca(
        binary_df,
        stage="stage3",
        out_dir=out_dir,
        label_model=label_model,
        pt_min=pt_min,
        pt_max=pt_max,
        pt_step=pt_step,
        plot_min=plot_min,
        plot_max=plot_max,
        min_sens=min_sens,
        slice_id=STAGE3_E3_PAIR_ID,
        title_suffix=STAGE3_E3_PAIR_ID,
    )
    meta.update(
        {
            "pair_id": STAGE3_E3_PAIR_ID,
            "positive_class": STAGE3_POSITIVE_LABEL,
            "negative_class": STAGE3_NEGATIVE_LABEL,
            "score_column": "prob_3",
            "n_e1": int((binary_df["y_true_multiclass"] == 0).sum()),
            "n_e2": int((binary_df["y_true_multiclass"] == 1).sum()),
            "n_e3": int((binary_df["y_true_multiclass"] == 2).sum()),
        }
    )
    meta_path = out_dir / "dca_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
    return meta


def slice_pred_csv(model_run_dir: Path, stage: str, slice_id: str) -> Path:
    path = Path(model_run_dir) / slice_dir_for_stage(stage) / f"predictions_{slice_id}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing slice predictions: {path}")
    return path


def load_predictions(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ("y_true", "prob_1"):
        if col not in df.columns:
            raise ValueError(f"{csv_path} missing column {col!r}")
    out = df.copy()
    out["y_true"] = out["y_true"].astype(int)
    out["prob_1"] = out["prob_1"].astype(float)
    if "No" in out.columns:
        out["No"] = out["No"].astype(str)
    return out


def extract_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    y = df["y_true"].to_numpy(dtype=int)
    prob = df["prob_1"].to_numpy(dtype=float)
    if len(y) == 0:
        raise ValueError("Empty prediction dataframe.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y_true must be binary 0/1 for DCA.")
    return y, prob


def net_benefit_model(y: np.ndarray, prob: np.ndarray, pt: float) -> float:
    n = len(y)
    if n == 0 or pt >= 1.0:
        return 0.0
    pred = prob >= pt
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    return tp / n - (fp / n) * (pt / (1.0 - pt))


def net_benefit_all(y: np.ndarray, pt: float) -> float:
    n = len(y)
    if n == 0 or pt >= 1.0:
        return 0.0
    tp = int(np.sum(y == 1))
    fp = int(np.sum(y == 0))
    return tp / n - (fp / n) * (pt / (1.0 - pt))


def net_benefit_none(_pt: float) -> float:
    return 0.0


def compute_dca(y: np.ndarray, prob: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    for pt in thresholds:
        rows.append(
            {
                "threshold": float(pt),
                "model": net_benefit_model(y, prob, float(pt)),
                "treat_all": net_benefit_all(y, float(pt)),
                "treat_none": net_benefit_none(float(pt)),
            }
        )
    return pd.DataFrame(rows)


def metrics_at_threshold(y: np.ndarray, prob: np.ndarray, pt: float) -> Dict[str, float]:
    pred = prob >= pt
    tp = int(np.sum(pred & (y == 1)))
    tn = int(np.sum((~pred) & (y == 0)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum((~pred) & (y == 1)))

    def _rate(num: int, den: int) -> float:
        return float(num / den) if den else float("nan")

    return {
        "threshold": float(pt),
        "n": int(len(y)),
        "prevalence": float(np.mean(y == 1)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": _rate(tp, tp + fn),
        "specificity": _rate(tn, tn + fp),
        "ppv": _rate(tp, tp + fp),
        "npv": _rate(tn, tn + fn),
        "accuracy": _rate(tp + tn, len(y)),
    }


def pick_operating_point(
    y: np.ndarray,
    prob: np.ndarray,
    thresholds: np.ndarray,
    min_sens: float,
) -> Dict[str, float]:
    """Rule A/B: among thresholds with sensitivity >= min_sens, maximize specificity."""
    rows = [metrics_at_threshold(y, prob, float(pt)) for pt in thresholds]
    df = pd.DataFrame(rows)
    ok = df[df["sensitivity"] >= min_sens]
    if len(ok):
        row = ok.sort_values(["specificity", "threshold"], ascending=[False, True]).iloc[0]
    else:
        row = df.sort_values(["sensitivity", "specificity"], ascending=[False, False]).iloc[0]
    out = row.to_dict()
    out["net_benefit_model"] = net_benefit_model(y, prob, float(out["threshold"]))
    out["net_benefit_all"] = net_benefit_all(y, float(out["threshold"]))
    out["net_benefit_none"] = 0.0
    return out


def useful_threshold_range(dca_df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    better = dca_df[
        dca_df["model"] > np.maximum(dca_df["treat_all"], dca_df["treat_none"])
    ]
    if better.empty:
        return None, None
    return float(better["threshold"].min()), float(better["threshold"].max())


def make_thresholds(pt_min: float, pt_max: float, pt_step: float) -> np.ndarray:
    if pt_max < pt_min:
        raise ValueError("pt_max must be >= pt_min")
    count = int(round((pt_max - pt_min) / pt_step)) + 1
    thresholds = pt_min + pt_step * np.arange(count, dtype=float)
    return np.clip(thresholds, pt_min, pt_max)


def preset_threshold_rows(
    y: np.ndarray,
    prob: np.ndarray,
    thresholds: np.ndarray,
    operating_pt: float,
) -> pd.DataFrame:
    preset = sorted({0.10, 0.20, 0.30, 0.50, round(float(operating_pt), 4)})
    rows = []
    for pt in preset:
        if pt < thresholds.min() - 1e-9 or pt > thresholds.max() + 1e-9:
            continue
        m = metrics_at_threshold(y, prob, pt)
        rows.append(
            {
                **m,
                "net_benefit_model": net_benefit_model(y, prob, pt),
                "net_benefit_all": net_benefit_all(y, pt),
                "net_benefit_none": 0.0,
            }
        )
    return pd.DataFrame(rows)


def run_single_dca(
    df: pd.DataFrame,
    stage: str,
    out_dir: Path,
    label_model: str,
    pt_min: float = 0.01,
    pt_max: float = 0.80,
    pt_step: float = 0.01,
    plot_min: Optional[float] = None,
    plot_max: Optional[float] = None,
    min_sens: Optional[float] = None,
    slice_id: Optional[str] = None,
    title_suffix: str = "overall",
) -> Dict[str, object]:
    defaults = stage_defaults(stage)
    if plot_min is None:
        plot_min = defaults["plot_min"]
    if plot_max is None:
        plot_max = defaults["plot_max"]
    if min_sens is None:
        min_sens = defaults["min_sens"]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y, prob = extract_xy(df)
    thresholds = make_thresholds(pt_min, pt_max, pt_step)
    dca_df = compute_dca(y, prob, thresholds)
    dca_df.to_csv(out_dir / "dca_curve.csv", index=False)

    op = pick_operating_point(y, prob, thresholds, min_sens=min_sens)
    pd.DataFrame([op]).to_csv(out_dir / "operating_point.csv", index=False)

    lo, hi = useful_threshold_range(dca_df)
    meta = {
        "stage": stage,
        "slice_id": slice_id or title_suffix,
        "label_model": label_model,
        "n": int(len(y)),
        "prevalence": float(np.mean(y == 1)),
        "min_sens_rule": float(min_sens),
        "useful_pt_min": lo,
        "useful_pt_max": hi,
        "operating_point": op,
    }
    with open(out_dir / "dca_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    preset_threshold_rows(y, prob, thresholds, op["threshold"]).to_csv(
        out_dir / "dca_preset_thresholds.csv", index=False
    )

    plot_max_eff = max(plot_max, float(op["threshold"]) + 0.02)
    plot_df = dca_df[(dca_df["threshold"] >= plot_min) & (dca_df["threshold"] <= plot_max_eff)]
    plot_dca_single(
        plot_df,
        label_model=label_model,
        operating_pt=float(op["threshold"]),
        title=f"Decision curve ({stage}, {title_suffix})",
        out_png=out_dir / f"dca_{title_suffix}.png",
    )

    return meta


def _save_dca_figure(fig, out_png: Path, *, bottom: float = 0.20) -> None:
    fig.subplots_adjust(bottom=bottom)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _place_dca_legend(ax, ncol: int = 2, fontsize: float = 9) -> None:
    """Place legend below axes so curves in the upper-right are not covered."""
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=ncol,
        frameon=True,
        fontsize=fontsize,
    )


def _place_multipanel_legend(fig, ax, ncol: int = 3, fontsize: float = 8) -> None:
    """Shared figure legend below all panels (avoids overlap with adjacent subplots)."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=ncol,
        frameon=True,
        fontsize=fontsize,
    )


def _save_dca_multipanel_figure(fig, out_png: Path) -> None:
    fig.subplots_adjust(top=0.92, bottom=0.12, hspace=0.38, wspace=0.28)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_dca_single(
    dca_df: pd.DataFrame,
    label_model: str,
    operating_pt: Optional[float],
    title: str,
    out_png: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(dca_df["threshold"], dca_df["model"], label=label_model, linewidth=2)
    ax.plot(dca_df["threshold"], dca_df["treat_all"], label="Treat all", linestyle="--")
    ax.plot(dca_df["threshold"], dca_df["treat_none"], label="Treat none", linestyle=":")
    if operating_pt is not None:
        ax.axvline(
            operating_pt,
            color="gray",
            linestyle="-.",
            linewidth=1,
            label=f"Operating pt={operating_pt:.2f}",
        )
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title(title)
    ax.set_xlim(float(dca_df["threshold"].min()), float(dca_df["threshold"].max()))
    _place_dca_legend(ax, ncol=2)
    _save_dca_figure(fig, out_png)


def plot_dca_overlay(
    dca_western: pd.DataFrame,
    dca_tcm: pd.DataFrame,
    label_western: str,
    label_tcm: str,
    operating_pt_w: Optional[float],
    operating_pt_t: Optional[float],
    title: str,
    out_png: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(dca_western["threshold"], dca_western["model"], label=label_western, linewidth=2)
    ax.plot(dca_tcm["threshold"], dca_tcm["model"], label=label_tcm, linewidth=2)
    ax.plot(dca_western["threshold"], dca_western["treat_all"], label="Treat all", linestyle="--", color="C2")
    ax.plot(dca_western["threshold"], dca_western["treat_none"], label="Treat none", linestyle=":", color="C3")
    if operating_pt_w is not None:
        ax.axvline(operating_pt_w, color="C0", linestyle="-.", linewidth=1, alpha=0.6, label=f"Op. pt W={operating_pt_w:.2f}")
    if operating_pt_t is not None and (operating_pt_t != operating_pt_w):
        ax.axvline(operating_pt_t, color="C1", linestyle="-.", linewidth=1, alpha=0.6, label=f"Op. pt TCM={operating_pt_t:.2f}")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title(title)
    x_max = max(dca_western["threshold"].max(), dca_tcm["threshold"].max())
    for pt in (operating_pt_w, operating_pt_t):
        if pt is not None:
            x_max = max(x_max, pt + 0.02)
    ax.set_xlim(float(dca_western["threshold"].min()), float(x_max))
    _place_dca_legend(ax, ncol=3, fontsize=8)
    _save_dca_figure(fig, out_png, bottom=0.22)


def plot_dca_multipanel_single_model(
    panel_data: Sequence[Tuple[str, pd.DataFrame, float]],
    label_model: str,
    stage: str,
    out_png: Path,
    suptitle: str,
) -> None:
    n = len(panel_data)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)
    for idx, (slice_id, dca_df, op_pt) in enumerate(panel_data):
        ax = axes[idx // ncols][idx % ncols]
        ax.plot(dca_df["threshold"], dca_df["model"], label=label_model, linewidth=2)
        ax.plot(dca_df["threshold"], dca_df["treat_all"], label="Treat all", linestyle="--")
        ax.plot(dca_df["threshold"], dca_df["treat_none"], label="Treat none", linestyle=":")
        ax.axvline(op_pt, color="gray", linestyle="-.", linewidth=1)
        ax.set_title(slice_id)
        ax.set_xlabel("Threshold probability")
        ax.set_ylabel("Net benefit")
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.suptitle(suptitle, y=0.98)
    _place_multipanel_legend(fig, axes[0][0], ncol=3, fontsize=8)
    _save_dca_multipanel_figure(fig, out_png)


def plot_dca_multipanel_overlay(
    panel_data: Sequence[Tuple[str, pd.DataFrame, pd.DataFrame]],
    label_western: str,
    label_tcm: str,
    stage: str,
    out_png: Path,
    suptitle: str,
) -> None:
    n = len(panel_data)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)
    for idx, (slice_id, dca_w, dca_t) in enumerate(panel_data):
        ax = axes[idx // ncols][idx % ncols]
        ax.plot(dca_w["threshold"], dca_w["model"], label=label_western, linewidth=2)
        ax.plot(dca_t["threshold"], dca_t["model"], label=label_tcm, linewidth=2)
        ax.plot(dca_w["threshold"], dca_w["treat_all"], label="Treat all", linestyle="--")
        ax.plot(dca_w["threshold"], dca_w["treat_none"], label="Treat none", linestyle=":")
        ax.set_title(slice_id)
        ax.set_xlabel("Threshold probability")
        ax.set_ylabel("Net benefit")
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.suptitle(suptitle, y=0.98)
    _place_multipanel_legend(fig, axes[0][0], ncol=4, fontsize=8)
    _save_dca_multipanel_figure(fig, out_png)


def discover_run_dirs(stage_root: Path) -> List[Path]:
    stage_root = Path(stage_root)
    if not stage_root.is_dir():
        raise NotADirectoryError(stage_root)

    runs: List[Tuple[int, Path]] = []
    for child in stage_root.iterdir():
        if not child.is_dir():
            continue
        if child.name in RUN_DIR_SKIP_NAMES or child.name.startswith("summary_reports"):
            continue
        m = RUN_DIR_PATTERN.match(child.name)
        if not m:
            continue
        if not (child / EXTERNAL_PRED_REL).is_file():
            continue
        runs.append((int(m.group(1)), child))
    runs.sort(key=lambda x: x[0])
    return [p for _, p in runs]


def run_index_from_dir(run_dir: Path) -> Optional[int]:
    m = RUN_DIR_PATTERN.match(run_dir.name)
    return int(m.group(1)) if m else None


def pair_run_by_index(western_root: Path, tcm_root: Path, run_index: int) -> Tuple[Optional[Path], Optional[Path]]:
    w_match = None
    t_match = None
    for p in discover_run_dirs(western_root):
        if run_index_from_dir(p) == run_index:
            w_match = p
            break
    for p in discover_run_dirs(tcm_root):
        if run_index_from_dir(p) == run_index:
            t_match = p
            break
    return w_match, t_match


def pair_stage3_run_by_index(
    western_root: Path, tcm_root: Path, run_index: int
) -> Tuple[Optional[Path], Optional[Path]]:
    w_match = None
    t_match = None
    for p in discover_stage3_run_dirs(western_root):
        if run_index_from_dir(p) == run_index:
            w_match = p
            break
    for p in discover_stage3_run_dirs(tcm_root):
        if run_index_from_dir(p) == run_index:
            t_match = p
            break
    return w_match, t_match


def merge_stage3_e3_overlay_predictions(
    western_csv: Path, tcm_csv: Path
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge Western vs TCM Stage3 predictions for E3 vs (E1+E2); scores are P(E3)."""
    wdf = binary_e3_vs_e12(load_stage3_predictions(western_csv))
    tdf = binary_e3_vs_e12(load_stage3_predictions(tcm_csv))
    if "No" not in wdf.columns or "No" not in tdf.columns:
        if len(wdf) != len(tdf):
            raise ValueError("Stage3 prediction files differ in length and lack No column for merge.")
        y = wdf["y_true"].to_numpy(dtype=int)
        if not np.array_equal(y, tdf["y_true"].to_numpy(dtype=int)):
            raise ValueError("Stage3 E3 binary y_true mismatch between western and TCM predictions.")
        return y, wdf["prob_1"].to_numpy(dtype=float), tdf["prob_1"].to_numpy(dtype=float)

    merged = wdf[["No", "y_true", "prob_1"]].merge(
        tdf[["No", "y_true", "prob_1"]],
        on="No",
        how="inner",
        suffixes=("_western", "_tcm"),
    )
    if merged.empty:
        raise ValueError(f"No overlapping No between {western_csv} and {tcm_csv}")
    if not (merged["y_true_western"] == merged["y_true_tcm"]).all():
        raise ValueError("Stage3 E3 binary y_true mismatch between western and TCM after merge on No.")
    y = merged["y_true_western"].astype(int).to_numpy()
    prob_w = merged["prob_1_western"].astype(float).to_numpy()
    prob_t = merged["prob_1_tcm"].astype(float).to_numpy()
    return y, prob_w, prob_t


def merge_overlay_predictions(western_csv: Path, tcm_csv: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    wdf = load_predictions(western_csv)
    tdf = load_predictions(tcm_csv)
    if "No" not in wdf.columns or "No" not in tdf.columns:
        if len(wdf) != len(tdf):
            raise ValueError("Prediction files differ in length and lack No column for merge.")
        y = wdf["y_true"].to_numpy(dtype=int)
        if not np.array_equal(y, tdf["y_true"].to_numpy(dtype=int)):
            raise ValueError("y_true mismatch between western and TCM predictions.")
        return y, wdf["prob_1"].to_numpy(dtype=float), tdf["prob_1"].to_numpy(dtype=float)

    merged = wdf[["No", "y_true", "prob_1"]].merge(
        tdf[["No", "prob_1"]],
        on="No",
        how="inner",
        suffixes=("_western", "_tcm"),
    )
    if merged.empty:
        raise ValueError(f"No overlapping No between {western_csv} and {tcm_csv}")
    if not np.array_equal(merged["y_true"].to_numpy(), merged["y_true"].to_numpy()):
        pass
    y = merged["y_true"].astype(int).to_numpy()
    prob_w = merged["prob_1_western"].astype(float).to_numpy()
    prob_t = merged["prob_1_tcm"].astype(float).to_numpy()
    return y, prob_w, prob_t


def delta_nb_dataframe(
    y: np.ndarray,
    prob_w: np.ndarray,
    prob_t: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for pt in thresholds:
        rows.append(
            {
                "threshold": float(pt),
                "nb_western": net_benefit_model(y, prob_w, float(pt)),
                "nb_tcm": net_benefit_model(y, prob_t, float(pt)),
                "delta_nb_tcm_minus_western": net_benefit_model(y, prob_t, float(pt))
                - net_benefit_model(y, prob_w, float(pt)),
            }
        )
    return pd.DataFrame(rows)


def run_overlay_dca(
    western_csv: Path,
    tcm_csv: Path,
    stage: str,
    out_dir: Path,
    label_western: str,
    label_tcm: str,
    pt_min: float,
    pt_max: float,
    pt_step: float,
    plot_min: float,
    plot_max: float,
    min_sens: float,
    slice_id: str,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    y, prob_w, prob_t = merge_overlay_predictions(western_csv, tcm_csv)
    thresholds = make_thresholds(pt_min, pt_max, pt_step)

    dca_w = compute_dca(y, prob_w, thresholds)
    dca_t = compute_dca(y, prob_t, thresholds)
    dca_w.to_csv(out_dir / "dca_curve_western.csv", index=False)
    dca_t.to_csv(out_dir / "dca_curve_tcm.csv", index=False)
    delta_nb_dataframe(y, prob_w, prob_t, thresholds).to_csv(out_dir / "delta_nb.csv", index=False)

    op_w = pick_operating_point(y, prob_w, thresholds, min_sens=min_sens)
    op_t = pick_operating_point(y, prob_t, thresholds, min_sens=min_sens)
    pd.DataFrame([op_w]).to_csv(out_dir / "operating_point_western.csv", index=False)
    pd.DataFrame([op_t]).to_csv(out_dir / "operating_point_tcm.csv", index=False)

    plot_max_eff = max(plot_max, float(op_w["threshold"]) + 0.02, float(op_t["threshold"]) + 0.02)
    plot_w = dca_w[(dca_w["threshold"] >= plot_min) & (dca_w["threshold"] <= plot_max_eff)]
    plot_t = dca_t[(dca_t["threshold"] >= plot_min) & (dca_t["threshold"] <= plot_max_eff)]
    plot_dca_overlay(
        plot_w,
        plot_t,
        label_western=label_western,
        label_tcm=label_tcm,
        operating_pt_w=float(op_w["threshold"]),
        operating_pt_t=float(op_t["threshold"]),
        title=f"Overlay DCA ({stage}, {slice_id})",
        out_png=out_dir / f"dca_overlay_{slice_id}.png",
    )

    meta = {
        "stage": stage,
        "slice_id": slice_id,
        "western_csv": str(western_csv),
        "tcm_csv": str(tcm_csv),
        "n": int(len(y)),
        "prevalence": float(y.mean()),
        "operating_point_western": op_w,
        "operating_point_tcm": op_t,
    }
    with open(out_dir / "dca_overlay_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def run_stage3_e3_overlay_dca(
    western_csv: Path,
    tcm_csv: Path,
    out_dir: Path,
    label_western: str,
    label_tcm: str,
    pt_min: float,
    pt_max: float,
    pt_step: float,
    plot_min: float,
    plot_max: float,
    min_sens: float,
) -> Dict[str, object]:
    """Overlay DCA for Stage3 E3 vs (E1+E2): Western vs TCM P(E3) on the same external cohort."""
    out_dir.mkdir(parents=True, exist_ok=True)
    y, prob_w, prob_t = merge_stage3_e3_overlay_predictions(western_csv, tcm_csv)
    thresholds = make_thresholds(pt_min, pt_max, pt_step)

    dca_w = compute_dca(y, prob_w, thresholds)
    dca_t = compute_dca(y, prob_t, thresholds)
    dca_w.to_csv(out_dir / "dca_curve_western.csv", index=False)
    dca_t.to_csv(out_dir / "dca_curve_tcm.csv", index=False)
    delta_nb_dataframe(y, prob_w, prob_t, thresholds).to_csv(out_dir / "delta_nb.csv", index=False)

    op_w = pick_operating_point(y, prob_w, thresholds, min_sens=min_sens)
    op_t = pick_operating_point(y, prob_t, thresholds, min_sens=min_sens)
    pd.DataFrame([op_w]).to_csv(out_dir / "operating_point_western.csv", index=False)
    pd.DataFrame([op_t]).to_csv(out_dir / "operating_point_tcm.csv", index=False)

    plot_max_eff = max(plot_max, float(op_w["threshold"]) + 0.02, float(op_t["threshold"]) + 0.02)
    plot_w = dca_w[(dca_w["threshold"] >= plot_min) & (dca_w["threshold"] <= plot_max_eff)]
    plot_t = dca_t[(dca_t["threshold"] >= plot_min) & (dca_t["threshold"] <= plot_max_eff)]
    plot_dca_overlay(
        plot_w,
        plot_t,
        label_western=label_western,
        label_tcm=label_tcm,
        operating_pt_w=float(op_w["threshold"]),
        operating_pt_t=float(op_t["threshold"]),
        title=f"Overlay DCA (stage3, {STAGE3_E3_PAIR_ID})",
        out_png=out_dir / f"dca_overlay_{STAGE3_E3_PAIR_ID}.png",
    )

    meta = {
        "stage": "stage3",
        "pair_id": STAGE3_E3_PAIR_ID,
        "positive_class": STAGE3_POSITIVE_LABEL,
        "negative_class": STAGE3_NEGATIVE_LABEL,
        "western_csv": str(western_csv),
        "tcm_csv": str(tcm_csv),
        "n": int(len(y)),
        "prevalence_e3": float(y.mean()),
        "min_sens_rule": float(min_sens),
        "operating_point_western": op_w,
        "operating_point_tcm": op_t,
    }
    with open(out_dir / "dca_overlay_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta
