# -*- coding: utf-8 -*-
"""Publication figure style: JPG 300 dpi, shared colors and fonts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt

COLOR_WESTERN = "#2166AC"
COLOR_BASELINE = COLOR_WESTERN  # Objective Factor Baseline Model
COLOR_TCM = "#D6604D"
COLOR_TREAT_ALL = "#969696"
COLOR_TREAT_NONE = "#BDBDBD"
COLOR_POS = "#C44E52"
COLOR_NEG = "#4C72B0"

MODEL_INDEX_MAP: dict[int, str] = {
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

METRIC_LABELS: dict[str, str] = {
    "auc": "AUC",
    "auprc": "AUPRC",
    "acc": "Accuracy",
    "f1": "F1",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "precision": "Precision",
}


def apply_pub_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": True,
            "legend.fontsize": 9,
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def save_pub_figure(fig: plt.Figure, out_path: Path, *, dpi: int = 300, quality: int = 95) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        fig.savefig(
            out_path,
            format="jpeg",
            dpi=dpi,
            bbox_inches="tight",
            pil_kwargs={"quality": quality},
        )
    else:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def style_axes(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
