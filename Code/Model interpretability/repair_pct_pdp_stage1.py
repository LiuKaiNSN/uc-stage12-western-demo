# -*- coding: utf-8 -*-
"""Fast PCT PDP repair (manual grid only, no sklearn brute)."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT))

from interpret_binary_best_model_Stage12 import (  # noqa: E402
    POSITIVE_CLASS_INDEX,
    _load_pipeline,
    _load_X,
    _manual_pdp_curve,
    _manual_pdp_grid_from_series,
    _read_features,
    plot_pdp_combined,
    ranked_plot_filename,
)
from _interpret_feature_labels import display_names
from _pub_plot_style import apply_pub_style, save_pub_figure, style_axes

import matplotlib.pyplot as plt

RUN_DIR = Path(r"F:\KeTi\Project\outputs\Stage1\3_run_20260707_142952")
TRAIN_CSV = Path(r"F:\KeTi\Project\TCM\data\ATrain-Stage1.csv")
OUT_DIR = RUN_DIR / "interpretability"
PUB_DIR = Path(r"F:\KeTi\Project\Figure\Stage12\figure5\stage1")
FEATURE = "PCT"
GRID_POINTS = 25
PDP_N = 400
SEED = 42


def main() -> None:
    pipe = _load_pipeline(RUN_DIR / "final_model.joblib")
    feats = _read_features(RUN_DIR / "final_features_used.txt")
    X_train, _ = _load_X(TRAIN_CSV, feats, "No", "Disease")
    imp = pd.read_csv(OUT_DIR / "shap_train" / "shap_feature_importance.csv")
    row = imp[imp["feature"] == FEATURE].iloc[0]
    rank = int(row["rank"])

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X_train), min(PDP_N, len(X_train)), replace=False)
    X_pdp = X_train.iloc[idx].copy()

    grid = _manual_pdp_grid_from_series(X_pdp[FEATURE], GRID_POINTS)
    y = _manual_pdp_curve(pipe, X_pdp, FEATURE, grid)
    print(f"PCT grid points: {len(grid)}; y range {y.min():.4f}-{y.max():.4f}")

    disp = display_names("stage1", feats)[feats.index(FEATURE)]
    me = OUT_DIR / "marginal_effects"
    pdp_dir = me / "pdp"
    pdp_dir.mkdir(parents=True, exist_ok=True)
    out_jpg = pdp_dir / ranked_plot_filename(rank, FEATURE, "jpg")

    apply_pub_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid, y, color="#C44E52", linewidth=2, marker="o", markersize=3)
    ax.set_xlabel(disp)
    ax.set_ylabel("Predicted P(Disease=1)")
    ax.set_title(f"Partial dependence: {disp}")
    ax.grid(True, alpha=0.25)
    style_axes(ax)
    save_pub_figure(fig, out_jpg)

    new_rows = pd.DataFrame(
        {
            "feature": FEATURE,
            "grid": grid,
            "pdp_p_disease_1": y,
            "pdp_method": "manual_data_grid",
            "shap_rank": rank,
        }
    )
    curves = me / "pdp_curves.csv"
    if curves.is_file():
        old = pd.read_csv(curves)
        old = old[old["feature"] != FEATURE]
        out_df = pd.concat([old, new_rows], ignore_index=True)
    else:
        out_df = new_rows
    out_df.to_csv(curves, index=False, encoding="utf-8-sig")

    disp_map = dict(zip(feats, display_names("stage1", feats)))
    combined_tables = []
    for _, r in imp.iterrows():
        sub = out_df[out_df["feature"] == r["feature"]]
        if not sub.empty:
            combined_tables.append(sub)
    plot_pdp_combined(
        combined_tables,
        imp["feature"].tolist(),
        disp_map,
        me / "pdp_combined.jpg",
        f"Partial dependence (training subset n={len(X_pdp)})",
    )

    PUB_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(me / "pdp_combined.jpg", PUB_DIR / "pdp_combined.jpg")
    shutil.copy2(out_jpg, PUB_DIR / out_jpg.name)
    print(f"[OK] {out_jpg}")
    print(f"[OK] {PUB_DIR / out_jpg.name}")


if __name__ == "__main__":
    main()
