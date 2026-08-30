# -*- coding: utf-8 -*-
"""
Stage1 / Stage2 binary best-model interpretability (tree ML only).

Full-sample TreeExplainer SHAP on training + external; publication figures:
  - SHAP importance + correlation composite
  - Combined beeswarm
  - Combined dependence (no interaction coloring)
  - Full-feature PDP (+ optional combined panel)

Data (Western Stage12): TCM/data/ATrain-Stage1.csv / ATrain-Stage2.csv

PowerShell::

  python "F:\\KeTi\\Project\\Script\\interpret_binary_best_model_Stage12.py" `
    --model-run-dir "F:\\KeTi\\Project\\outputs\\Stage1\\3_run_20260707_142952" `
    --train-csv "F:\\KeTi\\Project\\TCM\\data\\ATrain-Stage1.csv" `
    --external-csv "F:\\KeTi\\Project\\TCM\\data\\ATest-Stage1.csv" `
    --stage-label stage1 `
    --plot-format jpg `
    --pub-figure-dir "F:\\KeTi\\Project\\Figure\\Stage12\\figure5\\stage1"
"""

from __future__ import annotations

import argparse
import json
import shutil
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence
from sklearn.pipeline import Pipeline

try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    shap = None  # type: ignore

from external_validate_stage1 import build_feature_matrix
from _interpret_feature_labels import display_names
from _pub_plot_style import apply_pub_style, save_pub_figure, style_axes
from _shap_composite_plot import plot_shap_importance_correlation

TREE_STEP_NAMES = frozenset({"rf", "xgb", "lgbm", "cat"})
POSITIVE_CLASS_INDEX = 1
PUB_PLOT_NAMES = (
    "shap_importance_correlation",
    "shap_beeswarm_combined",
    "shap_dependence_combined",
    "pdp_combined",
)


def _safe_feature_token(feature: str) -> str:
    token = "".join(c if c.isalnum() or c in "._-" else "_" for c in feature.strip())
    return token or "feature"


def ranked_plot_filename(rank: int, feature: str, plot_ext: str) -> str:
    return f"rank{int(rank):02d}_{_safe_feature_token(feature)}.{plot_ext}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage1/Stage2 binary tree-model: full SHAP + marginal effects (publication JPG)"
    )
    p.add_argument("--model-run-dir", type=str, required=True)
    p.add_argument("--train-csv", type=str, required=True)
    p.add_argument("--external-csv", type=str, required=True)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--stage-label", type=str, default="stage2", choices=["stage1", "stage2"])
    p.add_argument("--id-col", type=str, default="No")
    p.add_argument("--label-col", type=str, default="Disease")
    p.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="PDP features: 0 = all locked features (default); else top-K by training SHAP",
    )
    p.add_argument("--grid-points", type=int, default=30, help="PDP grid resolution (default 30)")
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--plot-format", type=str, default="jpg", choices=["jpg", "jpeg", "png", "pdf"])
    p.add_argument(
        "--pub-figure-dir",
        type=str,
        default=None,
        help="Copy publication JPGs here (e.g. Figure/Stage12/figure5/stage1)",
    )
    p.add_argument(
        "--plot-max-samples",
        type=int,
        default=400,
        help="Max samples for beeswarm/dependence plots (importance uses all samples)",
    )
    p.add_argument(
        "--save-individual-pdp",
        action="store_true",
        help="Also save per-feature PDP JPGs (slower; default is combined panel only)",
    )
    p.add_argument("--save-shap-matrix", action="store_true")
    p.add_argument(
        "--pdp-max-samples",
        type=int,
        default=400,
        help="Subsample training rows for brute PDP only (SHAP still uses all samples)",
    )
    p.add_argument(
        "--skip-pdp",
        action="store_true",
        help="Skip PDP (SHAP figures only; faster smoke run)",
    )
    p.add_argument(
        "--pdp-only",
        action="store_true",
        help="Only (re)compute PDP; reuse existing shap_train/shap_feature_importance.csv",
    )
    p.add_argument(
        "--pdp-features",
        type=str,
        default=None,
        help="Comma-separated PDP features only (e.g. PCT); merges into existing pdp_curves.csv",
    )
    p.add_argument(
        "--pdp-manual-only",
        action="store_true",
        help="Skip sklearn brute PDP; use manual data grid only (fast repair)",
    )
    p.add_argument(
        "--composite-only",
        action="store_true",
        help="Only redraw SHAP importance+correlation composite (no SHAP/PDP; fast)",
    )
    p.add_argument(
        "--composite-bar-ymax-train",
        type=float,
        default=None,
        help="Fixed bar-chart ymax for train SHAP composite only (e.g. 1.2)",
    )
    p.add_argument(
        "--composite-bar-ymax-external",
        type=float,
        default=None,
        help="Fixed bar-chart ymax for external SHAP composite only (e.g. 1.2)",
    )
    return p.parse_args()


def _plot_ext(args: argparse.Namespace) -> str:
    ext = args.plot_format.lower()
    return "jpg" if ext == "jpeg" else ext


def _load_pipeline(model_path: Path) -> Pipeline:
    obj = joblib.load(model_path)
    if isinstance(obj, dict):
        raise TypeError("Deep-learning bundle not supported; use tree ML Pipeline only.")
    if not isinstance(obj, Pipeline):
        raise TypeError(f"Expected sklearn Pipeline, got {type(obj).__name__}")
    if len(obj.steps) < 2:
        raise ValueError("Pipeline must contain imputer + tree classifier steps.")
    tree_name = obj.steps[-1][0]
    if tree_name not in TREE_STEP_NAMES:
        raise ValueError(f"Unsupported final step {tree_name!r}.")
    return obj


def _transform_imputed(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(pipe.named_steps["imputer"].transform(X), dtype=float)


def _tree_estimator(pipe: Pipeline) -> Any:
    return pipe.steps[-1][1]


def _extract_positive_shap(shap_values: Any) -> np.ndarray:
    if isinstance(shap_values, list):
        if len(shap_values) == 1:
            return np.asarray(shap_values[0], dtype=float)
        return np.asarray(shap_values[POSITIVE_CLASS_INDEX], dtype=float)
    arr = np.asarray(shap_values, dtype=float)
    if arr.ndim == 3:
        return arr[:, :, POSITIVE_CLASS_INDEX]
    return arr


def compute_tree_shap(
    pipe: Pipeline,
    X: pd.DataFrame,
    feature_names: List[str],
) -> Tuple[np.ndarray, pd.DataFrame]:
    if not HAS_SHAP:
        raise RuntimeError("Package 'shap' is required.")

    X_imp = _transform_imputed(pipe, X)
    tree = _tree_estimator(pipe)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.TreeExplainer(tree)
        raw = explainer.shap_values(X_imp)
    sv = _extract_positive_shap(raw)
    if sv.shape != (len(X), len(feature_names)):
        raise RuntimeError(f"SHAP shape {sv.shape} != ({len(X)}, {len(feature_names)})")

    mean_abs = np.mean(np.abs(sv), axis=0)
    imp = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp["rank"] = np.arange(1, len(imp) + 1)
    imp["n_samples"] = len(X)
    imp["shap_method"] = "TreeExplainer"
    imp["target"] = "P(Disease=1)"
    return sv, imp


def _subsample_for_plot(
    sv: np.ndarray,
    X_imp: np.ndarray,
    max_n: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(sv) <= max_n:
        return sv, X_imp
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(sv), size=max_n, replace=False)
    return sv[idx], X_imp[idx]


def _feature_order(imp: pd.DataFrame) -> List[str]:
    return imp.sort_values("mean_abs_shap", ascending=False)["feature"].tolist()


def _reorder_shap(sv: np.ndarray, feature_names: List[str], order: List[str]) -> np.ndarray:
    idx = [feature_names.index(f) for f in order]
    return sv[:, idx]


def pearson_corr_matrix(X_imp: np.ndarray) -> np.ndarray:
    if X_imp.shape[0] < 2:
        return np.eye(X_imp.shape[1])
    return np.corrcoef(X_imp, rowvar=False)


def plot_beeswarm_combined(
    sv: np.ndarray,
    X_imp: np.ndarray,
    display_labels: List[str],
    out_path: Path,
    title: str,
    max_samples: int,
    seed: int,
) -> None:
    if not HAS_SHAP:
        return
    sv_p, x_p = _subsample_for_plot(sv, X_imp, max_samples, seed)
    apply_pub_style()
    plt.figure()
    shap.summary_plot(
        sv_p,
        x_p,
        feature_names=display_labels,
        max_display=len(display_labels),
        show=False,
        plot_size=(10, max(6, 0.38 * len(display_labels))),
    )
    fig = plt.gcf()
    note = f"n={len(sv)}" if len(sv) != len(sv_p) else f"n={len(sv)}"
    fig.suptitle(f"{title} ({note})", y=1.02, fontsize=12)
    save_pub_figure(fig, out_path)


def plot_dependence_combined(
    sv: np.ndarray,
    X_imp: np.ndarray,
    order: List[str],
    feature_names: List[str],
    display_labels: List[str],
    out_path: Path,
    title: str,
    max_samples: int,
    seed: int,
) -> None:
    sv_p, x_p = _subsample_for_plot(sv, X_imp, max_samples, seed + 1)
    n = len(order)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.8), squeeze=False)
    for k, feat in enumerate(order):
        ax = axes[k // ncols][k % ncols]
        label = display_labels[k]
        ax.scatter(
            x_p[:, k],
            sv_p[:, k],
            c="#2166AC",
            alpha=0.35,
            s=10,
            edgecolors="none",
        )
        ax.set_xlabel(label, fontsize=8)
        ax.set_ylabel("SHAP", fontsize=8)
        ax.tick_params(labelsize=7)
        style_axes(ax)
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle(
        f"{title} (n={len(sv)}, plot n={len(sv_p)})" if len(sv) != len(sv_p) else title,
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    save_pub_figure(fig, out_path)


def _manual_pdp_grid_from_series(series: pd.Series, grid_points: int) -> np.ndarray:
    """Grid from observed training values (quantiles or unique), for sparse labs like PCT."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return np.linspace(0.0, 1.0, max(grid_points, 2))
    vals = np.asarray(s.values, dtype=float)
    uniq = np.unique(vals)
    if len(uniq) == 1:
        return uniq
    if len(uniq) <= grid_points:
        return uniq
    qs = np.linspace(0.05, 0.95, grid_points)
    return np.unique(np.quantile(uniq, qs))


def _manual_pdp_curve(pipe: Pipeline, X: pd.DataFrame, feature: str, grid: np.ndarray) -> np.ndarray:
    X_work = X.copy()
    probs: List[float] = []
    for g in grid:
        X_work[feature] = float(g)
        p = pipe.predict_proba(X_work)[:, POSITIVE_CLASS_INDEX]
        probs.append(float(np.mean(p)))
    return np.asarray(probs, dtype=float)


def run_pdp(
    pipe: Pipeline,
    X: pd.DataFrame,
    feature: str,
    feature_names: List[str],
    grid_points: int,
    out_path: Path | None,
    display_label: str,
    manual_only: bool = False,
) -> pd.DataFrame:
    pdp_method = "manual_data_grid"
    if manual_only:
        grid = _manual_pdp_grid_from_series(X[feature], grid_points)
        y = _manual_pdp_curve(pipe, X, feature, grid)
    else:
        Xf = X.astype(float)
        feat_idx = feature_names.index(feature)
        pdp_method = "sklearn_brute"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pd_res = partial_dependence(
                pipe,
                Xf,
                features=[feat_idx],
                grid_resolution=grid_points,
                response_method="predict_proba",
                method="brute",
            )
        grid = np.asarray(pd_res["grid_values"][0], dtype=float)
        avg = np.asarray(pd_res["average"], dtype=float)
        if avg.ndim == 3:
            y = avg[POSITIVE_CLASS_INDEX, 0, :]
        elif avg.ndim == 2:
            y = avg[0, :] if avg.shape[0] == 1 else avg[POSITIVE_CLASS_INDEX, :]
        else:
            y = avg.ravel()

        if not np.all(np.isfinite(grid)) or len(grid) == 0 or not np.any(np.isfinite(y)):
            print(f"    [PDP] sklearn brute grid invalid for {feature}; using manual data grid")
            grid = _manual_pdp_grid_from_series(X[feature], grid_points)
            y = _manual_pdp_curve(pipe, X, feature, grid)
            pdp_method = "manual_data_grid"

    if out_path is not None:
        apply_pub_style()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(grid, y, color="#C44E52", linewidth=2, marker="o", markersize=3)
        ax.set_xlabel(display_label)
        ax.set_ylabel("Predicted P(Disease=1)")
        ax.set_title(f"Partial dependence: {display_label}")
        ax.grid(True, alpha=0.25)
        style_axes(ax)
        save_pub_figure(fig, out_path)
    return pd.DataFrame(
        {
            "feature": feature,
            "grid": grid,
            "pdp_p_disease_1": y,
            "pdp_method": pdp_method,
        }
    )


def plot_pdp_combined(
    pdp_tables: List[pd.DataFrame],
    order: List[str],
    display_map: Dict[str, str],
    out_path: Path,
    title: str,
) -> None:
    apply_pub_style()
    n = len(order)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.8), squeeze=False)
    table_map = {t["feature"].iloc[0]: t for t in pdp_tables}
    for k, feat in enumerate(order):
        ax = axes[k // ncols][k % ncols]
        t = table_map[feat]
        gx = pd.to_numeric(t["grid"], errors="coerce").to_numpy(dtype=float)
        gy = pd.to_numeric(t["pdp_p_disease_1"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(gx) & np.isfinite(gy)
        if ok.any():
            ax.plot(gx[ok], gy[ok], color="#C44E52", linewidth=2, marker="o", markersize=2)
        ax.set_xlabel(display_map.get(feat, feat), fontsize=8)
        ax.set_ylabel("P(Disease=1)", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)
        style_axes(ax)
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle(title, fontsize=12, y=1.01)
    fig.tight_layout()
    save_pub_figure(fig, out_path)


def _write_shap_bundle(
    split_name: str,
    out_root: Path,
    sv: np.ndarray | None,
    imp: pd.DataFrame,
    pipe: Pipeline,
    X: pd.DataFrame,
    feature_names: List[str],
    stage_label: str,
    corr_train: np.ndarray,
    ext: str,
    plot_max_samples: int,
    seed: int,
    composite_only: bool = False,
    composite_bar_ymax: float | None = None,
) -> None:
    root = out_root / f"shap_{split_name}"
    root.mkdir(parents=True, exist_ok=True)

    order = _feature_order(imp)
    disp_all = display_names(stage_label, feature_names)
    disp_map = dict(zip(feature_names, disp_all))
    disp_ordered = [disp_map[f] for f in order]

    if not composite_only:
        imp.to_csv(root / "shap_feature_importance.csv", index=False, encoding="utf-8-sig")
    print(f"    [{split_name}] composite SHAP+corr plot...")

    order_idx = [feature_names.index(f) for f in order]
    imp_ordered = imp.set_index("feature").reindex(order).reset_index()
    corr_ordered = corr_train[np.ix_(order_idx, order_idx)]

    plot_shap_importance_correlation(
        imp_ordered,
        corr_ordered,
        root / f"shap_importance_correlation.{ext}",
        title=f"SHAP importance & feature correlation ({split_name}, n={len(X)})",
        feature_label_map=disp_map,
        bar_ymax=composite_bar_ymax,
    )
    if composite_only:
        print(f"    [{split_name}] composite done.")
        return

    print(f"    [{split_name}] beeswarm...")

    X_imp = _transform_imputed(pipe, X)
    sv_ord = _reorder_shap(sv, feature_names, order)
    plot_beeswarm_combined(
        sv_ord,
        X_imp[:, order_idx],
        disp_ordered,
        root / f"shap_beeswarm_combined.{ext}",
        f"SHAP beeswarm ({split_name})",
        plot_max_samples,
        seed,
    )
    plot_dependence_combined(
        sv_ord,
        X_imp[:, order_idx],
        order,
        feature_names,
        disp_ordered,
        root / f"shap_dependence_combined.{ext}",
        f"SHAP dependence ({split_name})",
        plot_max_samples,
        seed,
    )
    print(f"    [{split_name}] dependence + plots done.")


def _composite_bar_ymax(split_name: str, args: argparse.Namespace) -> float | None:
    if split_name == "train" and args.composite_bar_ymax_train is not None:
        return float(args.composite_bar_ymax_train)
    if split_name == "external" and args.composite_bar_ymax_external is not None:
        return float(args.composite_bar_ymax_external)
    return None


def _copy_pub_figures(shap_dirs: List[Path], me_dir: Path, pub_dir: Path, ext: str) -> None:
    pub_dir.mkdir(parents=True, exist_ok=True)

    def _safe_copy(src: Path, dst: Path) -> None:
        """Copy with temp+replace so a locked destination JPG is easier to overwrite on Windows."""
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".tmpcopy")
        try:
            if tmp.exists():
                tmp.unlink()
            shutil.copy2(src, tmp)
            try:
                tmp.replace(dst)
            except OSError:
                # Fallback if replace fails while dst is open: try direct copy2.
                shutil.copy2(src, dst)
                try:
                    tmp.unlink()
                except OSError:
                    pass
            print(f"    [copied] {dst.name}")
        except OSError as exc:
            print(f"    [WARN] could not copy {src.name} -> {dst.name}: {exc}")
            print("           Close the JPG in any image viewer / Explorer preview, then re-run.")
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass

    for sd in shap_dirs:
        prefix = sd.name.replace("shap_", "")
        for stem in ("shap_importance_correlation", "shap_beeswarm_combined", "shap_dependence_combined"):
            src = sd / f"{stem}.{ext}"
            if not src.is_file():
                print(f"    [skip copy] missing {src}")
                continue
            _safe_copy(src, pub_dir / f"{prefix}_{stem}.jpg")
    pdp_combined = me_dir / f"pdp_combined.{ext}"
    if pdp_combined.is_file():
        _safe_copy(pdp_combined, pub_dir / "pdp_combined.jpg")
    pdp_sub = me_dir / "pdp"
    if pdp_sub.is_dir():
        copied = 0
        for jpg in pdp_sub.glob(f"*.{ext}"):
            before = (pub_dir / jpg.name).is_file()
            _safe_copy(jpg, pub_dir / jpg.name)
            if (pub_dir / jpg.name).is_file():
                copied += 1
        if copied:
            print(f"    Copied {copied} individual PDP file(s) -> {pub_dir}")


def _read_features(feats_path: Path) -> List[str]:
    return [x.strip() for x in feats_path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _load_X(
    csv_path: Path, feature_names: List[str], id_col: str, label_col: str
) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(csv_path)
    X, missing = build_feature_matrix(df, feature_names)
    if missing:
        print(
            f"    [note] {csv_path.name}: {len(missing)} feature(s) missing -> NaN: "
            f"{missing[:8]}{'...' if len(missing) > 8 else ''}"
        )
    return X, missing


def main() -> None:
    args = parse_args()
    run_dir = Path(args.model_run_dir).resolve()
    train_csv = Path(args.train_csv).resolve()
    external_csv = Path(args.external_csv).resolve()
    out_dir = Path(args.output_dir).resolve() if args.output_dir else (run_dir / "interpretability")
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = _plot_ext(args)

    model_path = run_dir / "final_model.joblib"
    feats_path = run_dir / "final_features_used.txt"
    if not model_path.exists() or not feats_path.exists():
        raise FileNotFoundError(f"Missing {model_path} or {feats_path}")

    pipe = _load_pipeline(model_path)
    feature_names = _read_features(feats_path)
    tree_step = pipe.steps[-1][0]
    n_feat = len(feature_names)
    top_k = n_feat if args.top_k <= 0 else min(args.top_k, n_feat)

    X_train, _ = _load_X(train_csv, feature_names, args.id_col, args.label_col)
    X_ext, _ = _load_X(external_csv, feature_names, args.id_col, args.label_col)

    shap_train_dir = out_dir / "shap_train"
    shap_ext_dir = out_dir / "shap_external"

    if args.composite_only:
        imp_path = shap_train_dir / "shap_feature_importance.csv"
        ext_imp_path = shap_ext_dir / "shap_feature_importance.csv"
        if not imp_path.is_file() or not ext_imp_path.is_file():
            raise FileNotFoundError(
                f"--composite-only requires {imp_path} and {ext_imp_path}; run full interpret first"
            )
        X_train_imp = _transform_imputed(pipe, X_train)
        corr_train = pearson_corr_matrix(X_train_imp)
        imp_train = pd.read_csv(imp_path)
        imp_ext = pd.read_csv(ext_imp_path)
        print(">>> Composite-only mode (redraw importance+correlation grid plots)")
        print(f"    Will write train + external under: {out_dir}")
        if args.pub_figure_dir:
            print(f"    Then copy BOTH to: {Path(args.pub_figure_dir).resolve()}")
        _write_shap_bundle(
            "train", out_dir, None, imp_train, pipe, X_train, feature_names,
            args.stage_label, corr_train, ext, args.plot_max_samples, args.random_seed,
            composite_only=True,
            composite_bar_ymax=_composite_bar_ymax("train", args),
        )
        print(f"    [written] {shap_train_dir / f'shap_importance_correlation.{ext}'}")
        _write_shap_bundle(
            "external", out_dir, None, imp_ext, pipe, X_ext, feature_names,
            args.stage_label, corr_train, ext, args.plot_max_samples, args.random_seed + 17,
            composite_only=True,
            composite_bar_ymax=_composite_bar_ymax("external", args),
        )
        print(f"    [written] {shap_ext_dir / f'shap_importance_correlation.{ext}'}")
        if args.pub_figure_dir:
            pub = Path(args.pub_figure_dir).resolve()
            pub.mkdir(parents=True, exist_ok=True)
            # Composite-only: copy ONLY the redrawn importance composites.
            # (Copying beeswarm/dependence/PDP here was unnecessary and could abort
            #  the copy loop on Windows before external_* was updated.)
            for split, sdir in (("train", shap_train_dir), ("external", shap_ext_dir)):
                src = sdir / f"shap_importance_correlation.{ext}"
                dst = pub / f"{split}_shap_importance_correlation.jpg"
                if not src.is_file():
                    print(f"    [WARN] missing source {src}")
                    continue
                tmp = dst.with_name(dst.name + ".tmpcopy")
                try:
                    if tmp.exists():
                        tmp.unlink()
                    shutil.copy2(src, tmp)
                    try:
                        tmp.replace(dst)
                    except OSError:
                        shutil.copy2(src, dst)
                        if tmp.exists():
                            tmp.unlink()
                    print(f"    [copied] {dst.name} size={dst.stat().st_size}")
                except Exception as exc:
                    print(f"    [WARN] could not copy {src.name} -> {dst.name}: {exc}")
                    print("           Close that JPG if it is open, then re-run.")
                    try:
                        if tmp.exists():
                            tmp.unlink()
                    except OSError:
                        pass
            train_pub = pub / "train_shap_importance_correlation.jpg"
            ext_pub = pub / "external_shap_importance_correlation.jpg"
            print(f"    Check pub train exists={train_pub.is_file()}  external exists={ext_pub.is_file()}")
            if train_pub.is_file():
                print(f"    train pub mtime={train_pub.stat().st_mtime} size={train_pub.stat().st_size}")
            if ext_pub.is_file():
                print(f"    external pub mtime={ext_pub.stat().st_mtime} size={ext_pub.stat().st_size}")
            print(f"    Publication copies -> {pub}")
        print(f"[OK] Composite plots under {out_dir}")
        return

    if args.pdp_only:
        imp_path = shap_train_dir / "shap_feature_importance.csv"
        if not imp_path.is_file():
            raise FileNotFoundError(
                f"--pdp-only requires existing {imp_path}; run full interpret first or remove --pdp-only"
            )
        imp_train = pd.read_csv(imp_path)
        print(">>> PDP-only mode (skipping SHAP; using existing importance table)")
    else:
        X_train_imp = _transform_imputed(pipe, X_train)
        corr_train = pearson_corr_matrix(X_train_imp)

        print(">>> SHAP (TreeExplainer, full samples, P(Disease=1))")
        print("    computing SHAP on training set...")
        sv_train, imp_train = compute_tree_shap(pipe, X_train, feature_names)
        print("    computing SHAP on external set...")
        sv_ext, imp_ext = compute_tree_shap(pipe, X_ext, feature_names)

        _write_shap_bundle(
            "train", out_dir, sv_train, imp_train, pipe, X_train, feature_names,
            args.stage_label, corr_train, ext, args.plot_max_samples, args.random_seed,
            composite_bar_ymax=_composite_bar_ymax("train", args),
        )
        _write_shap_bundle(
            "external", out_dir, sv_ext, imp_ext, pipe, X_ext, feature_names,
            args.stage_label, corr_train, ext, args.plot_max_samples, args.random_seed + 17,
            composite_bar_ymax=_composite_bar_ymax("external", args),
        )

        if args.save_shap_matrix:
            pd.DataFrame(sv_train, columns=feature_names).to_csv(
                shap_train_dir / "shap_values_matrix.csv", index=False, encoding="utf-8-sig"
            )
            pd.DataFrame(sv_ext, columns=feature_names).to_csv(
                shap_ext_dir / "shap_values_matrix.csv", index=False, encoding="utf-8-sig"
            )

    top_rows = imp_train.head(top_k).copy()
    if args.pdp_features:
        want = {x.strip() for x in args.pdp_features.split(",") if x.strip()}
        top_rows = imp_train[imp_train["feature"].isin(want)].copy()
        if top_rows.empty:
            raise ValueError(f"--pdp-features matched nothing in importance table: {want}")
        print(f"    PDP subset: {top_rows['feature'].tolist()}")

    top_rows[["rank", "feature", "mean_abs_shap"]].to_csv(
        out_dir / "top_features_for_pdp.csv", index=False, encoding="utf-8-sig"
    )
    order = _feature_order(imp_train)
    disp_map = dict(zip(feature_names, display_names(args.stage_label, feature_names)))

    me_dir = out_dir / "marginal_effects"
    pdp_dir = me_dir / "pdp"
    pdp_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_pdp:
        pdp_n = min(len(X_train), args.pdp_max_samples)
        if pdp_n < len(X_train):
            rng = np.random.default_rng(args.random_seed)
            pdp_idx = rng.choice(len(X_train), size=pdp_n, replace=False)
            X_pdp = X_train.iloc[pdp_idx].copy()
            print(f"    PDP: {top_k} features, grid={args.grid_points}, n={pdp_n} (subsampled for brute PDP)")
        else:
            X_pdp = X_train
            print(f"    PDP: {top_k} features, grid={args.grid_points}, n={len(X_train)}")

        pdp_tables: List[pd.DataFrame] = []
        for _, row in top_rows.iterrows():
            rank = int(row["rank"])
            feat = str(row["feature"])
            plot_label = disp_map[feat]
            print(f"    PDP rank{rank:02d} {feat} (label: {plot_label})...")
            pdp_out = None
            if args.save_individual_pdp:
                pdp_out = pdp_dir / ranked_plot_filename(rank, feat, ext)
            pdp_df = run_pdp(
                pipe,
                X_pdp,
                feat,
                feature_names,
                args.grid_points,
                pdp_out,
                plot_label,
                manual_only=args.pdp_manual_only,
            )
            pdp_df["shap_rank"] = rank
            pdp_tables.append(pdp_df)

        curves_path = me_dir / "pdp_curves.csv"
        if args.pdp_features and curves_path.is_file():
            old = pd.read_csv(curves_path)
            old = old[~old["feature"].isin(top_rows["feature"].tolist())]
            out_df = pd.concat([old, pd.concat(pdp_tables, ignore_index=True)], ignore_index=True)
        else:
            out_df = pd.concat(pdp_tables, ignore_index=True)
        out_df.to_csv(curves_path, index=False, encoding="utf-8-sig")

        combined_tables: List[pd.DataFrame] = []
        for _, row in imp_train.iterrows():
            sub = out_df[out_df["feature"] == row["feature"]]
            if not sub.empty:
                combined_tables.append(sub)
        plot_pdp_combined(
            combined_tables,
            imp_train["feature"].tolist(),
            disp_map,
            me_dir / f"pdp_combined.{ext}",
            f"Partial dependence (training subset n={len(X_pdp)})",
        )
    else:
        print("    PDP skipped (--skip-pdp)")

    if args.pub_figure_dir:
        pub = Path(args.pub_figure_dir).resolve()
        _copy_pub_figures([shap_train_dir, shap_ext_dir], me_dir, pub, ext)
        print(f"    Publication copies -> {pub}")

    manifest: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stage_label": args.stage_label,
        "model_run_dir": str(run_dir),
        "train_csv": str(train_csv),
        "external_csv": str(external_csv),
        "output_dir": str(out_dir),
        "pub_figure_dir": str(Path(args.pub_figure_dir).resolve()) if args.pub_figure_dir else None,
        "tree_step": tree_step,
        "n_features": n_feat,
        "feature_names": feature_names,
        "n_train": int(len(X_train)),
        "n_external": int(len(X_ext)),
        "shap_method": "TreeExplainer",
        "shap_target": "P(Disease=1)",
        "correlation_matrix_source": "training_set_imputed",
        "pdp_features": top_k,
        "plot_format": ext,
        "grid_points": args.grid_points,
        "random_seed": args.random_seed,
        "plot_max_samples": args.plot_max_samples,
        "pdp_max_samples": args.pdp_max_samples,
        "skip_pdp": args.skip_pdp,
        "pdp_only": args.pdp_only,
        "pdp_features": args.pdp_features,
        "pdp_manual_only": args.pdp_manual_only,
    }
    locked = run_dir / "locked_params.json"
    if locked.exists():
        manifest["locked_params"] = json.loads(locked.read_text(encoding="utf-8"))

    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(">>> Done")
    print(f"    Output: {out_dir}")


if __name__ == "__main__":
    main()
