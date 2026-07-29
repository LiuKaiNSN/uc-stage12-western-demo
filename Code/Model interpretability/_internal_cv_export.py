# -*- coding: utf-8 -*-
"""Shared helpers for nested-CV internal validation table export (binary & multiclass)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

BINARY_METRIC_KEYS = [
    "auc",
    "auprc",
    "accuracy",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "f1",
]

BINARY_METRIC_LABELS = {
    "auc": "AUC",
    "auprc": "AUPRC",
    "accuracy": "Accuracy",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "ppv": "PPV",
    "npv": "NPV",
    "f1": "F1",
}

MULTICLASS_METRIC_KEYS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "macro_precision",
    "macro_recall",
    "macro_auc_ovr",
]

MULTICLASS_METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "macro_f1": "Macro F1",
    "weighted_f1": "Weighted F1",
    "macro_precision": "Macro precision",
    "macro_recall": "Macro recall",
    "macro_auc_ovr": "Macro AUC-OVR",
}


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else float("nan")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_num(v: float, digits: int = 3) -> str:
    if not np.isfinite(v):
        return ""
    return f"{v:.{digits}f}"


def _fmt_mean_sd(mean: float, sd: float, digits: int = 3) -> str:
    if not np.isfinite(mean):
        return ""
    if not np.isfinite(sd):
        return _fmt_num(mean, digits)
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def _fmt_point_ci(pt: float, lo: float, hi: float, digits: int = 3) -> str:
    if not np.isfinite(pt):
        return ""
    if not np.isfinite(lo) or not np.isfinite(hi):
        return _fmt_num(pt, digits)
    return f"{pt:.{digits}f} ({lo:.{digits}f}–{hi:.{digits}f})"


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    ppv = float(precision_score(y_true, y_pred, zero_division=0))
    return {
        "auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) >= 2 else float("nan"),
        "auprc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) >= 2 else float("nan"),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": _safe_div(tp, tp + fn),
        "specificity": _safe_div(tn, tn + fp),
        "ppv": ppv,
        "npv": _safe_div(tn, tn + fn),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def bootstrap_binary_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_boot: int,
    seed: int,
) -> Dict[str, Tuple[float, float]]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals: Dict[str, List[float]] = {k: [] for k in BINARY_METRIC_KEYS}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ys = y_true[idx]
        ps = y_prob[idx]
        if len(np.unique(ys)) < 2:
            continue
        try:
            m = compute_binary_metrics(ys, ps, threshold)
        except Exception:
            continue
        for k in BINARY_METRIC_KEYS:
            v = m.get(k, np.nan)
            if np.isfinite(v):
                vals[k].append(float(v))
    ci: Dict[str, Tuple[float, float]] = {}
    for k in BINARY_METRIC_KEYS:
        arr = np.array(vals[k], dtype=float)
        if arr.size == 0:
            ci[k] = (float("nan"), float("nan"))
        else:
            ci[k] = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
    return ci


def compute_multiclass_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = np.argmax(y_prob, axis=1)
    out: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    try:
        out["macro_auc_ovr"] = float(
            roc_auc_score(
                y_true,
                y_prob,
                multi_class="ovr",
                average="macro",
                labels=list(range(n_classes)),
            )
        )
    except Exception:
        out["macro_auc_ovr"] = float("nan")
    return out


def bootstrap_multiclass_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_classes: int,
    n_boot: int,
    seed: int,
) -> Dict[str, Tuple[float, float]]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals: Dict[str, List[float]] = {k: [] for k in MULTICLASS_METRIC_KEYS}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ys = y_true[idx]
        ps = y_prob[idx]
        if len(np.unique(ys)) < n_classes:
            continue
        try:
            m = compute_multiclass_metrics(ys, ps, n_classes)
        except Exception:
            continue
        for k in MULTICLASS_METRIC_KEYS:
            v = m.get(k, np.nan)
            if np.isfinite(v):
                vals[k].append(float(v))
    ci: Dict[str, Tuple[float, float]] = {}
    for k in MULTICLASS_METRIC_KEYS:
        arr = np.array(vals[k], dtype=float)
        if arr.size == 0:
            ci[k] = (float("nan"), float("nan"))
        else:
            ci[k] = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
    return ci


def _mean_sd_row(df: pd.DataFrame, metric_keys: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    fmt_row: Dict[str, Any] = {"outer_fold": "Mean ± SD"}
    numeric: Dict[str, Any] = {}
    for k in metric_keys:
        if k in df.columns:
            arr = df[k].astype(float)
            mean_v = float(arr.mean())
            sd_v = float(arr.std(ddof=1))
            fmt_row[k] = _fmt_mean_sd(mean_v, sd_v, 3)
            numeric[f"{k}_mean"] = mean_v
            numeric[f"{k}_sd"] = sd_v
    if "n_features" in df.columns:
        fmt_row["n_features"] = ""
    return fmt_row, numeric


def _load_n_features_per_fold(run_dir: Path) -> Dict[int, int]:
    path = run_dir / "fold_metrics.csv"
    if not path.is_file():
        return {}
    fm = pd.read_csv(path)
    if "outer_fold" not in fm.columns or "n_features" not in fm.columns:
        return {}
    return {int(r.outer_fold): int(r.n_features) for r in fm.itertuples(index=False)}


def _model_label(run_dir: Path) -> str:
    meta_path = run_dir / "final_model_meta.json"
    if meta_path.is_file():
        meta = _read_json(meta_path)
        model = meta.get("model")
        if model:
            return str(model)
    return run_dir.name


def export_binary_internal_cv(
    run_dir: Path,
    stage_label: str,
    out_dir: Path,
    threshold: float = 0.5,
    bootstrap_iter: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    run_dir = run_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    oof_path = run_dir / "oof_predictions.csv"
    if not oof_path.is_file():
        raise FileNotFoundError(f"Missing {oof_path}")

    oof = pd.read_csv(oof_path)
    required = {"y_true", "y_prob", "outer_fold"}
    if not required.issubset(oof.columns):
        raise ValueError(f"{oof_path} must contain columns: {sorted(required)}")

    n_feat_map = _load_n_features_per_fold(run_dir)
    fold_rows: List[Dict[str, Any]] = []
    for fold in sorted(oof["outer_fold"].unique()):
        sub = oof[oof["outer_fold"] == fold]
        m = compute_binary_metrics(sub["y_true"].values, sub["y_prob"].values, threshold)
        row: Dict[str, Any] = {"outer_fold": int(fold), "n_samples": len(sub)}
        if int(fold) in n_feat_map:
            row["n_features"] = n_feat_map[int(fold)]
        row.update(m)
        fold_rows.append(row)

    fold_df = pd.DataFrame(fold_rows)
    summary_fmt, summary_numeric = _mean_sd_row(fold_df, BINARY_METRIC_KEYS)
    fold_pub = fold_df.copy()
    for k in BINARY_METRIC_KEYS:
        fold_pub[k] = fold_pub[k].map(lambda v: _fmt_num(float(v), 3))
    fold_pub = pd.concat([fold_pub, pd.DataFrame([summary_fmt])], ignore_index=True)

    y_all = oof["y_true"].astype(int).values
    p_all = oof["y_prob"].astype(float).values
    oof_point = compute_binary_metrics(y_all, p_all, threshold)
    oof_ci = bootstrap_binary_ci(y_all, p_all, threshold, bootstrap_iter, seed)

    oof_long_rows: List[Dict[str, Any]] = []
    for k in BINARY_METRIC_KEYS:
        pt = oof_point[k]
        lo, hi = oof_ci[k]
        oof_long_rows.append(
            {
                "metric_key": k,
                "metric_label": BINARY_METRIC_LABELS[k],
                "point_estimate": pt,
                "ci_lower": lo,
                "ci_upper": hi,
                "formatted": _fmt_point_ci(pt, lo, hi, 3),
            }
        )
    oof_long = pd.DataFrame(oof_long_rows)

    prefix = f"internal_cv_{stage_label}"
    fold_df.to_csv(out_dir / f"{prefix}_by_fold.csv", index=False, encoding="utf-8-sig")
    fold_pub.to_csv(out_dir / f"{prefix}_by_fold_formatted.csv", index=False, encoding="utf-8-sig")
    oof_long.to_csv(out_dir / f"{prefix}_oof_with_ci.csv", index=False, encoding="utf-8-sig")

    mean_sd_df = pd.DataFrame(
        [
            {
                "metric_key": k,
                "metric_label": BINARY_METRIC_LABELS[k],
                "mean": summary_numeric.get(f"{k}_mean"),
                "sd": summary_numeric.get(f"{k}_sd"),
                "formatted_mean_sd": summary_fmt.get(k, ""),
            }
            for k in BINARY_METRIC_KEYS
        ]
    )
    mean_sd_df.to_csv(out_dir / f"{prefix}_fold_mean_sd.csv", index=False, encoding="utf-8-sig")

    stage_a = _read_json(run_dir / "stage_a_meta.json") if (run_dir / "stage_a_meta.json").is_file() else {}
    cv_summary = _read_json(run_dir / "cv_summary.json") if (run_dir / "cv_summary.json").is_file() else {}

    manifest: Dict[str, Any] = {
        "task": "binary",
        "stage_label": stage_label,
        "model_run_dir": str(run_dir),
        "model": _model_label(run_dir),
        "threshold": threshold,
        "bootstrap_iter": bootstrap_iter,
        "seed": seed,
        "n_oof_samples": int(len(oof)),
        "outer_folds": int(stage_a.get("outer_folds", len(fold_df))),
        "inner_folds": stage_a.get("inner_folds"),
        "stage_a_meta": stage_a,
        "cv_summary_json": cv_summary,
        "output_files": {
            "by_fold": str(out_dir / f"{prefix}_by_fold.csv"),
            "by_fold_formatted": str(out_dir / f"{prefix}_by_fold_formatted.csv"),
            "fold_mean_sd": str(out_dir / f"{prefix}_fold_mean_sd.csv"),
            "oof_with_ci": str(out_dir / f"{prefix}_oof_with_ci.csv"),
        },
    }
    manifest_path = out_dir / f"{prefix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def export_multiclass_internal_cv(
    run_dir: Path,
    stage_label: str,
    out_dir: Path,
    bootstrap_iter: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    run_dir = run_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    oof_path = run_dir / "oof_predictions.csv"
    if not oof_path.is_file():
        raise FileNotFoundError(f"Missing {oof_path}")

    oof = pd.read_csv(oof_path)
    prob_cols = sorted([c for c in oof.columns if c.startswith("prob_")], key=lambda x: int(x.split("_", 1)[1]))
    if not prob_cols:
        raise ValueError(f"{oof_path} must contain prob_* columns for multiclass OOF")
    if "y_true" not in oof.columns or "outer_fold" not in oof.columns:
        raise ValueError(f"{oof_path} must contain y_true and outer_fold")

    n_classes = len(prob_cols)
    n_feat_map = _load_n_features_per_fold(run_dir)

    fold_rows: List[Dict[str, Any]] = []
    for fold in sorted(oof["outer_fold"].unique()):
        sub = oof[oof["outer_fold"] == fold]
        y = sub["y_true"].values
        p = sub[prob_cols].values.astype(float)
        m = compute_multiclass_metrics(y, p, n_classes)
        row: Dict[str, Any] = {"outer_fold": int(fold), "n_samples": len(sub)}
        if int(fold) in n_feat_map:
            row["n_features"] = n_feat_map[int(fold)]
        row.update(m)
        fold_rows.append(row)

    fold_df = pd.DataFrame(fold_rows)
    summary_fmt, summary_numeric = _mean_sd_row(fold_df, MULTICLASS_METRIC_KEYS)
    fold_pub = fold_df.copy()
    for k in MULTICLASS_METRIC_KEYS:
        fold_pub[k] = fold_pub[k].map(lambda v: _fmt_num(float(v), 3))
    fold_pub = pd.concat([fold_pub, pd.DataFrame([summary_fmt])], ignore_index=True)

    y_all = oof["y_true"].astype(int).values
    p_all = oof[prob_cols].values.astype(float)
    oof_point = compute_multiclass_metrics(y_all, p_all, n_classes)
    oof_ci = bootstrap_multiclass_ci(y_all, p_all, n_classes, bootstrap_iter, seed)

    oof_long_rows: List[Dict[str, Any]] = []
    for k in MULTICLASS_METRIC_KEYS:
        pt = oof_point[k]
        lo, hi = oof_ci[k]
        oof_long_rows.append(
            {
                "metric_key": k,
                "metric_label": MULTICLASS_METRIC_LABELS[k],
                "point_estimate": pt,
                "ci_lower": lo,
                "ci_upper": hi,
                "formatted": _fmt_point_ci(pt, lo, hi, 3),
            }
        )
    oof_long = pd.DataFrame(oof_long_rows)

    prefix = f"internal_cv_{stage_label}"
    fold_df.to_csv(out_dir / f"{prefix}_by_fold.csv", index=False, encoding="utf-8-sig")
    fold_pub.to_csv(out_dir / f"{prefix}_by_fold_formatted.csv", index=False, encoding="utf-8-sig")
    oof_long.to_csv(out_dir / f"{prefix}_oof_with_ci.csv", index=False, encoding="utf-8-sig")

    mean_sd_df = pd.DataFrame(
        [
            {
                "metric_key": k,
                "metric_label": MULTICLASS_METRIC_LABELS[k],
                "mean": summary_numeric.get(f"{k}_mean"),
                "sd": summary_numeric.get(f"{k}_sd"),
                "formatted_mean_sd": summary_fmt.get(k, ""),
            }
            for k in MULTICLASS_METRIC_KEYS
        ]
    )
    mean_sd_df.to_csv(out_dir / f"{prefix}_fold_mean_sd.csv", index=False, encoding="utf-8-sig")

    stage_a = _read_json(run_dir / "stage_a_meta.json") if (run_dir / "stage_a_meta.json").is_file() else {}
    cv_summary = _read_json(run_dir / "cv_summary.json") if (run_dir / "cv_summary.json").is_file() else {}
    final_meta = _read_json(run_dir / "final_model_meta.json") if (run_dir / "final_model_meta.json").is_file() else {}

    manifest: Dict[str, Any] = {
        "task": "multiclass",
        "stage_label": stage_label,
        "model_run_dir": str(run_dir),
        "model": _model_label(run_dir),
        "class_names": final_meta.get("class_names") or stage_a.get("class_names"),
        "bootstrap_iter": bootstrap_iter,
        "seed": seed,
        "n_oof_samples": int(len(oof)),
        "outer_folds": int(stage_a.get("outer_folds", len(fold_df))),
        "inner_folds": stage_a.get("inner_folds"),
        "stage_a_meta": stage_a,
        "cv_summary_json": cv_summary,
        "output_files": {
            "by_fold": str(out_dir / f"{prefix}_by_fold.csv"),
            "by_fold_formatted": str(out_dir / f"{prefix}_by_fold_formatted.csv"),
            "fold_mean_sd": str(out_dir / f"{prefix}_fold_mean_sd.csv"),
            "oof_with_ci": str(out_dir / f"{prefix}_oof_with_ci.csv"),
        },
    }
    manifest_path = out_dir / f"{prefix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
