# -*- coding: utf-8 -*-
"""
训练阶段结果汇总脚本（10模型）
--------------------------------
目标：
1) 读取 outputs 下 10 个模型目录，汇总参数表、变量表、内部性能表；
2) 基于各模型 oof_predictions.csv 计算补充性能与 95%CI；
3) 基于 Stage B final_model.joblib 计算 SHAP 并汇总。

说明：
- 目录映射按用户约定：前缀序号 1..10 -> 模型名
- 结果写入 outputs 下新建 summary_时间戳 文件夹
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    import shap

    HAS_SHAP = True
except Exception:
    HAS_SHAP = False
    shap = None  # type: ignore

try:
    from _deep_tabular_pipeline import predict_bundle

    HAS_DEEP_BUNDLE = True
except Exception:
    HAS_DEEP_BUNDLE = False
    predict_bundle = None  # type: ignore


MODEL_INDEX_MAP = {
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate outputs for 10 models")
    p.add_argument("--outputs-root", type=str, default=r"F:\KeTi\Project\outputs")
    p.add_argument("--data-path", type=str, default=r"F:\KeTi\Project\Data\ATrain.csv")
    p.add_argument("--id-col", type=str, default="No")
    p.add_argument("--label-col", type=str, default="Disease")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--bootstrap-iter", type=int, default=2000)
    p.add_argument("--ci-seed", type=int, default=42)
    p.add_argument("--skip-shap", action="store_true")
    p.add_argument("--shap-background-size", type=int, default=80)
    p.add_argument("--shap-eval-size", type=int, default=200)
    p.add_argument("--shap-nsamples", type=int, default=100)
    return p.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        nk = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_dict(v, nk))
        else:
            out[nk] = v
    return out


def _detect_model_dirs(outputs_root: Path) -> Dict[int, Path]:
    candidates: Dict[int, List[Path]] = {}
    for p in outputs_root.iterdir():
        if not p.is_dir():
            continue
        m = re.match(r"^(\d+)", p.name)
        if not m:
            continue
        idx = int(m.group(1))
        if idx in MODEL_INDEX_MAP:
            candidates.setdefault(idx, []).append(p)

    resolved: Dict[int, Path] = {}
    for idx, paths in candidates.items():
        # 同一序号多个目录时取最近修改
        paths_sorted = sorted(paths, key=lambda x: x.stat().st_mtime, reverse=True)
        resolved[idx] = paths_sorted[0]
    return resolved


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else float("nan")


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = _safe_div(tn, tn + fp)
    sensitivity = _safe_div(tp, tp + fn)
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    npv = _safe_div(tn, tn + fn)

    out = {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision_ppv": float(precision),
        "npv": float(npv),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    return out


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_boot: int,
    seed: int,
) -> Dict[str, Tuple[float, float]]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    metric_keys = [
        "auc",
        "auprc",
        "acc",
        "sensitivity",
        "specificity",
        "precision_ppv",
        "npv",
        "f1",
    ]
    vals: Dict[str, List[float]] = {k: [] for k in metric_keys}

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ys = y_true[idx]
        ps = y_prob[idx]
        # AUC/AP 要求有两类
        if len(np.unique(ys)) < 2:
            continue
        try:
            m = compute_metrics(ys, ps, threshold)
        except Exception:
            continue
        for k in metric_keys:
            v = m.get(k, np.nan)
            if np.isfinite(v):
                vals[k].append(v)

    ci: Dict[str, Tuple[float, float]] = {}
    for k in metric_keys:
        arr = np.array(vals[k], dtype=float)
        if arr.size == 0:
            ci[k] = (float("nan"), float("nan"))
        else:
            ci[k] = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
    return ci


def _load_full_X(data_path: Path, id_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    X = df.drop(columns=[id_col, label_col]).copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    return X


def _build_predict_fn(model_obj: Any, feature_names: List[str]) -> Callable[[np.ndarray], np.ndarray]:
    if isinstance(model_obj, dict) and "model_name" in model_obj and "state_dict" in model_obj:
        if not HAS_DEEP_BUNDLE:
            raise RuntimeError("检测到深度学习bundle，但无法导入 _deep_tabular_pipeline.predict_bundle")

        def _pred(arr: np.ndarray) -> np.ndarray:
            df = pd.DataFrame(arr, columns=feature_names)
            return np.asarray(predict_bundle(model_obj, df, batch_size=64), dtype=float).reshape(-1)

        return _pred

    if hasattr(model_obj, "predict_proba"):
        def _pred(arr: np.ndarray) -> np.ndarray:
            df = pd.DataFrame(arr, columns=feature_names)
            proba = model_obj.predict_proba(df)
            if proba.ndim == 2:
                return np.asarray(proba[:, 1], dtype=float)
            return np.asarray(proba, dtype=float).reshape(-1)

        return _pred

    if hasattr(model_obj, "decision_function"):
        def _pred(arr: np.ndarray) -> np.ndarray:
            df = pd.DataFrame(arr, columns=feature_names)
            z = np.asarray(model_obj.decision_function(df), dtype=float).reshape(-1)
            return 1.0 / (1.0 + np.exp(-z))

        return _pred

    raise TypeError("final_model.joblib 不支持 predict_proba / decision_function / deep bundle")


def _sample_rows(arr: np.ndarray, n_max: int, seed: int) -> np.ndarray:
    if len(arr) <= n_max:
        return arr
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(arr), size=n_max, replace=False)
    return arr[idx]


def compute_shap_for_model(
    model_obj: Any,
    X_model: pd.DataFrame,
    feature_names: List[str],
    background_size: int,
    eval_size: int,
    nsamples: int,
    seed: int,
) -> pd.DataFrame:
    if not HAS_SHAP:
        raise RuntimeError("未安装 shap，请先 pip install shap")

    pred_fn = _build_predict_fn(model_obj, feature_names)
    X_vals = X_model[feature_names].values.astype(float)
    X_vals = np.nan_to_num(X_vals, nan=np.nanmedian(X_vals, axis=0))

    background = _sample_rows(X_vals, background_size, seed=seed)
    eval_data = _sample_rows(X_vals, eval_size, seed=seed + 1)

    explainer = shap.KernelExplainer(pred_fn, background)
    shap_values = explainer.shap_values(eval_data, nsamples=nsamples)

    if isinstance(shap_values, list):
        # 二分类时通常返回 [class0, class1]
        sv = np.asarray(shap_values[-1], dtype=float)
    else:
        sv = np.asarray(shap_values, dtype=float)
        if sv.ndim == 3:
            sv = sv[:, :, -1]
    if sv.ndim != 2:
        raise RuntimeError(f"SHAP输出维度异常: {sv.shape}")

    mean_abs = np.mean(np.abs(sv), axis=0)
    out = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs,
        }
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out["n_eval_samples"] = len(eval_data)
    out["n_background_samples"] = len(background)
    return out


def main() -> None:
    args = parse_args()
    outputs_root = Path(args.outputs_root)
    data_path = Path(args.data_path)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = outputs_root / f"summary_reports_{ts}"
    result_dir.mkdir(parents=True, exist_ok=True)

    model_dirs = _detect_model_dirs(outputs_root)
    if len(model_dirs) == 0:
        raise FileNotFoundError(f"未在 {outputs_root} 检测到前缀1-10模型目录。")

    X_full = _load_full_X(data_path, args.id_col, args.label_col)

    table1_rows: List[Dict[str, Any]] = []
    table2_rows: List[Dict[str, Any]] = []
    table3_rows: List[Dict[str, Any]] = []
    table3_ci_rows: List[Dict[str, Any]] = []
    shap_rows: List[Dict[str, Any]] = []
    checklist_rows: List[Dict[str, Any]] = []

    checklist_rows.extend(
        [
            {"target_table": "table1_model_best_params.csv", "source_file": "locked_params.json", "columns_or_logic": "locked_params.* + locked_params_json"},
            {"target_table": "table2_model_features.csv", "source_file": "final_features_used.txt / auto_locked_features.txt", "columns_or_logic": "n_features + feature_names"},
            {"target_table": "table3_model_internal_performance.csv", "source_file": "cv_summary.json + oof_predictions.csv", "columns_or_logic": "cv_summary.* + oof_metric_*"},
            {"target_table": "table3_model_internal_performance_with_ci.csv", "source_file": "oof_predictions.csv", "columns_or_logic": "oof metrics + bootstrap 95%CI"},
            {"target_table": "shap_feature_importance_long.csv", "source_file": "final_model.joblib + final_features_used.txt + ATrain.csv", "columns_or_logic": "mean_abs_shap + rank"},
        ]
    )

    for idx in sorted(model_dirs.keys()):
        model_dir = model_dirs[idx]
        model_name = MODEL_INDEX_MAP[idx]
        locked_params_path = model_dir / "locked_params.json"
        final_feats_path = model_dir / "final_features_used.txt"
        auto_feats_path = model_dir / "auto_locked_features.txt"
        cv_summary_path = model_dir / "cv_summary.json"
        oof_path = model_dir / "oof_predictions.csv"
        final_model_path = model_dir / "final_model.joblib"

        # 表1：参数
        locked_params = _read_json(locked_params_path) if locked_params_path.exists() else {}
        row1: Dict[str, Any] = {
            "model_index": idx,
            "model": model_name,
            "folder_name": model_dir.name,
            "locked_params_json": json.dumps(locked_params, ensure_ascii=False),
        }
        row1.update(_flatten_dict(locked_params))
        table1_rows.append(row1)

        # 表2：变量
        if final_feats_path.exists():
            feats = [x.strip() for x in final_feats_path.read_text(encoding="utf-8").splitlines() if x.strip()]
            feat_source = "final_features_used.txt"
        elif auto_feats_path.exists():
            feats = [x.strip() for x in auto_feats_path.read_text(encoding="utf-8").splitlines() if x.strip()]
            feat_source = "auto_locked_features.txt"
        else:
            feats = []
            feat_source = "missing"
        table2_rows.append(
            {
                "model_index": idx,
                "model": model_name,
                "folder_name": model_dir.name,
                "feature_source": feat_source,
                "n_features": len(feats),
                "feature_names": ";".join(feats),
            }
        )

        # 表3：内部性能（cv_summary + oof）
        cv_summary = _read_json(cv_summary_path) if cv_summary_path.exists() else {}
        row3: Dict[str, Any] = {
            "model_index": idx,
            "model": model_name,
            "folder_name": model_dir.name,
        }
        row3.update(cv_summary)

        if oof_path.exists():
            oof_df = pd.read_csv(oof_path)
            y_true = oof_df["y_true"].astype(int).values
            y_prob = oof_df["y_prob"].astype(float).values
            oof_metrics = compute_metrics(y_true, y_prob, threshold=args.threshold)
            for k, v in oof_metrics.items():
                row3[f"oof_{k}"] = v

            ci = bootstrap_ci(
                y_true=y_true,
                y_prob=y_prob,
                threshold=args.threshold,
                n_boot=args.bootstrap_iter,
                seed=args.ci_seed + idx,
            )
            ci_row: Dict[str, Any] = {
                "model_index": idx,
                "model": model_name,
                "folder_name": model_dir.name,
            }
            for k, v in oof_metrics.items():
                ci_row[f"{k}_point"] = v
                lo, hi = ci[k]
                ci_row[f"{k}_ci_lower"] = lo
                ci_row[f"{k}_ci_upper"] = hi
            table3_ci_rows.append(ci_row)
        else:
            table3_ci_rows.append(
                {
                    "model_index": idx,
                    "model": model_name,
                    "folder_name": model_dir.name,
                    "note": "missing oof_predictions.csv",
                }
            )
        table3_rows.append(row3)

        # SHAP：基于 Stage B 模型
        if not args.skip_shap:
            if final_model_path.exists() and len(feats) > 0:
                try:
                    model_obj = joblib.load(final_model_path)
                    missing = [c for c in feats if c not in X_full.columns]
                    if len(missing) > 0:
                        raise ValueError(f"数据中缺少特征: {missing[:5]}")
                    X_model = X_full[feats].copy()
                    shap_df = compute_shap_for_model(
                        model_obj=model_obj,
                        X_model=X_model,
                        feature_names=feats,
                        background_size=args.shap_background_size,
                        eval_size=args.shap_eval_size,
                        nsamples=args.shap_nsamples,
                        seed=args.ci_seed + idx * 10,
                    )
                    shap_df["model_index"] = idx
                    shap_df["model"] = model_name
                    shap_df["folder_name"] = model_dir.name
                    shap_rows.extend(shap_df.to_dict(orient="records"))
                except Exception as e:
                    shap_rows.append(
                        {
                            "model_index": idx,
                            "model": model_name,
                            "folder_name": model_dir.name,
                            "feature": "__ERROR__",
                            "mean_abs_shap": float("nan"),
                            "rank": float("nan"),
                            "error": str(e),
                        }
                    )
            else:
                shap_rows.append(
                    {
                        "model_index": idx,
                        "model": model_name,
                        "folder_name": model_dir.name,
                        "feature": "__MISSING_FILES__",
                        "mean_abs_shap": float("nan"),
                        "rank": float("nan"),
                    }
                )

    table1_df = pd.DataFrame(table1_rows).sort_values(["model_index"])
    table2_df = pd.DataFrame(table2_rows).sort_values(["model_index"])
    table3_df = pd.DataFrame(table3_rows).sort_values(["model_index"])
    table3_ci_df = pd.DataFrame(table3_ci_rows).sort_values(["model_index"])
    checklist_df = pd.DataFrame(checklist_rows)

    table1_df.to_csv(result_dir / "table1_model_best_params.csv", index=False, encoding="utf-8-sig")
    table2_df.to_csv(result_dir / "table2_model_features.csv", index=False, encoding="utf-8-sig")
    table3_df.to_csv(result_dir / "table3_model_internal_performance.csv", index=False, encoding="utf-8-sig")
    table3_ci_df.to_csv(result_dir / "table3_model_internal_performance_with_ci.csv", index=False, encoding="utf-8-sig")
    checklist_df.to_csv(result_dir / "file_column_mapping_checklist.csv", index=False, encoding="utf-8-sig")

    if len(shap_rows) > 0:
        shap_long_df = pd.DataFrame(shap_rows).sort_values(["model_index", "rank"], na_position="last")
        shap_long_df.to_csv(result_dir / "shap_feature_importance_long.csv", index=False, encoding="utf-8-sig")
        ok_df = shap_long_df[
            (~shap_long_df["feature"].astype(str).str.startswith("__"))
            & (shap_long_df["mean_abs_shap"].notna())
        ].copy()
        if len(ok_df) > 0:
            shap_wide = ok_df.pivot_table(
                index="feature",
                columns="model",
                values="mean_abs_shap",
                aggfunc="mean",
            ).reset_index()
            shap_wide.to_csv(result_dir / "shap_feature_importance_wide.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "outputs_root": str(outputs_root.resolve()),
        "result_dir": str(result_dir.resolve()),
        "models_detected": {str(k): MODEL_INDEX_MAP[k] for k in sorted(model_dirs.keys())},
        "has_shap": HAS_SHAP,
        "skip_shap": bool(args.skip_shap),
        "bootstrap_iter": int(args.bootstrap_iter),
        "threshold": float(args.threshold),
    }
    (result_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(">>> 汇总完成")
    print(f"    结果目录: {result_dir}")
    print("    输出文件:")
    print("      - table1_model_best_params.csv")
    print("      - table2_model_features.csv")
    print("      - table3_model_internal_performance.csv")
    print("      - table3_model_internal_performance_with_ci.csv")
    print("      - file_column_mapping_checklist.csv")
    if not args.skip_shap:
        print("      - shap_feature_importance_long.csv")
        print("      - shap_feature_importance_wide.csv (若可生成)")


if __name__ == "__main__":
    main()