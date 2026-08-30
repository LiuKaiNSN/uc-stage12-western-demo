# -*- coding: utf-8 -*-
"""
UC vs 非UC: cnCV + ReliefF + RFE(XGB) + XGBoost 全流程脚本
---------------------------------------------------------
说明：
1) 该脚本与 SVM.py 的流程、输出结构保持一致，仅最终分类器与调参改为 XGBoost。
2) 阶段A：嵌套CV（外层评估、内层特征共识+调参）并输出透明化过程文件。
3) 阶段B：读取阶段A锁定参数与特征，训练最终模型；可选人工特征覆盖和外部验证。
4) 外层/内层特征共识：入选频率 freq=votes/n_splits，默认保留 freq > 0.6（外层5折仅 4/5 或 5/5 纳入）。

PyCharm「方式 A」运行配置（不要带 python 前缀）：
--stage A --data-path "F:\\KeTi\\Project\\Data\\ATrain.csv" --output-dir "F:\\KeTi\\Project\\outputs"
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.feature_selection import RFE
from sklearn.impute import IterativeImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:
    HAS_XGB = False
    XGBClassifier = None  # type: ignore

try:
    from skrebate import ReliefF as SkrebateReliefF

    HAS_SKREBATE = True
except Exception:
    HAS_SKREBATE = False
    SkrebateReliefF = None  # type: ignore


@dataclass
class Config:
    data_path: str
    external_path: Optional[str]
    output_dir: str
    stage: str
    id_col: str
    label_col: str
    outer_folds: int
    inner_folds: int
    random_state: int
    relief_top_k: int
    rfe_n_features: int
    rfe_step: int
    corr_threshold: float
    consensus_threshold: float
    final_max_features: int
    # 内层调参：RandomizedSearchCV 迭代次数（各超参分布见 tune_xgb_on_outer_train）
    random_search_n_iter: int
    manual_feature_file: Optional[str]
    run_dir: Optional[str]


def make_output_dir(base_dir: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(base_dir) / f"run_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(obj: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_last_run_pointer(base_output_dir: Path, run_dir: Path) -> None:
    base_output_dir.mkdir(parents=True, exist_ok=True)
    (base_output_dir / "LAST_RUN_DIR.txt").write_text(str(run_dir.resolve()), encoding="utf-8")


class ReliefFTopK:
    """折内 ReliefF 排序器，返回 top-k 列索引。"""

    def __init__(self, k: int = 20, n_neighbors: int = 5, discrete_threshold: int = 10):
        self.k = k
        self.n_neighbors = n_neighbors
        self.discrete_threshold = discrete_threshold

    def fit_score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        if HAS_SKREBATE:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fs = SkrebateReliefF(
                    n_neighbors=self.n_neighbors,
                    discrete_threshold=self.discrete_threshold,
                    verbose=False,
                    n_jobs=-1,
                )
                fs.fit(X, y)
                scores = np.asarray(fs.feature_importances_, dtype=float)
        else:
            # 回退：仅用于脚本可跑通，正式分析建议安装 skrebate
            scores = np.var(X, axis=0)
        return scores

    def topk_indices(self, scores: np.ndarray) -> np.ndarray:
        k = min(self.k, len(scores))
        idx = np.argsort(-scores)[:k]
        return np.sort(idx)


def load_data(data_path: str, id_col: str, label_col: str) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    df = pd.read_csv(data_path)
    if id_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"缺少必要列: id_col={id_col}, label_col={label_col}, 实际前10列={list(df.columns[:10])}"
        )
    y = df[label_col].astype(int).values
    X_df = df.drop(columns=[id_col, label_col]).copy()
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    return X_df, y, list(X_df.columns)


def build_imputer(random_state: int) -> IterativeImputer:
    return IterativeImputer(
        estimator=RandomForestRegressor(
            n_estimators=60,
            max_depth=8,
            random_state=random_state,
            n_jobs=-1,
        ),
        max_iter=20,
        random_state=random_state,
        sample_posterior=False,
    )


def get_rfe_estimator(random_state: int):
    """RFE 基学习器（与 SVM.py 一致：轻量 XGB）。"""
    if HAS_XGB:
        return XGBClassifier(
            n_estimators=40,
            max_depth=2,
            learning_rate=0.08,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=4,
            eval_metric="logloss",
        )
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )


def pearson_abs_with_y(X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    yv = pd.Series(y).astype(float).values
    for c in X.columns:
        xv = pd.to_numeric(X[c], errors="coerce").astype(float).values
        if np.nanstd(xv) == 0:
            out[c] = 0.0
            continue
        mask = ~np.isnan(xv)
        if mask.sum() < 3:
            out[c] = 0.0
            continue
        corr = np.corrcoef(xv[mask], yv[mask])[0, 1]
        out[c] = float(abs(corr)) if not np.isnan(corr) else 0.0
    return out


def spearman_prune(
    X_df_selected: pd.DataFrame,
    y: np.ndarray,
    corr_threshold: float = 0.8,
) -> Tuple[List[str], List[dict]]:
    work_cols = list(X_df_selected.columns)
    drop_logs: List[dict] = []
    if len(work_cols) <= 1:
        return work_cols, drop_logs

    y_corr = pearson_abs_with_y(X_df_selected[work_cols], y)

    while True:
        if len(work_cols) <= 1:
            break
        corr_mat = X_df_selected[work_cols].corr(method="spearman").abs().fillna(0.0)
        corr_vals = np.array(corr_mat.to_numpy(dtype=float, copy=True), copy=True)
        np.fill_diagonal(corr_vals, 0.0)

        max_val = float(corr_vals.max())
        if max_val <= corr_threshold:
            break
        pair_idx = np.unravel_index(int(np.argmax(corr_vals)), corr_vals.shape)
        f1 = corr_mat.index[pair_idx[0]]
        f2 = corr_mat.columns[pair_idx[1]]

        s1 = y_corr.get(f1, 0.0)
        s2 = y_corr.get(f2, 0.0)
        if s1 < s2:
            drop_f, keep_f = f1, f2
        elif s2 < s1:
            drop_f, keep_f = f2, f1
        else:
            drop_f = sorted([f1, f2])[-1]
            keep_f = f1 if drop_f == f2 else f2

        drop_logs.append(
            {
                "feature_drop": drop_f,
                "feature_keep": keep_f,
                "spearman_abs": float(max_val),
                "y_corr_drop": float(y_corr.get(drop_f, 0.0)),
                "y_corr_keep": float(y_corr.get(keep_f, 0.0)),
                "rule": "drop lower |corr(feature, y)|",
            }
        )
        work_cols.remove(drop_f)

    return work_cols, drop_logs


def select_features_in_fold(
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
    cfg: Config,
    random_state: int,
) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    # 1) 折内插补
    imputer = build_imputer(random_state=random_state)
    X_imp = imputer.fit_transform(X_train_df.values)
    all_features = list(X_train_df.columns)

    # 2) ReliefF
    relief = ReliefFTopK(k=cfg.relief_top_k, n_neighbors=5, discrete_threshold=10)
    scores = relief.fit_score(X_imp, y_train)
    top_idx = relief.topk_indices(scores)
    top_features = [all_features[i] for i in top_idx]
    relief_df = (
        pd.DataFrame({"feature": all_features, "relief_score": scores})
        .sort_values("relief_score", ascending=False)
        .reset_index(drop=True)
    )

    # 3) RFE(XGB)
    X_top = X_imp[:, top_idx]
    rfe_est = get_rfe_estimator(random_state=random_state)
    n_select = min(cfg.rfe_n_features, X_top.shape[1])
    rfe = RFE(estimator=rfe_est, n_features_to_select=n_select, step=cfg.rfe_step)
    rfe.fit(X_top, y_train)
    rfe_local_idx = np.where(rfe.support_)[0]
    rfe_features = [top_features[i] for i in rfe_local_idx]

    # 4) Spearman 去相关
    X_rfe_df = pd.DataFrame(X_imp, columns=all_features)[rfe_features]
    kept_features, drop_logs = spearman_prune(
        X_df_selected=X_rfe_df,
        y=y_train,
        corr_threshold=cfg.corr_threshold,
    )
    return kept_features, relief_df, pd.DataFrame(drop_logs)


def inner_consensus_features(
    X_outer_train: pd.DataFrame,
    y_outer_train: np.ndarray,
    cfg: Config,
    seed_offset: int = 0,
    out_dir_fold: Optional[Path] = None,
) -> Tuple[List[str], pd.DataFrame]:
    skf_inner = StratifiedKFold(
        n_splits=cfg.inner_folds,
        shuffle=True,
        random_state=cfg.random_state + seed_offset,
    )
    feature_counter = Counter()

    for inner_i, (tr_idx, _) in enumerate(skf_inner.split(X_outer_train, y_outer_train), start=1):
        X_tr = X_outer_train.iloc[tr_idx].copy()
        y_tr = y_outer_train[tr_idx]
        selected, relief_df, drop_df = select_features_in_fold(
            X_train_df=X_tr,
            y_train=y_tr,
            cfg=cfg,
            random_state=cfg.random_state + seed_offset + inner_i,
        )
        for f in selected:
            feature_counter[f] += 1

        if out_dir_fold is not None:
            relief_df.to_csv(out_dir_fold / f"inner{inner_i}_relief_ranking.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame({"selected_feature": selected}).to_csv(
                out_dir_fold / f"inner{inner_i}_selected_after_rfe_corr.csv",
                index=False,
                encoding="utf-8-sig",
            )
            if len(drop_df) > 0:
                drop_df.to_csv(
                    out_dir_fold / f"inner{inner_i}_spearman_drop_log.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

    consensus = [
        f
        for f, c in feature_counter.items()
        if (c / cfg.inner_folds) > cfg.consensus_threshold
    ]

    freq_df = (
        pd.DataFrame(
            [{"feature": f, "votes": c, "freq": c / cfg.inner_folds} for f, c in feature_counter.items()]
        )
        .sort_values(["votes", "feature"], ascending=[False, True])
        .reset_index(drop=True)
    )

    if len(consensus) > cfg.final_max_features:
        rank_map = {r["feature"]: r["votes"] for _, r in freq_df.iterrows()}
        consensus = sorted(consensus, key=lambda x: (-rank_map.get(x, 0), x))[: cfg.final_max_features]

    return sorted(consensus), freq_df


def build_xgb_pipeline(final_params: dict, random_state: int) -> Pipeline:
    """
    最终分类器改为 XGBoost；其余流程与 SVM.py 保持一致（插补、阶段A/B、输出结构一致）。
    """
    imputer = build_imputer(random_state=random_state)
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=int(final_params.get("n_estimators", 40)),
        max_depth=int(final_params.get("max_depth", 2)),
        learning_rate=float(final_params.get("learning_rate", 0.1)),
        subsample=float(final_params.get("subsample", 0.7)),
        colsample_bytree=float(final_params.get("colsample_bytree", 0.7)),
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=4,
    )
    return Pipeline([("imputer", imputer), ("xgb", xgb)])


def tune_xgb_on_outer_train(
    X_train_sel: pd.DataFrame,
    y_train: np.ndarray,
    cfg: Config,
    random_state: int,
) -> Tuple[dict, pd.DataFrame]:
    pipe = build_xgb_pipeline(
        final_params={
            "n_estimators": 40,
            "max_depth": 2,
            "learning_rate": 0.1,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
        },
        random_state=random_state,
    )

    param_distributions = {
        "xgb__n_estimators": [40, 80, 120, 160, 200],
        "xgb__max_depth": randint(3, 7),
        "xgb__learning_rate": loguniform(0.05, 0.1),
        "xgb__subsample": uniform(loc=0.6, scale=0.3),
        "xgb__colsample_bytree": uniform(loc=0.6, scale=0.3),
    }
    inner_cv = StratifiedKFold(
        n_splits=cfg.inner_folds,
        shuffle=True,
        random_state=random_state + 99,
    )
    rs = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=cfg.random_search_n_iter,
        scoring="roc_auc",
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
        random_state=random_state + 123,
    )
    rs.fit(X_train_sel, y_train)
    bp = rs.best_params_
    best_params = {
        "n_estimators": int(bp["xgb__n_estimators"]),
        "max_depth": int(bp["xgb__max_depth"]),
        "learning_rate": float(bp["xgb__learning_rate"]),
        "subsample": float(bp["xgb__subsample"]),
        "colsample_bytree": float(bp["xgb__colsample_bytree"]),
        "best_inner_auc": float(rs.best_score_),
    }
    cv_results = pd.DataFrame(rs.cv_results_).sort_values("rank_test_score")
    return best_params, cv_results


def evaluate_binary(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def aggregate_locked_params(best_params_per_fold: pd.DataFrame) -> dict:
    """将每折最优XGB参数聚合为最终锁定参数。"""
    return {
        "n_estimators": int(round(float(best_params_per_fold["n_estimators"].median()))),
        "max_depth": int(round(float(best_params_per_fold["max_depth"].median()))),
        "learning_rate": float(best_params_per_fold["learning_rate"].median()),
        "subsample": float(best_params_per_fold["subsample"].median()),
        "colsample_bytree": float(best_params_per_fold["colsample_bytree"].median()),
    }


def load_manual_features(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"manual feature file not found: {path}")
    if p.suffix.lower() in [".yml", ".yaml"]:
        with open(p, "r", encoding="utf-8") as f:
            obj = yaml.safe_load(f)
        if isinstance(obj, dict) and "features" in obj:
            feats = obj["features"]
        elif isinstance(obj, list):
            feats = obj
        else:
            raise ValueError("YAML格式错误，请使用 {'features':[...]} 或列表")
        return [str(x).strip() for x in feats if str(x).strip()]
    lines = p.read_text(encoding="utf-8").splitlines()
    return [x.strip() for x in lines if x.strip() and not x.strip().startswith("#")]


def run_stage_a(X_df: pd.DataFrame, y: np.ndarray, cfg: Config, out_dir: Path) -> None:
    print(">>> 阶段A开始：嵌套CV + 特征共识 + 调参 + 内部评估")
    skf_outer = StratifiedKFold(n_splits=cfg.outer_folds, shuffle=True, random_state=cfg.random_state)
    fold_metrics: List[dict] = []
    fold_best_params: List[dict] = []
    outer_feature_sets: List[List[str]] = []
    oof_records: List[dict] = []

    for fold_i, (tr_idx, va_idx) in enumerate(skf_outer.split(X_df, y), start=1):
        fold_dir = out_dir / f"outer_fold_{fold_i}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        X_tr, y_tr = X_df.iloc[tr_idx].copy(), y[tr_idx]
        X_va, y_va = X_df.iloc[va_idx].copy(), y[va_idx]

        consensus_feats, inner_freq_df = inner_consensus_features(
            X_outer_train=X_tr,
            y_outer_train=y_tr,
            cfg=cfg,
            seed_offset=fold_i * 1000,
            out_dir_fold=fold_dir,
        )
        pd.DataFrame({"feature": consensus_feats}).to_csv(
            fold_dir / "consensus_features_this_outer_fold.csv",
            index=False,
            encoding="utf-8-sig",
        )
        inner_freq_df.to_csv(fold_dir / "inner_feature_frequency.csv", index=False, encoding="utf-8-sig")
        outer_feature_sets.append(consensus_feats)

        X_tr_sel = X_tr[consensus_feats].copy()
        X_va_sel = X_va[consensus_feats].copy()
        best_params, cv_results = tune_xgb_on_outer_train(
            X_train_sel=X_tr_sel,
            y_train=y_tr,
            cfg=cfg,
            random_state=cfg.random_state + fold_i,
        )
        cv_results.to_csv(fold_dir / "gridsearch_cv_results.csv", index=False, encoding="utf-8-sig")

        pipe = build_xgb_pipeline(best_params, random_state=cfg.random_state + fold_i)
        pipe.fit(X_tr_sel, y_tr)
        y_prob = pipe.predict_proba(X_va_sel)[:, 1]

        m = evaluate_binary(y_va, y_prob, threshold=0.5)
        m["outer_fold"] = fold_i
        m["n_features"] = len(consensus_feats)
        fold_metrics.append(m)

        fold_best_params.append(
            {
                "outer_fold": fold_i,
                "n_estimators": best_params["n_estimators"],
                "max_depth": best_params["max_depth"],
                "learning_rate": best_params["learning_rate"],
                "subsample": best_params["subsample"],
                "colsample_bytree": best_params["colsample_bytree"],
                "best_inner_auc": best_params["best_inner_auc"],
                "n_features": len(consensus_feats),
            }
        )

        for idx, prob, yt in zip(va_idx, y_prob, y_va):
            oof_records.append(
                {"row_index": int(idx), "y_true": int(yt), "y_prob": float(prob), "outer_fold": fold_i}
            )

    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_df.to_csv(out_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")

    summary = {
        "auc_mean": float(fold_metrics_df["auc"].mean()),
        "auc_std": float(fold_metrics_df["auc"].std(ddof=1)),
        "auprc_mean": float(fold_metrics_df["auprc"].mean()),
        "auprc_std": float(fold_metrics_df["auprc"].std(ddof=1)),
        "acc_mean": float(fold_metrics_df["acc"].mean()),
        "f1_mean": float(fold_metrics_df["f1"].mean()),
        "precision_mean": float(fold_metrics_df["precision"].mean()),
        "recall_mean": float(fold_metrics_df["recall"].mean()),
    }
    save_json(summary, out_dir / "cv_summary.json")

    best_params_df = pd.DataFrame(fold_best_params)
    best_params_df.to_csv(out_dir / "best_params_per_fold.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(oof_records).sort_values("row_index").to_csv(
        out_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig"
    )

    feat_counter = Counter()
    for feats in outer_feature_sets:
        feat_counter.update(feats)
    freq_df = (
        pd.DataFrame(
            [{"feature": f, "votes": c, "freq": c / cfg.outer_folds} for f, c in feat_counter.items()]
        )
        .sort_values(["votes", "feature"], ascending=[False, True])
        .reset_index(drop=True)
    )
    freq_df.to_csv(out_dir / "feature_frequency.csv", index=False, encoding="utf-8-sig")

    auto_feats = freq_df.loc[
        (freq_df["votes"] / cfg.outer_folds) > cfg.consensus_threshold,
        "feature",
    ].tolist()
    if len(auto_feats) > cfg.final_max_features:
        auto_feats = auto_feats[: cfg.final_max_features]
    (out_dir / "auto_locked_features.txt").write_text("\n".join(auto_feats) + "\n", encoding="utf-8")

    locked_params = aggregate_locked_params(best_params_df)
    save_json(locked_params, out_dir / "locked_params.json")

    stage_a_meta = {
        "model": "XGBoost",
        "consensus_threshold": cfg.consensus_threshold,
        "outer_folds": cfg.outer_folds,
        "inner_folds": cfg.inner_folds,
        "relief_top_k": cfg.relief_top_k,
        "rfe_n_features": cfg.rfe_n_features,
        "rfe_step": cfg.rfe_step,
        "corr_threshold": cfg.corr_threshold,
        "final_max_features": cfg.final_max_features,
        "hyperparam_search": {
            "method": "RandomizedSearchCV",
            "n_iter": cfg.random_search_n_iter,
            "scoring": "roc_auc",
            "param_distributions": {
                "xgb__n_estimators": {"type": "choice", "values": [40, 80, 120, 160, 200]},
                "xgb__max_depth": {"type": "randint", "low": 3, "high_exclusive": 7, "integers": "3..6"},
                "xgb__learning_rate": {"type": "loguniform", "low": 0.05, "high": 0.1},
                "xgb__subsample": {"type": "uniform", "low": 0.6, "high": 0.9},
                "xgb__colsample_bytree": {"type": "uniform", "low": 0.6, "high": 0.9},
            },
        },
        "has_skrebate": HAS_SKREBATE,
        "has_xgboost": HAS_XGB,
        "auto_locked_feature_count": len(auto_feats),
    }
    save_json(stage_a_meta, out_dir / "stage_a_meta.json")

    print(">>> 阶段A完成")
    print(f"    内部CV AUC(mean±sd): {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}")
    print(f"    自动锁定特征数: {len(auto_feats)}")
    print(f"    结果目录: {out_dir}")


def run_stage_b(X_df: pd.DataFrame, y: np.ndarray, cfg: Config, out_dir: Path) -> None:
    print(">>> 阶段B开始：加载锁定结果 -> 可选人工覆盖 -> 全开发集训练最终模型")
    auto_feat_path = out_dir / "auto_locked_features.txt"
    locked_params_path = out_dir / "locked_params.json"
    if not auto_feat_path.exists() or not locked_params_path.exists():
        raise FileNotFoundError(
            "缺少阶段A输出文件，请先运行阶段A，或确保目录中存在 auto_locked_features.txt + locked_params.json"
        )

    auto_feats = [x.strip() for x in auto_feat_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    locked_params = json.loads(locked_params_path.read_text(encoding="utf-8"))

    used_feature_source = "auto_locked_features"
    final_feats = auto_feats
    if cfg.manual_feature_file:
        manual_feats = load_manual_features(cfg.manual_feature_file)
        manual_feats = [f for f in manual_feats if f in X_df.columns]
        if len(manual_feats) == 0:
            raise ValueError("manual feature file 已提供，但没有一个特征名与数据列匹配")
        final_feats = sorted(manual_feats)
        used_feature_source = f"manual_override({cfg.manual_feature_file})"

    X_sel = X_df[final_feats].copy()
    final_pipe = build_xgb_pipeline(locked_params, random_state=cfg.random_state + 777)
    final_pipe.fit(X_sel, y)

    y_prob_train = final_pipe.predict_proba(X_sel)[:, 1]
    train_metrics = evaluate_binary(y, y_prob_train, threshold=0.5)

    joblib.dump(final_pipe, out_dir / "final_model.joblib")
    (out_dir / "final_features_used.txt").write_text("\n".join(final_feats) + "\n", encoding="utf-8")
    save_json(
        {
            "model": "XGBoost",
            "feature_source": used_feature_source,
            "n_final_features": len(final_feats),
            "final_params": locked_params,
            "train_metrics_demo": train_metrics,
        },
        out_dir / "final_model_meta.json",
    )

    if cfg.external_path:
        print(">>> 检测到 external_path，开始外部验证")
        X_ext, y_ext, _ = load_data(cfg.external_path, cfg.id_col, cfg.label_col)
        miss = [c for c in final_feats if c not in X_ext.columns]
        if miss:
            raise ValueError(f"外部数据缺少最终特征列: {miss}")
        X_ext_sel = X_ext[final_feats].copy()
        y_prob_ext = final_pipe.predict_proba(X_ext_sel)[:, 1]
        ext_metrics = evaluate_binary(y_ext, y_prob_ext, threshold=0.5)
        pd.DataFrame({"y_true": y_ext.astype(int), "y_prob": y_prob_ext.astype(float)}).to_csv(
            out_dir / "external_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        save_json(ext_metrics, out_dir / "external_metrics.json")
        print(f">>> 外部验证AUC: {ext_metrics['auc']:.4f}")

    print(">>> 阶段B完成")
    print(f"    最终特征数: {len(final_feats)}")
    print(f"    特征来源: {used_feature_source}")
    print(f"    结果目录: {out_dir}")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="cnCV + ReliefF + RFE(XGB) + XGBoost for UC vs non-UC")
    parser.add_argument("--data-path", type=str, default=r"F:\KeTi\Project\Data\ATrain.csv")
    parser.add_argument("--external-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=r"F:\KeTi\Project\outputs")
    parser.add_argument("--stage", type=str, default="A", choices=["A", "B", "all"])
    parser.add_argument("--id-col", type=str, default="No")
    parser.add_argument("--label-col", type=str, default="Disease")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)

    # 特征工程默认与 SVM.py 一致
    parser.add_argument("--relief-top-k", type=int, default=30)
    parser.add_argument("--rfe-n-features", type=int, default=20)
    parser.add_argument("--rfe-step", type=int, default=3)
    parser.add_argument("--corr-threshold", type=float, default=0.8)
    parser.add_argument(
        "--consensus-threshold",
        type=float,
        default=0.6,
        help="保留 freq=votes/n_splits > 该值；默认0.6，外层5折下仅 4/5 或 5/5 纳入",
    )
    parser.add_argument("--final-max-features", type=int, default=20)

    parser.add_argument(
        "--random-search-n-iter",
        type=int,
        default=8,
        help="内层 RandomizedSearchCV 的 n_iter（默认 8）",
    )

    parser.add_argument("--manual-feature-file", type=str, default=None)
    parser.add_argument("--run-dir", type=str, default=None)
    args = parser.parse_args()

    return Config(
        data_path=args.data_path,
        external_path=args.external_path,
        output_dir=args.output_dir,
        stage=args.stage,
        id_col=args.id_col,
        label_col=args.label_col,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        random_state=args.random_state,
        relief_top_k=args.relief_top_k,
        rfe_n_features=args.rfe_n_features,
        rfe_step=args.rfe_step,
        corr_threshold=args.corr_threshold,
        consensus_threshold=args.consensus_threshold,
        final_max_features=args.final_max_features,
        random_search_n_iter=args.random_search_n_iter,
        manual_feature_file=args.manual_feature_file,
        run_dir=args.run_dir,
    )


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if not HAS_XGB:
        raise ImportError(
            "当前环境未检测到 xgboost，无法运行 XGBoost.py。请先执行: pip install xgboost"
        )

    cfg = parse_args()
    base_out = Path(cfg.output_dir)

    if cfg.stage in ["A", "all"]:
        out_dir = make_output_dir(str(base_out))
    else:
        if cfg.run_dir:
            if str(cfg.run_dir).strip().lower() == "auto":
                last_file = base_out / "LAST_RUN_DIR.txt"
                if not last_file.exists():
                    raise FileNotFoundError(
                        f"未找到 {last_file}。请先成功运行一次阶段A，或改用 --run-dir 指向具体 run_ 文件夹。"
                    )
                out_dir = Path(last_file.read_text(encoding="utf-8").strip())
            else:
                out_dir = Path(cfg.run_dir)
        else:
            out_dir = Path(cfg.output_dir)
        if not out_dir.exists():
            raise FileNotFoundError(f"阶段B指定的 run 目录不存在: {out_dir}")

    print(">>> 读取数据")
    X_df, y, feature_names = load_data(cfg.data_path, cfg.id_col, cfg.label_col)
    print(f"    样本数: {len(y)}; 特征数: {len(feature_names)}; 阳性率: {y.mean():.4f}")
    print(f"    HAS_SKREBATE={HAS_SKREBATE}, HAS_XGB={HAS_XGB}")

    if cfg.stage in ["A", "all"]:
        run_stage_a(X_df, y, cfg, out_dir)
        write_last_run_pointer(base_out, out_dir)
        print(f">>> 已写入最新 run 目录指针: {(base_out / 'LAST_RUN_DIR.txt').resolve()}")
        print(f"    内容为: {out_dir.resolve()}")

    if cfg.stage in ["B", "all"]:
        run_stage_b(X_df, y, cfg, out_dir)

    print(">>> 全部完成")


if __name__ == "__main__":
    main()
