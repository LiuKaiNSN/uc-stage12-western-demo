# -*- coding: utf-8 -*-
"""
UC vs 非UC: cnCV + ReliefF + RFE(XGB) + SVM 全流程脚本
-------------------------------------------------
作者：你自己（建议保留）
说明：
1) 阶段A（全自动）:
   - 嵌套CV（外层评估，内层特征共识 + 调参）
   - 输出特征频率、自动锁定特征、每折最优参数、汇总参数等
2) 阶段B（可选人工闸门）:
   - 若提供 manual_features.yaml/txt 则覆盖自动特征
   - 训练最终模型并保存，可选外部验证
3) 关键设计：
   - 避免信息泄漏：所有特征选择都在训练折内部完成
   - 频率共识：外层/内层均以「折间入选频率 > 0.6」判定（外层5折仅 4/5 或 5/5 纳入）

==============================================================================
PyCharm「方式 A」运行配置（不要带 python 前缀）
------------------------------------------------------------------------------
1) 打开本文件 → 右上角 Run 旁下拉 → Edit Configurations…
2) Script path 选： F:\\KeTi\\Project\\Script\\SVM.py
3) Parameters 粘贴下面「整行之一」：

【推荐新手】仅阶段A（嵌套CV + 输出；耗时较长但比 all 短）：
--stage A --data-path "F:\\KeTi\\Project\\Data\\ATrain.csv" --output-dir "F:\\KeTi\\Project\\outputs"

【一次跑完】阶段A+阶段B（最慢；会训练最终模型）：
--stage all --data-path "F:\\KeTi\\Project\\Data\\ATrain.csv" --output-dir "F:\\KeTi\\Project\\outputs"

【仅阶段B】在阶段A完成后：自动读取 outputs\\LAST_RUN_DIR.txt（见阶段A结束打印）
--stage B --data-path "F:\\KeTi\\Project\\Data\\ATrain.csv" --output-dir "F:\\KeTi\\Project\\outputs" --run-dir auto

【仅阶段B + 人工特征】同上，再加：
--manual-feature-file "F:\\KeTi\\Project\\manual_features.yaml"

4) 直接点绿色运行、且 Parameters 留空时：
   默认等价于「仅阶段A」（避免误点 all 跑一整天）
==============================================================================
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
from scipy.stats import loguniform

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
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
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# xgboost 作为RFE基学习器（若环境无xgboost则降级RF）
try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:
    HAS_XGB = False
    XGBClassifier = None  # type: ignore

# 标准 ReliefF（skrebate）
try:
    from skrebate import ReliefF as SkrebateReliefF

    HAS_SKREBATE = True
except Exception:
    HAS_SKREBATE = False
    SkrebateReliefF = None  # type: ignore


# ==========================
# 数据类与基础工具
# ==========================
@dataclass
class Config:
    data_path: str
    external_path: Optional[str]
    output_dir: str
    stage: str  # A / B / all
    id_col: str
    label_col: str

    outer_folds: int
    inner_folds: int
    random_state: int

    # 特征工程相关
    relief_top_k: int
    rfe_n_features: int
    rfe_step: int
    corr_threshold: float
    # 与折数配合使用：入选频率 freq = votes/n_splits，保留 freq > consensus_threshold（默认 0.6）
    consensus_threshold: float
    final_max_features: int

    # SVM调参候选（用于随机搜索）
    svm_c_grid: List[float]
    svm_gamma_grid: List[Union[str, float]]
    random_search_n_iter: int

    # 人工闸门文件（可选）
    manual_feature_file: Optional[str]
    # 阶段B：指向某次阶段A生成的 run_时间戳 目录；或填 auto 读取 LAST_RUN_DIR.txt
    run_dir: Optional[str]


def make_output_dir(base_dir: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(base_dir) / f"run_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(obj: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_last_run_pointer(base_output_dir: Path, run_dir: Path) -> None:
    """
    在 outputs 根目录写入 LAST_RUN_DIR.txt，便于阶段B使用 --run-dir auto。
    """
    base_output_dir.mkdir(parents=True, exist_ok=True)
    p = base_output_dir / "LAST_RUN_DIR.txt"
    p.write_text(str(run_dir.resolve()), encoding="utf-8")


# ==========================
# ReliefF 封装
# ==========================
class ReliefFTopK:
    """
    非sklearn transformer（只用于折内特征排名）
    """
    def __init__(self, k: int = 30, n_neighbors: int = 5, discrete_threshold: int = 10):
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
            # 回退方案：若未安装skrebate，使用方差近似（仅为脚本可运行）
            # 正式论文建议必须安装skrebate
            scores = np.var(X, axis=0)

        return scores

    def topk_indices(self, scores: np.ndarray) -> np.ndarray:
        k = min(self.k, len(scores))
        idx = np.argsort(-scores)[:k]
        return np.sort(idx)


# ==========================
# 核心方法函数
# ==========================
def load_data(data_path: str, id_col: str, label_col: str) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    读取CSV并返回:
    - X_df: 仅特征
    - y: 标签
    - feature_names: 特征名列表
    """
    df = pd.read_csv(data_path)
    if id_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"缺少必要列: id_col={id_col}, label_col={label_col}, 实际前10列={list(df.columns[:10])}"
        )

    y = df[label_col].astype(int).values
    X_df = df.drop(columns=[id_col, label_col]).copy()

    # 全部转数值，异常值转NaN（后续由插补处理）
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    feature_names = list(X_df.columns)
    return X_df, y, feature_names


def build_imputer(random_state: int) -> IterativeImputer:
    """
    折内插补器：
    说明：这里使用RF回归器驱动IterativeImputer，保留你设定的方法路线。
    """
    imputer = IterativeImputer(
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
    return imputer


def get_rfe_estimator(random_state: int):
    """
    RFE基学习器：优先XGBoost，若环境不可用则退化到RandomForestClassifier
    """
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
    """
    计算每个特征与y的绝对相关性（用于共线冲突时决定删谁）
    """
    out = {}
    yv = pd.Series(y).astype(float).values
    for c in X.columns:
        xv = pd.to_numeric(X[c], errors="coerce").astype(float).values
        if np.nanstd(xv) == 0:
            out[c] = 0.0
            continue
        # np.corrcoef遇nan需处理
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
    """
    对候选特征做Spearman相关性剪枝：
    - 若两变量|rho| > 阈值，删除与y相关性更低的变量
    返回:
    - 保留特征
    - 删除记录（用于透明化输出）
    """
    work_cols = list(X_df_selected.columns)
    drop_logs = []

    if len(work_cols) <= 1:
        return work_cols, drop_logs

    # 与y相关性（用于冲突裁决）
    y_corr = pearson_abs_with_y(X_df_selected[work_cols], y)

    while True:
        if len(work_cols) <= 1:
            break

        corr_mat = X_df_selected[work_cols].corr(method="spearman").abs().fillna(0.0)
        # 必须 copy：corr_mat.values 可能是只读视图，fill_diagonal 会报 read-only
        corr_vals = np.array(corr_mat.to_numpy(dtype=float, copy=True), copy=True)
        np.fill_diagonal(corr_vals, 0.0)

        # 找最大相关的一对
        max_val = float(corr_vals.max())
        if max_val <= corr_threshold:
            break

        pair_idx = np.unravel_index(int(np.argmax(corr_vals)), corr_vals.shape)
        f1 = corr_mat.index[pair_idx[0]]
        f2 = corr_mat.columns[pair_idx[1]]
        # 删除与y相关性更低者；若相等，删字典序靠后的（保证确定性）
        s1 = y_corr.get(f1, 0.0)
        s2 = y_corr.get(f2, 0.0)

        if s1 < s2:
            drop_f = f1
            keep_f = f2
        elif s2 < s1:
            drop_f = f2
            keep_f = f1
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
    """
    单个“训练折”内部执行完整特征工程：
    1) 插补
    2) ReliefF Top-K
    3) RFE(XGB)
    4) Spearman去相关
    返回：
    - 最终特征列表
    - relief排名明细
    - spearman删除明细
    """
    # 1) 折内拟合插补器（避免泄漏）
    imputer = build_imputer(random_state=random_state)
    X_imp = imputer.fit_transform(X_train_df.values)
    all_features = list(X_train_df.columns)

    # 2) ReliefF排序
    relief = ReliefFTopK(k=cfg.relief_top_k, n_neighbors=5, discrete_threshold=10)
    scores = relief.fit_score(X_imp, y_train)
    top_idx = relief.topk_indices(scores)
    top_features = [all_features[i] for i in top_idx]

    relief_df = pd.DataFrame({
        "feature": all_features,
        "relief_score": scores
    }).sort_values("relief_score", ascending=False).reset_index(drop=True)

    # 3) RFE(XGB) 在ReliefF候选子集内精筛
    X_top = X_imp[:, top_idx]
    rfe_est = get_rfe_estimator(random_state=random_state)
    n_select = min(cfg.rfe_n_features, X_top.shape[1])
    rfe = RFE(estimator=rfe_est, n_features_to_select=n_select, step=cfg.rfe_step)
    rfe.fit(X_top, y_train)
    rfe_local_idx = np.where(rfe.support_)[0]
    rfe_features = [top_features[i] for i in rfe_local_idx]

    # 4) Spearman去相关（在RFE保留集中执行）
    X_rfe_df = pd.DataFrame(X_imp, columns=all_features)[rfe_features]
    kept_features, drop_logs = spearman_prune(
        X_df_selected=X_rfe_df,
        y=y_train,
        corr_threshold=cfg.corr_threshold
    )

    drop_df = pd.DataFrame(drop_logs)
    return kept_features, relief_df, drop_df


def inner_consensus_features(
    X_outer_train: pd.DataFrame,
    y_outer_train: np.ndarray,
    cfg: Config,
    seed_offset: int = 0,
    out_dir_fold: Optional[Path] = None,
) -> Tuple[List[str], pd.DataFrame]:
    """
    cnCV核心：在外层训练集内部做内层K折特征选择，
    用“入选频率”形成共识特征集（默认 freq > 0.6）
    """
    skf_inner = StratifiedKFold(
        n_splits=cfg.inner_folds,
        shuffle=True,
        random_state=cfg.random_state + seed_offset,
    )

    feature_counter = Counter()
    per_inner_records = []

    for inner_i, (tr_idx, va_idx) in enumerate(skf_inner.split(X_outer_train, y_outer_train), start=1):
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

        # 透明化输出：每个内折都落地记录
        if out_dir_fold is not None:
            relief_df.to_csv(out_dir_fold / f"inner{inner_i}_relief_ranking.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame({"selected_feature": selected}).to_csv(
                out_dir_fold / f"inner{inner_i}_selected_after_rfe_corr.csv",
                index=False,
                encoding="utf-8-sig",
            )
            if len(drop_df) > 0:
                drop_df.to_csv(out_dir_fold / f"inner{inner_i}_spearman_drop_log.csv", index=False, encoding="utf-8-sig")

        per_inner_records.append({
            "inner_fold": inner_i,
            "n_selected": len(selected),
            "selected_features": "|".join(selected),
        })

    consensus = [
        f
        for f, c in feature_counter.items()
        if (c / cfg.inner_folds) > cfg.consensus_threshold
    ]

    freq_df = pd.DataFrame(
        [{"feature": f, "votes": c, "freq": c / cfg.inner_folds} for f, c in feature_counter.items()]
    ).sort_values(["votes", "feature"], ascending=[False, True]).reset_index(drop=True)

    # 若太多，截到final_max_features（按频次优先）
    if len(consensus) > cfg.final_max_features:
        rank_map = {r["feature"]: r["votes"] for _, r in freq_df.iterrows()}
        consensus = sorted(consensus, key=lambda x: (-rank_map.get(x, 0), x))[: cfg.final_max_features]

    return sorted(consensus), freq_df


def build_svm_pipeline(final_params: dict, random_state: int) -> Pipeline:
    """
    最终SVM管道（用于调参/重训/预测）
    说明：插补与标准化统一放入pipeline，保证训练与推理一致
    """
    imputer = build_imputer(random_state=random_state)
    scaler = StandardScaler()

    svm = SVC(
        kernel=final_params.get("kernel", "rbf"),
        C=float(final_params.get("C", 1.0)),
        gamma=final_params.get("gamma", "scale"),
        probability=True,
        class_weight="balanced",
        random_state=random_state,
    )

    pipe = Pipeline([
        ("imputer", imputer),
        ("scaler", scaler),
        ("svm", svm),
    ])
    return pipe


def tune_svm_on_outer_train(
    X_train_sel: pd.DataFrame,
    y_train: np.ndarray,
    cfg: Config,
    random_state: int,
) -> Tuple[dict, pd.DataFrame]:
    """
    在外层训练集上进行SVM调参（内层CV）
    """
    pipe = build_svm_pipeline(
        final_params={"kernel": "rbf", "C": 1.0, "gamma": "scale"},
        random_state=random_state,
    )

    c_candidates = sorted(float(x) for x in cfg.svm_c_grid if float(x) > 0)
    if len(c_candidates) == 0:
        raise ValueError("svm_c_grid 需至少包含一个正数")
    if len(c_candidates) == 1:
        c_dist: Union[List[float], object] = [c_candidates[0]]
    else:
        c_dist = loguniform(c_candidates[0], c_candidates[-1])

    param_distributions = {
        "svm__C": c_dist,
        "svm__gamma": cfg.svm_gamma_grid,
        "svm__kernel": ["rbf"],
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

    best_params = {
        "kernel": rs.best_params_["svm__kernel"],
        "C": float(rs.best_params_["svm__C"]),
        "gamma": rs.best_params_["svm__gamma"],
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
    """
    将每折最优参数聚合为最终锁定参数：
    - 数值参数用中位数
    - 离散参数用众数
    """
    # kernel
    kernel_mode = best_params_per_fold["kernel"].mode().iloc[0]

    # C
    C_med = float(best_params_per_fold["C"].median())

    # gamma 可能是字符串或数字：先转字符串再取众数，避免混合类型导致 mode 异常
    gm = best_params_per_fold["gamma"].astype(str).mode()
    gamma_mode: Union[str, float]
    if len(gm) == 0:
        gamma_mode = "scale"
    else:
        g0 = gm.iloc[0]
        if g0.lower() in ("scale", "auto"):
            gamma_mode = g0.lower()
        else:
            try:
                gamma_mode = float(g0)
            except ValueError:
                gamma_mode = g0

    return {"kernel": kernel_mode, "C": C_med, "gamma": gamma_mode}


def load_manual_features(path: str) -> List[str]:
    """
    手工闸门：
    支持两种格式：
    1) yaml:
       features:
         - Alb
         - CRP
    或直接列表:
       - Alb
       - CRP
    2) txt: 每行一个特征名
    """
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
        feats = [str(x).strip() for x in feats if str(x).strip()]
        return feats

    # txt
    with open(p, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    feats = [x for x in lines if x and not x.startswith("#")]
    return feats


# ==========================
# 阶段A：全自动开发 + 透明化输出
# ==========================
def run_stage_a(X_df: pd.DataFrame, y: np.ndarray, cfg: Config, out_dir: Path):
    print(">>> 阶段A开始：嵌套CV + 特征共识 + 调参 + 内部评估")

    skf_outer = StratifiedKFold(
        n_splits=cfg.outer_folds,
        shuffle=True,
        random_state=cfg.random_state,
    )

    fold_metrics = []
    fold_best_params = []
    outer_feature_sets = []
    oof_records = []

    for fold_i, (tr_idx, va_idx) in enumerate(skf_outer.split(X_df, y), start=1):
        fold_dir = out_dir / f"outer_fold_{fold_i}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        X_tr = X_df.iloc[tr_idx].copy()
        y_tr = y[tr_idx]
        X_va = X_df.iloc[va_idx].copy()
        y_va = y[va_idx]

        # 1) 内层cnCV得到本外层折的共识特征
        consensus_feats, inner_freq_df = inner_consensus_features(
            X_outer_train=X_tr,
            y_outer_train=y_tr,
            cfg=cfg,
            seed_offset=fold_i * 1000,
            out_dir_fold=fold_dir,
        )
        pd.DataFrame({"feature": consensus_feats}).to_csv(
            fold_dir / "consensus_features_this_outer_fold.csv", index=False, encoding="utf-8-sig"
        )
        inner_freq_df.to_csv(fold_dir / "inner_feature_frequency.csv", index=False, encoding="utf-8-sig")
        outer_feature_sets.append(consensus_feats)

        # 2) 在本外层训练集上，基于共识特征做SVM调参
        X_tr_sel = X_tr[consensus_feats].copy()
        X_va_sel = X_va[consensus_feats].copy()

        best_params, cv_results = tune_svm_on_outer_train(
            X_train_sel=X_tr_sel,
            y_train=y_tr,
            cfg=cfg,
            random_state=cfg.random_state + fold_i,
        )
        cv_results.to_csv(fold_dir / "gridsearch_cv_results.csv", index=False, encoding="utf-8-sig")

        # 3) 用本折最优参数重训并评估外层验证折
        pipe = build_svm_pipeline(best_params, random_state=cfg.random_state + fold_i)
        pipe.fit(X_tr_sel, y_tr)
        y_prob = pipe.predict_proba(X_va_sel)[:, 1]

        m = evaluate_binary(y_va, y_prob, threshold=0.5)
        m["outer_fold"] = fold_i
        m["n_features"] = len(consensus_feats)
        fold_metrics.append(m)

        fold_best_params.append({
            "outer_fold": fold_i,
            "kernel": best_params["kernel"],
            "C": best_params["C"],
            "gamma": best_params["gamma"],
            "best_inner_auc": best_params["best_inner_auc"],
            "n_features": len(consensus_feats),
        })

        # 保存OOF预测
        for idx, prob, yt in zip(va_idx, y_prob, y_va):
            oof_records.append(
                {"row_index": int(idx), "y_true": int(yt), "y_prob": float(prob), "outer_fold": fold_i}
            )

    # 4) 外层结果汇总
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

    # 5) 基于外层折特征集合做全局频率统计（透明输出）
    feat_counter = Counter()
    for feats in outer_feature_sets:
        feat_counter.update(feats)

    freq_rows = []
    for f, c in feat_counter.items():
        freq_rows.append({
            "feature": f,
            "votes": c,
            "freq": c / cfg.outer_folds,
        })
    freq_df = pd.DataFrame(freq_rows).sort_values(["votes", "feature"], ascending=[False, True]).reset_index(drop=True)
    freq_df.to_csv(out_dir / "feature_frequency.csv", index=False, encoding="utf-8-sig")

    # 6) 自动锁定特征：freq = votes/outer_folds，保留 freq > consensus_threshold（默认 0.6）；仅上限截断
    auto_feats = freq_df.loc[
        (freq_df["votes"] / cfg.outer_folds) > cfg.consensus_threshold,
        "feature",
    ].tolist()

    if len(auto_feats) > cfg.final_max_features:
        auto_feats = auto_feats[: cfg.final_max_features]

    with open(out_dir / "auto_locked_features.txt", "w", encoding="utf-8") as f:
        for x in auto_feats:
            f.write(f"{x}\n")

    # 7) 参数自动锁定
    locked_params = aggregate_locked_params(best_params_df)
    save_json(locked_params, out_dir / "locked_params.json")

    # 保存阶段A元信息
    stage_a_meta = {
        "consensus_threshold": cfg.consensus_threshold,
        "outer_folds": cfg.outer_folds,
        "inner_folds": cfg.inner_folds,
        "relief_top_k": cfg.relief_top_k,
        "rfe_n_features": cfg.rfe_n_features,
        "rfe_step": cfg.rfe_step,
        "corr_threshold": cfg.corr_threshold,
        "final_max_features": cfg.final_max_features,
        "has_skrebate": HAS_SKREBATE,
        "has_xgboost": HAS_XGB,
        "auto_locked_feature_count": len(auto_feats),
        "hyperparam_search": {
            "method": "RandomizedSearchCV",
            "n_iter": cfg.random_search_n_iter,
            "scoring": "roc_auc",
            "param_distributions": {
                "svm__C": {"type": "loguniform", "low": min(cfg.svm_c_grid), "high": max(cfg.svm_c_grid)},
                "svm__gamma": cfg.svm_gamma_grid,
                "svm__kernel": ["rbf"],
            },
        },
    }
    save_json(stage_a_meta, out_dir / "stage_a_meta.json")

    print(">>> 阶段A完成")
    print(f"    内部CV AUC(mean±sd): {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}")
    print(f"    自动锁定特征数: {len(auto_feats)}")
    print(f"    结果目录: {out_dir}")


# ==========================
# 阶段B：可选人工闸门 + 最终模型
# ==========================
def run_stage_b(X_df: pd.DataFrame, y: np.ndarray, cfg: Config, out_dir: Path):
    print(">>> 阶段B开始：加载锁定结果 -> 可选人工覆盖 -> 全开发集训练最终模型")

    # 必需文件检查
    auto_feat_path = out_dir / "auto_locked_features.txt"
    locked_params_path = out_dir / "locked_params.json"
    if not auto_feat_path.exists() or not locked_params_path.exists():
        raise FileNotFoundError(
            "缺少阶段A输出文件，请先运行阶段A，或确保目录中存在 auto_locked_features.txt + locked_params.json"
        )

    # 1) 读取自动特征和参数
    auto_feats = [line.strip() for line in auto_feat_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    with open(locked_params_path, "r", encoding="utf-8") as f:
        locked_params = json.load(f)

    # 2) 可选人工闸门（手工覆盖特征）
    # -------------------------
    # 人为操作说明：
    # 你可以创建 manual_features.yaml / txt 来覆盖自动特征，适用于临床冗余变量人工裁决（如Alb与A/G冲突）
    # 若不提供此文件，则直接使用auto_locked_features
    # -------------------------
    used_feature_source = "auto_locked_features"
    final_feats = auto_feats

    if cfg.manual_feature_file:
        manual_feats = load_manual_features(cfg.manual_feature_file)
        # 只保留在当前数据存在的特征，避免拼写错误导致崩溃
        manual_feats = [f for f in manual_feats if f in X_df.columns]
        if len(manual_feats) == 0:
            raise ValueError("manual feature file 已提供，但没有一个特征名与数据列匹配")
        final_feats = sorted(manual_feats)
        used_feature_source = f"manual_override({cfg.manual_feature_file})"

    # 3) 全开发集训练最终模型
    X_sel = X_df[final_feats].copy()
    final_pipe = build_svm_pipeline(locked_params, random_state=cfg.random_state + 777)
    final_pipe.fit(X_sel, y)

    # 训练集内演示指标（仅参考，不能代替外部验证）
    y_prob_train = final_pipe.predict_proba(X_sel)[:, 1]
    train_metrics = evaluate_binary(y, y_prob_train, threshold=0.5)

    # 4) 保存最终模型与配置
    joblib.dump(final_pipe, out_dir / "final_model.joblib")
    with open(out_dir / "final_features_used.txt", "w", encoding="utf-8") as f:
        for c in final_feats:
            f.write(f"{c}\n")

    final_meta = {
        "feature_source": used_feature_source,
        "n_final_features": len(final_feats),
        "final_params": locked_params,
        "train_metrics_demo": train_metrics,  # 仅演示
    }
    save_json(final_meta, out_dir / "final_model_meta.json")

    # 5) 可选外部验证
    if cfg.external_path:
        print(">>> 检测到 external_path，开始外部验证")
        X_ext, y_ext, ext_feats = load_data(cfg.external_path, cfg.id_col, cfg.label_col)

        # 确保外部数据有全部最终特征
        miss = [c for c in final_feats if c not in X_ext.columns]
        if len(miss) > 0:
            raise ValueError(f"外部数据缺少最终特征列: {miss}")

        X_ext_sel = X_ext[final_feats].copy()
        y_prob_ext = final_pipe.predict_proba(X_ext_sel)[:, 1]
        ext_metrics = evaluate_binary(y_ext, y_prob_ext, threshold=0.5)

        pd.DataFrame({
            "y_true": y_ext.astype(int),
            "y_prob": y_prob_ext.astype(float),
        }).to_csv(out_dir / "external_predictions.csv", index=False, encoding="utf-8-sig")

        save_json(ext_metrics, out_dir / "external_metrics.json")
        print(f">>> 外部验证AUC: {ext_metrics['auc']:.4f}")

    print(">>> 阶段B完成")
    print(f"    最终特征数: {len(final_feats)}")
    print(f"    特征来源: {used_feature_source}")
    print(f"    结果目录: {out_dir}")


# ==========================
# 命令行入口
# ==========================
def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="cnCV + ReliefF + RFE(XGB) + SVM for UC vs non-UC")

    parser.add_argument("--data-path", type=str, default=r"F:\KeTi\Project\Data\ATrain.csv",
                        help="训练开发数据CSV路径（含No和Disease）")
    parser.add_argument("--external-path", type=str, default=None,
                        help="可选外部验证CSV路径（列结构需一致）")
    parser.add_argument("--output-dir", type=str, default=r"F:\KeTi\Project\outputs",
                        help="输出目录根路径（阶段A/all 会在此下新建 run_时间戳；阶段B可配合 --run-dir）")
    parser.add_argument(
        "--stage",
        type=str,
        default="A",
        choices=["A", "B", "all"],
        help="A=仅阶段A（默认，适合先跑通）; B=仅阶段B; all=两阶段一次跑完（最慢）",
    )
    parser.add_argument("--id-col", type=str, default="No")
    parser.add_argument("--label-col", type=str, default="Disease")

    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--relief-top-k", type=int, default=30,
                        help="ReliefF初筛上限（当前默认30）")
    parser.add_argument("--rfe-n-features", type=int, default=20,
                        help="RFE目标特征数（当前默认20）")
    parser.add_argument("--rfe-step", type=int, default=3,
                        help="RFE每轮删除特征步长（默认3，提速明显）")
    parser.add_argument("--corr-threshold", type=float, default=0.8,
                        help="Spearman去相关阈值")
    parser.add_argument(
        "--consensus-threshold",
        type=float,
        default=0.6,
        help="共识用严格不等式：保留 freq=votes/n_splits > 该值；默认0.6，外层5折下仅 4/5 或 5/5 纳入",
    )
    parser.add_argument("--final-max-features", type=int, default=20,
                        help="自动锁定最多特征数（无下限补齐）")

    # 窄网格：避免过宽搜索，符合你的要求
    parser.add_argument("--svm-c-grid", type=str, default="1,3",
                        help="逗号分隔")
    parser.add_argument("--svm-gamma-grid", type=str, default="scale,0.1",
                        help="逗号分隔，可混合字符串/数值")
    parser.add_argument(
        "--random-search-n-iter",
        type=int,
        default=8,
        help="内层 RandomizedSearchCV 的 n_iter（默认 8）",
    )

    parser.add_argument("--manual-feature-file", type=str, default=None,
                        help="可选人工闸门文件：yaml/txt；用于覆盖auto_locked_features")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help='阶段B专用：指向某次阶段A生成的 outputs\\run_时间戳 绝对路径；填 auto 则从 --output-dir 读取 LAST_RUN_DIR.txt',
    )

    args = parser.parse_args()

    # 解析网格
    c_grid = [float(x.strip()) for x in args.svm_c_grid.split(",") if x.strip()]

    gamma_grid: List[Union[str, float]] = []
    for x in args.svm_gamma_grid.split(","):
        xx = x.strip()
        if not xx:
            continue
        if xx.lower() in ["scale", "auto"]:
            gamma_grid.append(xx.lower())
        else:
            gamma_grid.append(float(xx))

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
        svm_c_grid=c_grid,
        svm_gamma_grid=gamma_grid,
        random_search_n_iter=args.random_search_n_iter,
        manual_feature_file=args.manual_feature_file,
        run_dir=args.run_dir,
    )


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    cfg = parse_args()
    base_out = Path(cfg.output_dir)

    # 输出目录策略：
    # - stage=A 或 all：在 output-dir 下创建新 run_时间戳
    # - stage=B：必须指向某次 run_ 目录；推荐 --run-dir auto 读取 LAST_RUN_DIR.txt
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
            # 兼容旧用法：直接把 --output-dir 指到 run_ 目录
            out_dir = Path(cfg.output_dir)
        if not out_dir.exists():
            raise FileNotFoundError(f"阶段B指定的 run 目录不存在: {out_dir}")

    print(">>> 读取数据")
    X_df, y, feature_names = load_data(cfg.data_path, cfg.id_col, cfg.label_col)
    print(f"    样本数: {len(y)}; 特征数: {len(feature_names)}; 阳性率: {y.mean():.4f}")
    print(f"    HAS_SKREBATE={HAS_SKREBATE}, HAS_XGB={HAS_XGB}")

    if cfg.stage in ["A", "all"]:
        run_stage_a(X_df, y, cfg, out_dir)
        # 写入指针文件，便于下次 --stage B --run-dir auto
        write_last_run_pointer(base_out, out_dir)
        print(f">>> 已写入最新 run 目录指针: {(base_out / 'LAST_RUN_DIR.txt').resolve()}")
        print(f"    内容为: {out_dir.resolve()}")

    if cfg.stage in ["B", "all"]:
        # stage=all 时直接沿用同一out_dir；stage=B 需要 out_dir 指向含阶段A产物的 run_ 目录
        run_stage_b(X_df, y, cfg, out_dir)

    print(">>> 全部完成")


if __name__ == "__main__":
    main()