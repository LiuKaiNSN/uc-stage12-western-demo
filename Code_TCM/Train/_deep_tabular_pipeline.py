# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import random
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

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
class CommonConfig:
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
    manual_feature_file: Optional[str]
    run_dir: Optional[str]
    # fixed training budget
    batch_size: int
    lr: float
    weight_decay: float
    max_epochs: int
    early_stopping_patience: int


class ReliefFTopK:
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
            scores = np.var(X, axis=0)
        return scores

    def topk_indices(self, scores: np.ndarray) -> np.ndarray:
        k = min(self.k, len(scores))
        idx = np.argsort(-scores)[:k]
        return np.sort(idx)


def make_output_dir(base_dir: str, model_tag: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(base_dir) / f"run_{model_tag}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(obj: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_last_run_pointer(base_output_dir: Path, run_dir: Path) -> None:
    base_output_dir.mkdir(parents=True, exist_ok=True)
    (base_output_dir / "LAST_RUN_DIR.txt").write_text(str(run_dir.resolve()), encoding="utf-8")


def load_data(data_path: str, id_col: str, label_col: str) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    df = pd.read_csv(data_path)
    if id_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"缺少必要列: id_col={id_col}, label_col={label_col}")
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


def spearman_prune(X_df_selected: pd.DataFrame, y: np.ndarray, corr_threshold: float = 0.8) -> Tuple[List[str], List[dict]]:
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
    cfg: CommonConfig,
    random_state: int,
) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    imputer = build_imputer(random_state=random_state)
    X_imp = imputer.fit_transform(X_train_df.values)
    all_features = list(X_train_df.columns)

    relief = ReliefFTopK(k=cfg.relief_top_k, n_neighbors=5, discrete_threshold=10)
    scores = relief.fit_score(X_imp, y_train)
    top_idx = relief.topk_indices(scores)
    top_features = [all_features[i] for i in top_idx]
    relief_df = pd.DataFrame({"feature": all_features, "relief_score": scores}).sort_values(
        "relief_score", ascending=False
    )

    X_top = X_imp[:, top_idx]
    rfe_est = get_rfe_estimator(random_state=random_state)
    n_select = min(cfg.rfe_n_features, X_top.shape[1])
    rfe = RFE(estimator=rfe_est, n_features_to_select=n_select, step=cfg.rfe_step)
    rfe.fit(X_top, y_train)
    rfe_local_idx = np.where(rfe.support_)[0]
    rfe_features = [top_features[i] for i in rfe_local_idx]

    X_rfe_df = pd.DataFrame(X_imp, columns=all_features)[rfe_features]
    kept_features, drop_logs = spearman_prune(
        X_df_selected=X_rfe_df,
        y=y_train,
        corr_threshold=cfg.corr_threshold,
    )
    return kept_features, relief_df.reset_index(drop=True), pd.DataFrame(drop_logs)


def inner_consensus_features(
    X_outer_train: pd.DataFrame,
    y_outer_train: np.ndarray,
    cfg: CommonConfig,
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
                drop_df.to_csv(out_dir_fold / f"inner{inner_i}_spearman_drop_log.csv", index=False, encoding="utf-8-sig")

    consensus = [f for f, c in feature_counter.items() if (c / cfg.inner_folds) > cfg.consensus_threshold]
    freq_df = (
        pd.DataFrame([{"feature": f, "votes": c, "freq": c / cfg.inner_folds} for f, c in feature_counter.items()])
        .sort_values(["votes", "feature"], ascending=[False, True])
        .reset_index(drop=True)
    )
    if len(consensus) > cfg.final_max_features:
        rank_map = {r["feature"]: r["votes"] for _, r in freq_df.iterrows()}
        consensus = sorted(consensus, key=lambda x: (-rank_map.get(x, 0), x))[: cfg.final_max_features]
    return sorted(consensus), freq_df


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, n_layers: int = 2, dropout: float = 0.2, activation: str = "gelu"):
        super().__init__()
        act_cls = nn.GELU if activation.lower() == "gelu" else nn.ReLU
        layers: List[nn.Module] = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), act_cls(), nn.Dropout(dropout)])
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)).squeeze(1)


class CrossLayer(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(input_dim) * 0.02)
        self.b = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        dot = torch.sum(xl * self.w[None, :], dim=1, keepdim=True)
        return xl + x0 * (dot + self.b[None, :])


class DCNV2Classifier(nn.Module):
    def __init__(self, input_dim: int, num_cross_layers: int = 2, deep_layers: int = 1, deep_hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.cross_layers = nn.ModuleList([CrossLayer(input_dim) for _ in range(num_cross_layers)])
        deep: List[nn.Module] = []
        in_dim = input_dim
        for _ in range(deep_layers):
            deep.extend([nn.Linear(in_dim, deep_hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = deep_hidden_dim
        self.deep_tower = nn.Sequential(*deep)
        self.head = nn.Linear(input_dim + in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        xl = x
        for layer in self.cross_layers:
            xl = layer(x0, xl)
        xd = self.deep_tower(x)
        return self.head(torch.cat([xl, xd], dim=1)).squeeze(1)


class NumericTokenizer(nn.Module):
    def __init__(self, n_features: int, d_model: int, token_dropout: float):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, d_model))
        self.dropout = nn.Dropout(token_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        return self.dropout(tokens)


class FTTransformerClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int = 32,
        n_blocks: int = 1,
        n_heads: int = 2,
        ffn_mult: float = 1.5,
        attn_dropout: float = 0.15,
        ff_dropout: float = 0.2,
        token_dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.tokenizer = NumericTokenizer(n_features=n_features, d_model=d_model, token_dropout=token_dropout)
        ff_dim = int(d_model * ffn_mult)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=attn_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_blocks)
        self.ff_dropout = nn.Dropout(ff_dropout)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        enc = self.encoder(tokens)
        pooled = self.ff_dropout(enc.mean(dim=1))
        return self.head(pooled).squeeze(1)


class TabTransformerClassifier(nn.Module):
    """连续特征版本的轻量 TabTransformer（列嵌入 + TransformerEncoder）。"""

    def __init__(
        self,
        n_features: int,
        d_model: int = 32,
        n_blocks: int = 1,
        n_heads: int = 2,
        ffn_mult: float = 1.5,
        attn_dropout: float = 0.15,
        ff_dropout: float = 0.2,
        token_dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.value_proj = nn.Linear(1, d_model)
        self.column_embed = nn.Embedding(n_features, d_model)
        self.token_dropout = nn.Dropout(token_dropout)
        ff_dim = int(d_model * ffn_mult)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=attn_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_blocks)
        self.ff_dropout = nn.Dropout(ff_dropout)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.register_buffer("col_idx", torch.arange(n_features, dtype=torch.long), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x.unsqueeze(-1)  # [B, F, 1]
        tokens = self.value_proj(x1)
        col_emb = self.column_embed(self.col_idx)[None, :, :]
        tokens = self.token_dropout(tokens + col_emb)
        enc = self.encoder(tokens)
        pooled = self.ff_dropout(enc.mean(dim=1))
        return self.head(pooled).squeeze(1)


def build_model(model_name: str, input_dim: int, model_params: Dict[str, Any]) -> nn.Module:
    name = model_name.lower()
    if name == "mlp":
        return MLPClassifier(input_dim=input_dim, **model_params)
    if name == "dcnv2":
        return DCNV2Classifier(input_dim=input_dim, **model_params)
    if name == "fttransformer":
        return FTTransformerClassifier(n_features=input_dim, **model_params)
    if name == "tabtransformer":
        return TabTransformerClassifier(n_features=input_dim, **model_params)
    raise ValueError(f"Unsupported model_name: {model_name}")


def _build_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32)))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_with_early_stopping(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: CommonConfig,
    seed: int,
) -> Tuple[nn.Module, float]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    tr_loader = _build_loader(x_train, y_train, cfg.batch_size, shuffle=True)
    pos = float(y_train.sum())
    neg = float(len(y_train) - y_train.sum())
    pos_weight = torch.tensor([max(neg / max(pos, 1.0), 1.0)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_auc = -np.inf
    best_state: Optional[Dict[str, torch.Tensor]] = None
    patience = 0
    for _ in range(cfg.max_epochs):
        model.train()
        for xb, yb in tr_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        val_prob = predict_proba_torch_model(model, x_val, cfg.batch_size, device)
        val_auc = roc_auc_score(y_val, val_prob)
        if val_auc > best_auc:
            best_auc = float(val_auc)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc


def predict_proba_torch_model(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    loader = DataLoader(torch.from_numpy(x.astype(np.float32)), batch_size=batch_size, shuffle=False, drop_last=False)
    out: List[np.ndarray] = []
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            probs = torch.sigmoid(model(xb)).detach().cpu().numpy()
            out.append(probs)
    return np.concatenate(out, axis=0)


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


def fit_bundle(
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
    cfg: CommonConfig,
    random_state: int,
    model_name: str,
    model_params: Dict[str, Any],
) -> Dict[str, Any]:
    x_tr_raw, x_val_raw, y_tr, y_val = train_test_split(
        X_train_df.values,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=random_state,
    )
    imputer = build_imputer(random_state=random_state)
    x_tr_imp = imputer.fit_transform(x_tr_raw)
    x_val_imp = imputer.transform(x_val_raw)
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(x_tr_imp).astype(np.float32)
    x_val = scaler.transform(x_val_imp).astype(np.float32)

    model = build_model(model_name, input_dim=x_tr.shape[1], model_params=model_params)
    model, _ = train_with_early_stopping(
        model=model,
        x_train=x_tr,
        y_train=y_tr,
        x_val=x_val,
        y_val=y_val,
        cfg=cfg,
        seed=random_state + 17,
    )
    bundle = {
        "model_name": model_name,
        "model_params": model_params,
        "imputer": imputer,
        "scaler": scaler,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    return bundle


def predict_bundle(bundle: Dict[str, Any], X_df: pd.DataFrame, batch_size: int) -> np.ndarray:
    x_imp = bundle["imputer"].transform(X_df.values)
    x_std = bundle["scaler"].transform(x_imp).astype(np.float32)
    model = build_model(
        model_name=bundle["model_name"],
        input_dim=x_std.shape[1],
        model_params=bundle["model_params"],
    )
    model.load_state_dict(bundle["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return predict_proba_torch_model(model, x_std, batch_size=batch_size, device=device)


def evaluate_fixed_params_on_outer_train(
    X_train_sel: pd.DataFrame,
    y_train: np.ndarray,
    cfg: CommonConfig,
    random_state: int,
    model_name: str,
    model_params: Dict[str, Any],
) -> Tuple[dict, pd.DataFrame]:
    inner_cv = StratifiedKFold(
        n_splits=cfg.inner_folds,
        shuffle=True,
        random_state=random_state + 99,
    )
    split_scores: Dict[str, float] = {}
    all_scores: List[float] = []
    for i, (tr_idx, va_idx) in enumerate(inner_cv.split(X_train_sel, y_train)):
        X_tr_fold = X_train_sel.iloc[tr_idx].copy()
        y_tr_fold = y_train[tr_idx]
        X_va_fold = X_train_sel.iloc[va_idx].copy()
        y_va_fold = y_train[va_idx]
        bundle = fit_bundle(
            X_train_df=X_tr_fold,
            y_train=y_tr_fold,
            cfg=cfg,
            random_state=random_state + i * 37,
            model_name=model_name,
            model_params=model_params,
        )
        y_prob = predict_bundle(bundle, X_va_fold, batch_size=cfg.batch_size)
        auc = float(roc_auc_score(y_va_fold, y_prob))
        split_scores[f"split{i}_test_score"] = auc
        all_scores.append(auc)

    mean_auc = float(np.mean(all_scores))
    std_auc = float(np.std(all_scores))
    best_params = dict(model_params)
    best_params["best_inner_auc"] = mean_auc
    row = {
        **split_scores,
        "mean_test_score": mean_auc,
        "std_test_score": std_auc,
        "rank_test_score": 1,
        "params": json.dumps(model_params, ensure_ascii=False),
    }
    cv_results = pd.DataFrame([row])
    return best_params, cv_results


def aggregate_locked_params(best_params_per_fold: pd.DataFrame, model_params: Dict[str, Any]) -> dict:
    out = dict(model_params)
    if "best_inner_auc" in best_params_per_fold.columns:
        out["best_inner_auc_median"] = float(best_params_per_fold["best_inner_auc"].median())
    return out


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


def run_stage_a(
    X_df: pd.DataFrame,
    y: np.ndarray,
    cfg: CommonConfig,
    out_dir: Path,
    model_name: str,
    model_params: Dict[str, Any],
) -> None:
    print(">>> 阶段A开始：嵌套CV + 特征共识 + 固定轻量参数评估 + 内部评估")
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
        best_params, cv_results = evaluate_fixed_params_on_outer_train(
            X_train_sel=X_tr_sel,
            y_train=y_tr,
            cfg=cfg,
            random_state=cfg.random_state + fold_i,
            model_name=model_name,
            model_params=model_params,
        )
        cv_results.to_csv(fold_dir / "gridsearch_cv_results.csv", index=False, encoding="utf-8-sig")

        bundle = fit_bundle(
            X_train_df=X_tr_sel,
            y_train=y_tr,
            cfg=cfg,
            random_state=cfg.random_state + fold_i * 101,
            model_name=model_name,
            model_params=model_params,
        )
        y_prob = predict_bundle(bundle, X_va_sel, batch_size=cfg.batch_size)
        m = evaluate_binary(y_va, y_prob, threshold=0.5)
        m["outer_fold"] = fold_i
        m["n_features"] = len(consensus_feats)
        fold_metrics.append(m)

        row = {"outer_fold": fold_i, "best_inner_auc": best_params["best_inner_auc"], "n_features": len(consensus_feats)}
        for k, v in model_params.items():
            row[k] = v
        fold_best_params.append(row)

        for idx, prob, yt in zip(va_idx, y_prob, y_va):
            oof_records.append({"row_index": int(idx), "y_true": int(yt), "y_prob": float(prob), "outer_fold": fold_i})

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
    pd.DataFrame(oof_records).sort_values("row_index").to_csv(out_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")

    feat_counter = Counter()
    for feats in outer_feature_sets:
        feat_counter.update(feats)
    freq_df = (
        pd.DataFrame([{"feature": f, "votes": c, "freq": c / cfg.outer_folds} for f, c in feat_counter.items()])
        .sort_values(["votes", "feature"], ascending=[False, True])
        .reset_index(drop=True)
    )
    freq_df.to_csv(out_dir / "feature_frequency.csv", index=False, encoding="utf-8-sig")
    auto_feats = freq_df.loc[(freq_df["votes"] / cfg.outer_folds) > cfg.consensus_threshold, "feature"].tolist()
    if len(auto_feats) > cfg.final_max_features:
        auto_feats = auto_feats[: cfg.final_max_features]
    (out_dir / "auto_locked_features.txt").write_text("\n".join(auto_feats) + "\n", encoding="utf-8")

    locked_params = aggregate_locked_params(best_params_df, model_params=model_params)
    save_json(locked_params, out_dir / "locked_params.json")

    stage_a_meta = {
        "model": model_name,
        "consensus_threshold": cfg.consensus_threshold,
        "outer_folds": cfg.outer_folds,
        "inner_folds": cfg.inner_folds,
        "relief_top_k": cfg.relief_top_k,
        "rfe_n_features": cfg.rfe_n_features,
        "rfe_step": cfg.rfe_step,
        "corr_threshold": cfg.corr_threshold,
        "final_max_features": cfg.final_max_features,
        "hyperparam_search": {
            "method": "FixedParams",
            "n_iter": 1,
            "model_params": model_params,
        },
        "train_budget": {
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "max_epochs": cfg.max_epochs,
            "early_stopping_patience": cfg.early_stopping_patience,
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


def run_stage_b(
    X_df: pd.DataFrame,
    y: np.ndarray,
    cfg: CommonConfig,
    out_dir: Path,
    model_name: str,
    model_params: Dict[str, Any],
) -> None:
    print(">>> 阶段B开始：加载锁定结果 -> 可选人工覆盖 -> 全开发集训练最终模型")
    auto_feat_path = out_dir / "auto_locked_features.txt"
    locked_params_path = out_dir / "locked_params.json"
    if not auto_feat_path.exists() or not locked_params_path.exists():
        raise FileNotFoundError("缺少阶段A输出文件，请先运行阶段A，或确保目录中存在 auto_locked_features.txt + locked_params.json")

    auto_feats = [x.strip() for x in auto_feat_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    _ = json.loads(locked_params_path.read_text(encoding="utf-8"))  # 保留兼容读取

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
    bundle = fit_bundle(
        X_train_df=X_sel,
        y_train=y,
        cfg=cfg,
        random_state=cfg.random_state + 777,
        model_name=model_name,
        model_params=model_params,
    )
    y_prob_train = predict_bundle(bundle, X_sel, batch_size=cfg.batch_size)
    train_metrics = evaluate_binary(y, y_prob_train, threshold=0.5)

    joblib.dump(bundle, out_dir / "final_model.joblib")
    (out_dir / "final_features_used.txt").write_text("\n".join(final_feats) + "\n", encoding="utf-8")
    save_json(
        {
            "model": model_name,
            "feature_source": used_feature_source,
            "n_final_features": len(final_feats),
            "final_params": model_params,
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
        y_prob_ext = predict_bundle(bundle, X_ext_sel, batch_size=cfg.batch_size)
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


def run_full_pipeline(
    cfg: CommonConfig,
    model_name: str,
    model_params: Dict[str, Any],
    model_tag: str,
) -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    base_out = Path(cfg.output_dir)

    if cfg.stage in ["A", "all"]:
        out_dir = make_output_dir(str(base_out), model_tag=model_tag)
    else:
        if cfg.run_dir:
            if str(cfg.run_dir).strip().lower() == "auto":
                last_file = base_out / "LAST_RUN_DIR.txt"
                if not last_file.exists():
                    raise FileNotFoundError(f"未找到 {last_file}。请先成功运行一次阶段A，或改用 --run-dir 指向具体 run_ 文件夹。")
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
        run_stage_a(X_df, y, cfg, out_dir, model_name=model_name, model_params=model_params)
        write_last_run_pointer(base_out, out_dir)
        print(f">>> 已写入最新 run 目录指针: {(base_out / 'LAST_RUN_DIR.txt').resolve()}")
        print(f"    内容为: {out_dir.resolve()}")

    if cfg.stage in ["B", "all"]:
        run_stage_b(X_df, y, cfg, out_dir, model_name=model_name, model_params=model_params)
    print(">>> 全部完成")
