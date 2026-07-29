# -*- coding: utf-8 -*-
"""
UC vs 非UC: cnCV + ReliefF + RFE(XGB) + TabTransformer(固定轻量) 全流程脚本
"""

from __future__ import annotations

import argparse

from _deep_tabular_pipeline import CommonConfig, run_full_pipeline


FIXED_MODEL_PARAMS = {
    "d_model": 48,
    "n_blocks": 1,
    "n_heads": 2,
    "ffn_mult": 2.0,
    "attn_dropout": 0.15,
    "ff_dropout": 0.2,
    "token_dropout": 0.1,
}


def parse_args() -> CommonConfig:
    p = argparse.ArgumentParser(description="cnCV + ReliefF + RFE(XGB) + TabTransformer (fixed lightweight)")
    p.add_argument("--data-path", type=str, default=r"F:\KeTi\Project\Data\ATrain.csv")
    p.add_argument("--external-path", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=r"F:\KeTi\Project\outputs")
    p.add_argument("--stage", type=str, default="A", choices=["A", "B", "all"])
    p.add_argument("--id-col", type=str, default="No")
    p.add_argument("--label-col", type=str, default="Disease")
    p.add_argument("--outer-folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--relief-top-k", type=int, default=30)
    p.add_argument("--rfe-n-features", type=int, default=20)
    p.add_argument("--rfe-step", type=int, default=3)
    p.add_argument("--corr-threshold", type=float, default=0.8)
    p.add_argument("--consensus-threshold", type=float, default=0.6)
    p.add_argument("--final-max-features", type=int, default=20)
    p.add_argument("--manual-feature-file", type=str, default=None)
    p.add_argument("--run-dir", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-epochs", type=int, default=120)
    p.add_argument("--early-stopping-patience", type=int, default=12)
    a = p.parse_args()
    return CommonConfig(
        data_path=a.data_path,
        external_path=a.external_path,
        output_dir=a.output_dir,
        stage=a.stage,
        id_col=a.id_col,
        label_col=a.label_col,
        outer_folds=a.outer_folds,
        inner_folds=a.inner_folds,
        random_state=a.random_state,
        relief_top_k=a.relief_top_k,
        rfe_n_features=a.rfe_n_features,
        rfe_step=a.rfe_step,
        corr_threshold=a.corr_threshold,
        consensus_threshold=a.consensus_threshold,
        final_max_features=a.final_max_features,
        manual_feature_file=a.manual_feature_file,
        run_dir=a.run_dir,
        batch_size=a.batch_size,
        lr=a.lr,
        weight_decay=a.weight_decay,
        max_epochs=a.max_epochs,
        early_stopping_patience=a.early_stopping_patience,
    )


def main() -> None:
    cfg = parse_args()
    run_full_pipeline(
        cfg=cfg,
        model_name="TabTransformer",
        model_params=FIXED_MODEL_PARAMS,
        model_tag="tabtransformer",
    )


if __name__ == "__main__":
    main()
