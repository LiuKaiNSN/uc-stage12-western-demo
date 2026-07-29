# -*- coding: utf-8 -*-
"""
Stage1 二分类深度学习外部验证：(IE+IBS) vs (UC+CD+IC+CRC)

加载阶段 B 的 ``final_model.joblib``（torch bundle dict）+ ``final_features_used.txt``，
在外部 CSV 上评估。验证策略、分层与输出结构与 ``external_validate_stage1.py`` 一致。

与 ``external_validate_stage_dl.py``（Stage2 DL）分离，避免 Stage1/Stage2 任务混淆。

支持的 bundle 来源：MLP.py / DCNV2.py / FTTransformer.py / TabTransformer.py
（``_deep_tabular_pipeline.predict_bundle``）。

正类：``Disease=1``（UC+CD+IC+CRC 组）；bundle 输出 P(1) 一维概率，脚本内构造 prob_0 / prob_1。

启动示例（PowerShell）::

  python "F:\\KeTi\\Project\\Script\\external_validate_stage1_dl.py" ^
    --model-run-dir "F:\\KeTi\\Project\\TCM\\output\\Stage1\\7_run_mlp_20260607_222237" ^
    --external-csv "F:\\KeTi\\Project\\TCM\\data\\ATest-Stage1.csv"

请在训练该模型时所用的同一 Python 环境中运行（需 torch）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from _deep_tabular_pipeline import predict_bundle  # noqa: E402
from external_validate_stage1 import (  # noqa: E402
    TYPE_IE_IBS,
    TYPE_INFLAM,
    build_feature_matrix,
    build_pairwise_tables,
    build_type_slices,
    encode_labels,
    subgroup_block,
)

DL_MODEL_NAMES = frozenset({"MLP", "DCNV2", "FTTransformer", "TabTransformer"})


def _load_torch_bundle(model_path: Path) -> Dict[str, Any]:
    obj = joblib.load(model_path)
    if not isinstance(obj, dict):
        raise TypeError(
            f"{model_path.name} 不是深度学习 bundle（dict），而是 {type(obj).__name__}。"
            "请对 sklearn Pipeline 模型使用 external_validate_stage1.py。"
        )
    required = {"model_name", "model_params", "imputer", "scaler", "state_dict"}
    missing = required - set(obj.keys())
    if missing:
        raise ValueError(f"bundle 缺少字段: {sorted(missing)}")
    return obj


def _resolve_class_names_and_mapping(
    run_dir: Path,
    class_path: Path,
) -> Tuple[List[str], Dict[str, int]]:
    """Stage1 二分类默认 0=IE+IBS, 1=UC+CD+IC+CRC；可选 class_mapping.json。"""
    default_names = ["0", "1"]
    default_map = {"0": 0, "1": 1}
    if not class_path.exists():
        return default_names, default_map

    class_map = json.loads(class_path.read_text(encoding="utf-8"))
    jti = {str(k): int(v) for k, v in class_map["class_to_idx"].items()}
    if sorted(jti.values()) != [0, 1]:
        raise ValueError(f"class_mapping.json 须为二分类 idx 0/1，当前: {jti}")
    idx_to_class = {int(v): str(k) for k, v in jti.items()}
    class_names = [idx_to_class[0], idx_to_class[1]]
    return class_names, jti


def _resolve_batch_size(run_dir: Path, cli_batch_size: int | None) -> int:
    if cli_batch_size is not None and cli_batch_size > 0:
        return int(cli_batch_size)
    meta_path = run_dir / "stage_a_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        tb = meta.get("train_budget") or {}
        bs = tb.get("batch_size")
        if bs is not None:
            return int(bs)
    return 32


def _predict_proba_binary(bundle: Dict[str, Any], X: pd.DataFrame, batch_size: int) -> np.ndarray:
    """返回 shape (n, 2)：列0=P(IE+IBS 组), 列1=P(UC+CD+IC+CRC 组)。"""
    raw = predict_bundle(bundle, X, batch_size=batch_size)
    p_pos = np.asarray(raw, dtype=float).reshape(-1)
    if len(p_pos) != len(X):
        raise ValueError(f"预测长度 {len(p_pos)} 与样本数 {len(X)} 不一致")
    p_pos = np.clip(p_pos, 0.0, 1.0)
    p_neg = 1.0 - p_pos
    return np.column_stack([p_neg, p_pos])


def main() -> None:
    p = argparse.ArgumentParser(
        description="Stage1 二分类外部验证（深度学习 bundle：MLP/DCNV2/FTTransformer/TabTransformer）"
    )
    p.add_argument(
        "--model-run-dir",
        type=str,
        required=True,
        help=r"阶段 B 产出目录（含 final_model.joblib + final_features_used.txt）",
    )
    p.add_argument("--external-csv", type=str, required=True, help="外部验证集 CSV（如 ATest-Stage1.csv）")
    p.add_argument("--id-col", type=str, default="No")
    p.add_argument("--source-col", type=str, default="Source")
    p.add_argument("--type-col", type=str, default="Type")
    p.add_argument("--agerange-col", type=str, default="Agerange")
    p.add_argument("--label-col", type=str, default="Disease")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="默认: model-run-dir/external_validation_stage1/",
    )
    p.add_argument("--batch-size", type=int, default=None, help="默认读 stage_a_meta.json 或 32")
    p.add_argument("--skip-type-slices", action="store_true")
    args = p.parse_args()

    run_dir = Path(args.model_run_dir).resolve()
    ext_path = Path(args.external_csv).resolve()
    out_dir = Path(args.output_dir).resolve() if args.output_dir else (run_dir / "external_validation_stage1")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / "final_model.joblib"
    feats_path = run_dir / "final_features_used.txt"
    class_path = run_dir / "class_mapping.json"
    if not model_path.exists() or not feats_path.exists():
        raise FileNotFoundError(f"缺少必要文件: {model_path} 或 {feats_path}")

    bundle = _load_torch_bundle(model_path)
    model_name = str(bundle.get("model_name", ""))
    if model_name and model_name not in DL_MODEL_NAMES:
        print(f"    [提示] bundle.model_name={model_name!r} 不在已知 DL 列表 {sorted(DL_MODEL_NAMES)}，仍尝试预测。")

    batch_size = _resolve_batch_size(run_dir, args.batch_size)
    class_names, class_to_idx = _resolve_class_names_and_mapping(run_dir, class_path)

    final_feats = [x.strip() for x in feats_path.read_text(encoding="utf-8").splitlines() if x.strip()]

    df = pd.read_csv(ext_path)
    need = {args.id_col, args.source_col, args.type_col, args.agerange_col, args.label_col}
    if need - set(df.columns):
        raise ValueError(f"缺少列: {sorted(need - set(df.columns))}")

    X, feat_miss = build_feature_matrix(df, final_feats)
    if feat_miss:
        print(
            f"    [警告] 外部 CSV 缺少 {len(feat_miss)} 个训练特征列，已以 NaN 占位（imputer 会填补）: "
            f"{feat_miss[:10]}{'...' if len(feat_miss) > 10 else ''}"
        )

    y, bad = encode_labels(df[args.label_col].values, class_to_idx)
    if any(v == -1 for v in y):
        raise ValueError(f"标签无法映射: {sorted(set(bad))}")

    y_prob_full = _predict_proba_binary(bundle, X, batch_size=batch_size)
    y_prob_pos = y_prob_full[:, 1]
    y_pred = (y_prob_pos >= args.threshold).astype(int)
    pair_name = f"{class_names[0]}_vs_{class_names[1]}"

    report: Dict[str, Any] = {
        "task": "stage1_binary",
        "model_type": "deep_learning_torch_bundle",
        "model_name": model_name or "unknown",
        "batch_size": batch_size,
        "model_run_dir": str(run_dir),
        "external_csv": str(ext_path),
        "label_col": args.label_col,
        "type_mapping": {
            "IE_IBS_Disease_0": sorted(TYPE_IE_IBS),
            "INFLAM_Disease_1": sorted(TYPE_INFLAM),
        },
        "class_names": class_names,
        "positive_score": f"prob_{class_names[1]} (UC+CD+IC+CRC 组)",
        "overall": subgroup_block("全验证集", y, y_prob_pos, class_names, args.threshold),
        "by_source": {},
        "by_agerange": {},
    }

    for s in sorted(df[args.source_col].astype(str).unique()):
        m = df[args.source_col].astype(str) == s
        report["by_source"][s] = subgroup_block(f"Source={s}", y[m], y_prob_pos[m], class_names, args.threshold)

    for a in sorted(df[args.agerange_col].unique(), key=lambda x: (str(type(x)), str(x))):
        m = df[args.agerange_col] == a
        report["by_agerange"][str(a)] = subgroup_block(
            f"Agerange={a}", y[m], y_prob_pos[m], class_names, args.threshold
        )

    strata_rows = [
        {
            "stratum_id": "overall",
            "stratum_label": "全验证集",
            "n_samples": report["overall"]["n_samples"],
            "pairwise_binary": report["overall"].get("pairwise_binary", []),
        }
    ]
    for s in sorted(report["by_source"]):
        b = report["by_source"][s]
        strata_rows.append(
            {
                "stratum_id": f"source__{s}",
                "stratum_label": f"Source={s}",
                "n_samples": b["n_samples"],
                "pairwise_binary": b.get("pairwise_binary", []),
            }
        )
    for k in sorted(report["by_agerange"], key=lambda x: (len(str(x)), str(x))):
        b = report["by_agerange"][k]
        strata_rows.append(
            {
                "stratum_id": f"agerange__{k}",
                "stratum_label": f"Agerange={k}",
                "n_samples": b["n_samples"],
                "pairwise_binary": b.get("pairwise_binary", []),
            }
        )

    long_df, wide_df = build_pairwise_tables(strata_rows, pair_name)
    long_df.to_csv(out_dir / "external_metrics_long.csv", index=False, encoding="utf-8-sig")
    wide_df.to_csv(out_dir / "external_metrics_wide.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(
        {
            args.id_col: df[args.id_col],
            args.source_col: df[args.source_col],
            args.type_col: df[args.type_col],
            args.agerange_col: df[args.agerange_col],
            args.label_col: df[args.label_col],
            "y_true": y,
            "y_pred": y_pred,
            f"prob_{class_names[0]}": y_prob_full[:, 0],
            f"prob_{class_names[1]}": y_prob_full[:, 1],
        }
    ).to_csv(out_dir / "external_predictions.csv", index=False, encoding="utf-8-sig")

    if not args.skip_type_slices:
        td = out_dir / "type_slices"
        type_report, _, _ = build_type_slices(
            df,
            y,
            y_prob_pos,
            y_prob_full,
            class_names,
            args.threshold,
            args.id_col,
            args.source_col,
            args.agerange_col,
            args.label_col,
            args.type_col,
            td,
        )
        report["type_slices"] = type_report

    report["validation_note"] = (
        "Stage1 深度学习二分类外部验证；bundle 经 predict_bundle 推理；"
        "分层与输出结构与 external_validate_stage1.py 对齐。"
    )

    with open(out_dir / "external_validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(">>> Stage1 深度学习外部验证完成")
    print(f"    模型: {model_name or 'unknown'} | batch_size={batch_size}")
    print(f"    输出目录: {out_dir}")
    m = report["overall"]["metrics"]
    print(f"    全验证集 n={report['overall']['n_samples']} auc={m['auc']:.4f} acc={m['acc']:.4f}")
    if not args.skip_type_slices:
        print(f"    Type 分层: {out_dir / 'type_slices'}")
        print(f"    各病种识别简表: {out_dir / 'type_slices' / 'type_per_disease_recognition.csv'}")


if __name__ == "__main__":
    main()
