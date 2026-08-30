# -*- coding: utf-8 -*-
"""
二分类（Stage2）深度学习外部验证：加载阶段 B 的 ``final_model.joblib``（torch bundle dict）
+ ``final_features_used.txt``，在外部 CSV 上评估。

与 ``external_validate_stage.py``（sklearn Pipeline）分离，避免影响传统 ML 外部验证复现。
输出目录与文件结构与 ``external_validate_stage.py`` 一致（``external_validation_binary/`` 等）。

支持的 bundle 来源：MLP.py / DCNV23.py / FTTransformer3.py / TabTransformer3.py
（``_deep_tabular_pipeline.predict_bundle``）。

正类：``Disease=1``（UC）；bundle 输出为 P(UC) 一维概率，脚本内构造 ``prob_0`` / ``prob_1``。

启动示例（PowerShell）::

  python "F:\\KeTi\\Project\\Script\\external_validate_stage_dl.py" ^
    --model-run-dir "F:\\KeTi\\Project\\TCM\\output\\Stage2\\7_run_mlp_20260601_132351" ^
    --external-csv "F:\\KeTi\\Project\\TCM\\Data\\ATest-Stage2.csv"

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
from external_validate_stage import (  # noqa: E402
    build_pairwise_long_and_wide,
    build_type_montreal_slice_results,
    encode_labels,
    subgroup_block,
    _print_type_montreal_slice_table,
)

DL_MODEL_NAMES = frozenset({"MLP", "DCNV2", "FTTransformer", "TabTransformer"})


def _load_torch_bundle(model_path: Path) -> Dict[str, Any]:
    obj = joblib.load(model_path)
    if not isinstance(obj, dict):
        raise TypeError(
            f"{model_path.name} 不是深度学习 bundle（dict），而是 {type(obj).__name__}。"
            "请对 sklearn Pipeline 模型使用 external_validate_stage.py。"
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
    """Stage2 二分类默认 0=非UC, 1=UC；可选 class_mapping.json 与 ML 脚本一致。"""
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
    """返回 shape (n, 2)：列0=P(负类), 列1=P(UC)。"""
    raw = predict_bundle(bundle, X, batch_size=batch_size)
    p_pos = np.asarray(raw, dtype=float).reshape(-1)
    if len(p_pos) != len(X):
        raise ValueError(f"预测长度 {len(p_pos)} 与样本数 {len(X)} 不一致")
    p_pos = np.clip(p_pos, 0.0, 1.0)
    p_neg = 1.0 - p_pos
    return np.column_stack([p_neg, p_pos])


def main() -> None:
    p = argparse.ArgumentParser(
        description="Stage2 二分类外部验证（深度学习 bundle：MLP/DCNV2/FTTransformer/TabTransformer）"
    )
    p.add_argument(
        "--model-run-dir",
        type=str,
        required=True,
        help=r"阶段 B 产出目录（含 final_model.joblib + final_features_used.txt）",
    )
    p.add_argument("--external-csv", type=str, required=True, help="外部验证集 CSV")
    p.add_argument("--id-col", type=str, default="No")
    p.add_argument("--label-col", type=str, default="Disease")
    p.add_argument("--source-col", type=str, default="Source")
    p.add_argument("--agerange-col", type=str, default="Agerange")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="默认: model-run-dir/external_validation_binary/",
    )
    p.add_argument("--batch-size", type=int, default=None, help="默认读 stage_a_meta.json 或 32")
    p.add_argument("--type-col", type=str, default="Type")
    p.add_argument("--montreal-col", type=str, default="Montreal")
    p.add_argument(
        "--skip-type-montreal-slices",
        action="store_true",
        help="不计算 UC vs CD/IC/CRC 及 NA 池 vs E1/E2/E3 六组子集",
    )
    args = p.parse_args()

    run_dir = Path(args.model_run_dir).resolve()
    ext_path = Path(args.external_csv).resolve()
    out_dir = Path(args.output_dir).resolve() if args.output_dir else (run_dir / "external_validation_binary")
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
    need = {args.id_col, args.label_col, args.source_col, args.agerange_col}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"外部 CSV 缺少列: {sorted(miss)}")

    feat_miss = [c for c in final_feats if c not in df.columns]
    if feat_miss:
        raise ValueError(
            f"外部数据缺少模型所需特征列 ({len(feat_miss)} 个): {feat_miss[:20]}{'...' if len(feat_miss) > 20 else ''}"
        )

    y_raw = df[args.label_col].values
    y, bad = encode_labels(y_raw, class_to_idx)
    if any(v == -1 for v in y):
        bad_u = sorted(set(bad))
        raise ValueError(f"标签无法映射: {bad_u}；允许值为 {list(class_to_idx.keys())}")

    X = df[final_feats].copy().apply(pd.to_numeric, errors="coerce")
    y_prob_full = _predict_proba_binary(bundle, X, batch_size=batch_size)
    y_prob_pos = y_prob_full[:, 1]
    y_pred = (y_prob_pos >= args.threshold).astype(int)

    pair_name = f"{class_names[0]}_vs_{class_names[1]}"

    report: Dict[str, Any] = {
        "task": "binary",
        "model_type": "deep_learning_torch_bundle",
        "model_name": model_name or "unknown",
        "batch_size": batch_size,
        "model_run_dir": str(run_dir),
        "external_csv": str(ext_path),
        "id_col": args.id_col,
        "label_col": args.label_col,
        "source_col": args.source_col,
        "agerange_col": args.agerange_col,
        "threshold": args.threshold,
        "n_features": len(final_feats),
        "class_names": class_names,
        "positive_class_name": class_names[1],
        "overall": subgroup_block("全验证集", y, y_prob_pos, class_names, args.threshold),
        "by_source": {},
        "by_agerange": {},
    }

    for s in sorted(df[args.source_col].astype(str).unique()):
        m = df[args.source_col].astype(str) == s
        report["by_source"][s] = subgroup_block(f"Source={s}", y[m], y_prob_pos[m], class_names, args.threshold)

    for a in sorted(df[args.agerange_col].unique(), key=lambda x: (str(type(x)), str(x))):
        m = df[args.agerange_col] == a
        key = str(a)
        report["by_agerange"][key] = subgroup_block(f"Agerange={key}", y[m], y_prob_pos[m], class_names, args.threshold)

    strata_rows: List[Dict[str, Any]] = [
        {
            "stratum_id": "overall",
            "stratum_label": "全验证集",
            "n_samples": report["overall"]["n_samples"],
            "pairwise_binary": report["overall"].get("pairwise_binary", []),
        }
    ]
    for s in sorted(report["by_source"].keys()):
        b = report["by_source"][s]
        strata_rows.append(
            {
                "stratum_id": f"source__{s}",
                "stratum_label": f"Source={s}",
                "n_samples": b["n_samples"],
                "pairwise_binary": b.get("pairwise_binary", []),
            }
        )
    for key in sorted(report["by_agerange"].keys(), key=lambda x: (len(str(x)), str(x))):
        b = report["by_agerange"][key]
        strata_rows.append(
            {
                "stratum_id": f"agerange__{key}",
                "stratum_label": f"Agerange={key}",
                "n_samples": b["n_samples"],
                "pairwise_binary": b.get("pairwise_binary", []),
            }
        )

    long_df, wide_df = build_pairwise_long_and_wide(strata_rows, pair_name)
    long_path = out_dir / "external_pairwise_summary_long.csv"
    wide_path = out_dir / "external_pairwise_summary_wide.csv"
    long_df.to_csv(long_path, index=False, encoding="utf-8-sig")
    wide_df.to_csv(wide_path, index=False, encoding="utf-8-sig")
    strata_meta = long_df.groupby("stratum_id", as_index=False).agg(
        stratum_label=("stratum_label", "first"),
        n_samples_stratum=("n_samples", "first"),
    )
    strata_meta.to_csv(out_dir / "external_stratum_descriptor.csv", index=False, encoding="utf-8-sig")

    pred_df = pd.DataFrame(
        {
            args.id_col: df[args.id_col],
            args.source_col: df[args.source_col],
            args.agerange_col: df[args.agerange_col],
            "y_true": y,
            "y_true_label": [class_names[i] for i in y],
            "y_pred": y_pred,
            "y_pred_label": [class_names[i] for i in y_pred],
            f"prob_{class_names[0]}": y_prob_full[:, 0],
            f"prob_{class_names[1]}": y_prob_full[:, 1],
        }
    )
    pred_path = out_dir / "external_predictions.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    report_path = out_dir / "external_validation_report.json"
    report["pairwise_summary_note"] = (
        "深度学习二分类外部验证；bundle 经 predict_bundle 推理；"
        "正类列 prob_1 对应 Disease=1(UC)。输出结构与 external_validate_stage.py 对齐。"
    )

    if not args.skip_type_montreal_slices:
        tc, mc = args.type_col, args.montreal_col
        if tc not in df.columns or mc not in df.columns:
            print(f">>> 未运行 Type/Montreal 子集：CSV 缺少列 {tc!r} 或 {mc!r}")
        else:
            slices_dir = out_dir / "type_montreal_slices"
            slices_dir.mkdir(parents=True, exist_ok=True)
            slice_report, sum_df = build_type_montreal_slice_results(
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
                tc,
                mc,
                slices_dir,
            )
            report["type_montreal_slices"] = slice_report
            report["type_montreal_slices_meta"] = {
                "type_col": tc,
                "montreal_col": mc,
                "summary_csv": str((slices_dir / "type_montreal_slices_summary.csv").resolve()),
                "predictions_glob": str(slices_dir / "predictions_*.csv"),
            }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    def _print_block(title: str, b: Dict[str, Any]) -> None:
        print(f"\n--- {title} ---")
        print(f"    n={b['n_samples']}; 真实标签分布: {b.get('true_class_counts', {})}")
        if "note" in b:
            print(f"    说明: {b['note']}")
        if "metrics" in b:
            m = b["metrics"]
            print(
                f"    auc={m['auc']} auprc={m['auprc']} acc={m['acc']:.4f} "
                f"f1={m['f1']:.4f} precision={m['precision']:.4f} recall={m['recall']:.4f}"
            )

    print(">>> 深度学习二分类外部验证完成")
    print(f"    模型: {model_name or 'unknown'} | batch_size={batch_size}")
    print(f"    模型目录: {run_dir}")
    print(f"    外部数据: {ext_path} (N={len(df)})")
    print(f"    阈值: {args.threshold}; 正类(高概率列): {class_names[1]}")
    print(f"    报告: {report_path}")
    print(f"    逐例预测: {pred_path}")
    print(f"    指标长表: {long_path}")
    print(f"    指标宽表: {wide_path}")
    _print_block("全验证集", report["overall"])
    for k in sorted(report["by_source"].keys()):
        _print_block(f"Source={k}", report["by_source"][k])
    for k in sorted(report["by_agerange"].keys(), key=lambda x: (len(str(x)), str(x))):
        _print_block(f"Agerange={k}", report["by_agerange"][k])

    if "type_montreal_slices" in report:
        print("\n--- Type/Montreal 六组子集 ---")
        sd = out_dir / "type_montreal_slices"
        sum_csv = sd / "type_montreal_slices_summary.csv"
        if sum_csv.exists():
            _print_type_montreal_slice_table(pd.read_csv(sum_csv))
        print(f"    子集汇总: {sum_csv}")


if __name__ == "__main__":
    main()
