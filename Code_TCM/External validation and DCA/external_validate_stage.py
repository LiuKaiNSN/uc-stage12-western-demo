# -*- coding: utf-8 -*-
"""
二分类（Stage2）外部验证：加载阶段 B 的 ``final_model.joblib`` + ``final_features_used.txt``，
在外部 CSV 上评估；支持可选 ``class_mapping.json``（与三分类格式相同，仅 2 个类别），
若缺失则从 ``Pipeline`` 最终分类器的 ``classes_`` 推断。

输出结构与 ``external_validate_stage3.py`` 对齐思路：全验证集 + Source + Agerange 五分层、
``external_validation_report.json``、逐例预测、以及 **单列「伪 pairwise」** 汇总表
（行名 ``{负类}_vs_{正类}``，列与三分类 pairwise 表一致：auc/acc/f1/sensitivity/specificity），
便于与 Stage3 汇总流程在 Excel 中对照。

**Type / Montreal 扩展子集**（当 CSV 含 ``Type``、``Montreal`` 且未加 ``--skip-type-montreal-slices`` 时）：

1. ``uc_vs_cd`` / ``uc_vs_ic`` / ``uc_vs_crc``：仅保留 UC 与对应病种行，金标准为 ``Disease``（与训练一致）。
2. ``na_pool_vs_uc_e1`` / ``..._e2`` / ``..._e3``：阴性为 CD+IC+CRC（Montreal 多为 NA），阳性为 UC 且 Montreal 为 1/2/3（对应 E1/E2/E3），
   仍用同一模型的 **P(UC)** 做判别（探索性子群）。

正类概率：``predict_proba`` 的第二列（对应 ``classes_[1]``，与 RF.py 中 ``[:, 1]`` 一致）。
默认阈值 0.5，可用 ``--threshold`` 修改。

启动示例（PowerShell）::

  python "F:\\KeTi\\Project\\Script\\external_validate_stage.py" ^
    --model-run-dir "F:\\KeTi\\Project\\outputs\\Stage2\\6_run_时间戳" ^
    --external-csv "F:\\KeTi\\Project\\Data\\ATest-Stage2.csv"

请在训练该模型时所用的同一 Python 环境中运行，以免 joblib 反序列化失败。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


def _pipeline_classes(pipe: Any) -> np.ndarray:
    if hasattr(pipe, "classes_"):
        return np.asarray(pipe.classes_)
    last = pipe.steps[-1][1]
    if not hasattr(last, "classes_"):
        raise AttributeError("无法从 Pipeline 获取 classes_，请检查模型是否为分类器")
    return np.asarray(last.classes_)


def encode_labels(y_raw: np.ndarray, class_to_idx: Dict[str, int]) -> Tuple[np.ndarray, List[str]]:
    ys: List[int] = []
    bad: List[str] = []
    for v in y_raw:
        k = str(v).strip()
        if k not in class_to_idx:
            bad.append(k)
            ys.append(-1)
        else:
            ys.append(class_to_idx[k])
    return np.array(ys, dtype=int), bad


def binary_metrics_at_threshold(
    y_true: np.ndarray,
    y_prob_positive: np.ndarray,
    threshold: float,
    pair_name: str,
) -> Dict[str, Any]:
    """与三分类 pairwise 表字段对齐：auc 与阈值无关；acc/f1/sens/spec 在阈值下。"""
    y_true = np.asarray(y_true, dtype=int).ravel()
    p = np.asarray(y_prob_positive, dtype=float).ravel()
    row: Dict[str, Any] = {"pair": pair_name}
    if len(y_true) == 0:
        row.update(
            {
                "auc": float("nan"),
                "acc": float("nan"),
                "f1": float("nan"),
                "sensitivity": float("nan"),
                "specificity": float("nan"),
                "auprc": float("nan"),
            }
        )
        return row
    uniq = np.unique(y_true)
    if len(uniq) < 2:
        row["auc"] = float("nan")
        row["auprc"] = float("nan")
    else:
        row["auc"] = float(roc_auc_score(y_true, p))
        row["auprc"] = float(average_precision_score(y_true, p))
    y_pred = (p >= threshold).astype(int)
    row["acc"] = float(accuracy_score(y_true, y_pred))
    row["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    row["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    row["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    row["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    row["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    return row


def cm_to_dict_binary(cm: np.ndarray, class_names: List[str]) -> Dict[str, Any]:
    return {
        "matrix": cm.tolist(),
        "rows": [f"true_{c}" for c in class_names],
        "cols": [f"pred_{c}" for c in class_names],
    }


def subgroup_block(
    name: str,
    y: np.ndarray,
    y_prob_positive: np.ndarray,
    class_names: List[str],
    threshold: float,
) -> Dict[str, Any]:
    n = int(len(y))
    uniq, cnt = np.unique(y, return_counts=True)
    dist = {class_names[int(u)]: int(c) for u, c in zip(uniq, cnt)}
    pair_name = f"{class_names[0]}_vs_{class_names[1]}"
    block: Dict[str, Any] = {
        "name": name,
        "n_samples": n,
        "true_class_counts": dist,
        "classes_present": [class_names[int(u)] for u in uniq],
        "threshold": threshold,
        "positive_class": class_names[1],
    }
    if n == 0:
        block["note"] = "无样本"
        block["pairwise_binary"] = []
        return block
    if len(uniq) < 2:
        block["note"] = "仅含单一真实类别；AUC/AUPRC 不可用或参考意义有限"
    y_pred = (y_prob_positive >= threshold).astype(int)
    block["metrics"] = {
        "auc": float(roc_auc_score(y, y_prob_positive)) if len(uniq) >= 2 else float("nan"),
        "auprc": float(average_precision_score(y, y_prob_positive)) if len(uniq) >= 2 else float("nan"),
        "acc": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
    }
    block["confusion_matrix"] = cm_to_dict_binary(
        confusion_matrix(y, y_pred, labels=[0, 1]),
        class_names,
    )
    block["pairwise_binary"] = [binary_metrics_at_threshold(y, y_prob_positive, threshold, pair_name)]
    return block


def build_pairwise_long_and_wide(
    strata_rows: List[Dict[str, Any]],
    pair_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_cols = ["n_samples", "auc", "auprc", "acc", "f1", "sensitivity", "specificity", "precision", "recall"]
    long_records: List[Dict[str, Any]] = []
    for row in strata_rows:
        sid = row["stratum_id"]
        slab = row["stratum_label"]
        n = int(row["n_samples"])
        pws = row.get("pairwise_binary") or []
        rec = pws[0] if pws else None
        if rec is None:
            long_records.append(
                {
                    "stratum_id": sid,
                    "stratum_label": slab,
                    "n_samples": n,
                    "pair": pair_name,
                    "auc": np.nan,
                    "auprc": np.nan,
                    "acc": np.nan,
                    "f1": np.nan,
                    "sensitivity": np.nan,
                    "specificity": np.nan,
                    "precision": np.nan,
                    "recall": np.nan,
                    "pair_computable": False,
                }
            )
        else:
            long_records.append(
                {
                    "stratum_id": sid,
                    "stratum_label": slab,
                    "n_samples": n,
                    "pair": pair_name,
                    "auc": rec.get("auc", np.nan),
                    "auprc": rec.get("auprc", np.nan),
                    "acc": rec.get("acc", np.nan),
                    "f1": rec.get("f1", np.nan),
                    "sensitivity": rec.get("sensitivity", np.nan),
                    "specificity": rec.get("specificity", np.nan),
                    "precision": rec.get("precision", np.nan),
                    "recall": rec.get("recall", np.nan),
                }
            )
    long_df = pd.DataFrame(long_records)
    long_df["pair_computable"] = long_df["auc"].notna() & long_df["n_samples"].gt(0)

    wide_rows: List[Dict[str, Any]] = []
    wrow: Dict[str, Any] = {"pair": pair_name}
    stratum_ids = [r["stratum_id"] for r in strata_rows]
    for sid in stratum_ids:
        sub = long_df[long_df["stratum_id"] == sid]
        if len(sub) == 0:
            for mc in metric_cols:
                wrow[f"{sid}__{mc}"] = np.nan
            wrow[f"{sid}__pair_computable"] = False
            continue
        r = sub.iloc[0]
        for mc in metric_cols:
            wrow[f"{sid}__{mc}"] = r[mc]
        wrow[f"{sid}__pair_computable"] = bool(r["pair_computable"])
    wide_rows.append(wrow)
    wide_df = pd.DataFrame(wide_rows)
    return long_df, wide_df


def _type_series(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].astype(str).str.strip().str.upper()


def _montreal_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def _warn_disease_type_mismatch(df: pd.DataFrame, typ: pd.Series, label_col: str, tag: str) -> None:
    """Disease 与 Type 不一致时仅告警，不中断。"""
    try:
        d = pd.to_numeric(df[label_col], errors="coerce").fillna(-1).astype(int).values
    except Exception:
        return
    t = np.asarray(typ.values)
    bad_uc = int(np.sum((t == "UC") & (d != 1)))
    bad_non = int(np.sum(np.isin(t, np.array(["CD", "IC", "CRC"])) & (d != 0)))
    if bad_uc or bad_non:
        print(
            f"    [警告][{tag}] Disease 与 Type 不一致: UC行但Disease!=1 有 {bad_uc} 条; "
            f"CD/IC/CRC行但Disease!=0 有 {bad_non} 条（仍按 Disease 作金标准）。"
        )


def _summary_row(sid: str, blk: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"slice_id": sid, "n_samples": blk.get("n_samples", 0), "note": blk.get("note", "")}
    dist = blk.get("true_class_counts") or {}
    row["true_class_counts_json"] = json.dumps(dist, ensure_ascii=False)
    if "metrics" in blk:
        row.update(blk["metrics"])
    else:
        for k in ("auc", "auprc", "acc", "precision", "recall", "f1"):
            row[k] = np.nan
    if blk.get("pairwise_binary"):
        pr = blk["pairwise_binary"][0]
        row["sensitivity"] = pr.get("sensitivity", np.nan)
        row["specificity"] = pr.get("specificity", np.nan)
    else:
        row["sensitivity"] = np.nan
        row["specificity"] = np.nan
    return row


def build_type_montreal_slice_results(
    df: pd.DataFrame,
    y_full: np.ndarray,
    y_prob_pos: np.ndarray,
    y_prob_full: np.ndarray,
    class_names: List[str],
    threshold: float,
    id_col: str,
    source_col: str,
    agerange_col: str,
    label_col: str,
    type_col: str,
    montreal_col: str,
    slices_dir: Path,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    slices_dir.mkdir(parents=True, exist_ok=True)
    typ = _type_series(df, type_col)
    mont = _montreal_numeric(df, montreal_col)

    report_slices: Dict[str, Any] = {}
    summary_rows: List[Dict[str, Any]] = []

    # --- UC vs CD / IC / CRC：金标准 = Disease（与训练一致）---
    for comp in ("CD", "IC", "CRC"):
        sid = f"uc_vs_{comp.lower()}"
        mask = typ.isin(["UC", comp]).values
        _warn_disease_type_mismatch(df.loc[mask], typ.loc[mask], label_col, sid)
        y_sub = y_full[mask]
        p_sub = y_prob_pos[mask]
        name = f"子集[{sid}]: Type∈{{UC,{comp}}}, 金标准={label_col}"
        blk = subgroup_block(name, y_sub, p_sub, class_names, threshold)
        report_slices[sid] = {
            "description": f"UC 与 {comp} 对照，使用全表 {label_col} 作为二分类金标准",
            **{k: v for k, v in blk.items() if k != "name"},
            "slice_name": name,
        }
        sub_df = df.loc[mask].copy()
        pred_slice = pd.DataFrame(
            {
                id_col: sub_df[id_col],
                type_col: sub_df[type_col],
                montreal_col: sub_df[montreal_col],
                source_col: sub_df[source_col],
                agerange_col: sub_df[agerange_col],
                "y_true": y_sub,
                "y_true_label": [class_names[i] for i in y_sub],
                "y_pred": (p_sub >= threshold).astype(int),
                "y_pred_label": [class_names[i] for i in (p_sub >= threshold).astype(int)],
                f"prob_{class_names[0]}": y_prob_full[mask, 0],
                f"prob_{class_names[1]}": y_prob_full[mask, 1],
            }
        )
        pred_slice.to_csv(slices_dir / f"predictions_{sid}.csv", index=False, encoding="utf-8-sig")
        summary_rows.append(_summary_row(sid, blk))

    # --- (CD+IC+CRC) vs UC 且 Montreal=k：金标准 阳性=该 UC 亚组 ---
    for k, e_label in [(1, "E1"), (2, "E2"), (3, "E3")]:
        sid = f"na_pool_vs_uc_{e_label.lower()}"
        m_uc = (typ == "UC") & (mont == k)
        m_neg = typ.isin(["CD", "IC", "CRC"])
        mask = (m_uc | m_neg).values
        idx = np.where(mask)[0]
        y_custom = np.zeros(len(idx), dtype=int)
        for ii, i in enumerate(idx):
            if typ.iloc[i] == "UC" and pd.notna(mont.iloc[i]) and float(mont.iloc[i]) == float(k):
                y_custom[ii] = 1
            else:
                y_custom[ii] = 0
        p_sub = y_prob_pos[mask]
        name = f"子集[{sid}]: (CD+IC+CRC) vs UC-{e_label}(Montreal={k})"
        blk = subgroup_block(name, y_custom, p_sub, class_names, threshold)
        report_slices[sid] = {
            "description": f"阴性=Type∈{{CD,IC,CRC}}；阳性=UC 且 {montreal_col}={k}（{e_label}）；得分仍为 P({class_names[1]})",
            **{k2: v for k2, v in blk.items() if k2 != "name"},
            "slice_name": name,
        }
        sub_df = df.loc[mask].copy()
        pred_slice = pd.DataFrame(
            {
                id_col: sub_df[id_col],
                type_col: sub_df[type_col],
                montreal_col: sub_df[montreal_col],
                source_col: sub_df[source_col],
                agerange_col: sub_df[agerange_col],
                "y_true_slice": y_custom,
                "y_true_slice_label": np.where(y_custom == 1, f"UC_{e_label}", "NonUC_CD_IC_CRC"),
                "y_pred": (p_sub >= threshold).astype(int),
                "y_pred_label": [class_names[i] for i in (p_sub >= threshold).astype(int)],
                f"prob_{class_names[0]}": y_prob_full[mask, 0],
                f"prob_{class_names[1]}": y_prob_full[mask, 1],
            }
        )
        pred_slice.to_csv(slices_dir / f"predictions_{sid}.csv", index=False, encoding="utf-8-sig")
        summary_rows.append(_summary_row(sid, blk))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(slices_dir / "type_montreal_slices_summary.csv", index=False, encoding="utf-8-sig")
    return report_slices, summary_df


def _print_type_montreal_slice_table(summary_df: pd.DataFrame) -> None:
    for _, r in summary_df.iterrows():
        auc_v = r.get("auc", np.nan)
        f1_v = r.get("f1", np.nan)
        print(
            f"    [{r['slice_id']}] n={int(r['n_samples'])} "
            f"auc={auc_v} auprc={r.get('auprc', np.nan)} f1={f1_v} "
            f"sens={r.get('sensitivity', np.nan)} spec={r.get('specificity', np.nan)}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Stage2 二分类外部验证（全量 + Source + Agerange）")
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
    p.add_argument("--threshold", type=float, default=0.5, help="二分类决策阈值（与 RF.py evaluate_binary 默认一致）")
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="默认: model-run-dir/external_validation_binary/",
    )
    p.add_argument("--type-col", type=str, default="Type", help="疾病类型列：UC/CD/IC/CRC")
    p.add_argument("--montreal-col", type=str, default="Montreal", help="UC 的 Montreal/E1–E3 列（1/2/3），非 UC 多为 NA")
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

    pipe = joblib.load(model_path)
    cls_arr = _pipeline_classes(pipe)
    if len(cls_arr) != 2:
        raise ValueError(f"本脚本用于二分类，当前模型 classes_ 长度={len(cls_arr)}: {cls_arr}")

    if class_path.exists():
        class_map = json.loads(class_path.read_text(encoding="utf-8"))
        class_names = [str(c) for c in cls_arr]
        jti = {str(k): int(v) for k, v in class_map["class_to_idx"].items()}
        for i, lab in enumerate(class_names):
            if jti.get(lab) != i:
                raise ValueError(
                    f"class_mapping.json 与模型 classes_ 不一致："
                    f"要求 jti['{lab}']=={i}（与 predict_proba 列顺序一致），当前为 {jti.get(lab)}。"
                    f"可删除该文件以自动按模型 classes_ 解析标签。"
                )
        class_to_idx = dict(jti)
    else:
        class_names = [str(c) for c in cls_arr]
        class_to_idx = {str(c): i for i, c in enumerate(cls_arr)}

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
    y_prob_full = pipe.predict_proba(X)
    if y_prob_full.shape[1] != 2:
        raise ValueError(f"predict_proba 列数异常: {y_prob_full.shape}")
    y_prob_pos = y_prob_full[:, 1]
    y_pred = (y_prob_pos >= args.threshold).astype(int)

    pair_name = f"{class_names[0]}_vs_{class_names[1]}"

    report: Dict[str, Any] = {
        "task": "binary",
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
        "二分类仅一行 pair（负类_vs_正类）；sensitivity/specificity/acc/f1 在 --threshold 下计算；"
        "auc/auprc 与阈值无关。宽表便于与 Stage3 的 external_pairwise_summary_wide 对照列结构。"
    )

    if not args.skip_type_montreal_slices:
        tc, mc = args.type_col, args.montreal_col
        if tc not in df.columns or mc not in df.columns:
            print(f">>> 未运行 Type/Montreal 子集：CSV 缺少列 {tc!r} 或 {mc!r}（可加 --skip-type-montreal-slices 静默）")
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

    print(">>> 二分类外部验证完成")
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
        print(f"    子集逐例预测: {sd / 'predictions_*.csv'}")


if __name__ == "__main__":
    main()
