# -*- coding: utf-8 -*-
"""
Stage1 二分类外部验证：(IE+IBS) vs (UC+CD+IC+CRC)

数据列约定（ATest-Stage1.csv）：
  No, Source, Type, Agerange, Disease, <特征...>

训练目标 ``Disease``：0 = IE/FDIBS 组，1 = UC/CD/IC/CRC 组（与外部集 Type 分布一致）。

输出：
  - 全验证集 + Source 亚组 + Agerange 亚组
  - 各 Type 单病种子集（看该 Type 下模型是否判对训练侧分组）
  - 各 Type 与 IE+IBS 池（IE+FDIBS）或 炎症池（UC+CD+IC+CRC）的诱导二分类 AUC（样本够时）

启动示例::

  python external_validate_stage1.py ^
    --model-run-dir "F:\\KeTi\\Project\\outputs\\Stage1\\1_run_时间戳" ^
    --external-csv "F:\\KeTi\\Project\\Data\\ATest-Stage1.csv"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

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

# Type → 训练用 Disease（0=IE+IBS，1=UC+CD+IC+CRC）
TYPE_IE_IBS: Set[str] = {"IE", "FDIBS"}
TYPE_INFLAM: Set[str] = {"UC", "CD", "IC", "CRC"}
TYPE_ALL_ORDERED: List[str] = ["UC", "CD", "IC", "IE", "CRC", "FDIBS"]


def _pipeline_classes(pipe: Any) -> np.ndarray:
    if hasattr(pipe, "classes_"):
        return np.asarray(pipe.classes_)
    last = pipe.steps[-1][1]
    if not hasattr(last, "classes_"):
        raise AttributeError("无法从 Pipeline 获取 classes_")
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


def _type_series(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].astype(str).str.strip().str.upper()


def build_feature_matrix(df: pd.DataFrame, final_feats: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """按训练特征顺序取列；外部 CSV 缺少的列（如 Unnamed: 40）补 NaN，供 Pipeline 内 imputer 填补。"""
    missing = [c for c in final_feats if c not in df.columns]
    X = df.reindex(columns=final_feats)
    X = X.apply(pd.to_numeric, errors="coerce")
    return X, missing


def expected_disease_from_type(t: str) -> int:
    t = t.upper()
    if t in TYPE_IE_IBS:
        return 0
    if t in TYPE_INFLAM:
        return 1
    raise ValueError(f"未知 Type: {t}")


def subgroup_block(
    name: str,
    y: np.ndarray,
    y_prob_pos: np.ndarray,
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
        "score_column": f"prob_{class_names[1]}",
        "score_meaning": f"P({class_names[1]}) = P(UC+CD+IC+CRC 组)",
    }
    if n == 0:
        block["note"] = "无样本"
        block["pairwise_binary"] = []
        return block
    if len(uniq) < 2:
        block["note"] = "仅含单一真实类别；AUC/AUPRC 不可用"
    y_pred = (y_prob_pos >= threshold).astype(int)
    block["metrics"] = {
        "auc": float(roc_auc_score(y, y_prob_pos)) if len(uniq) >= 2 else float("nan"),
        "auprc": float(average_precision_score(y, y_prob_pos)) if len(uniq) >= 2 else float("nan"),
        "acc": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
    }
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    block["confusion_matrix"] = {
        "matrix": confusion_matrix(y, y_pred, labels=[0, 1]).tolist(),
        "rows": [f"true_{c}" for c in class_names],
        "cols": [f"pred_{c}" for c in class_names],
    }
    block["pairwise_binary"] = [
        {
            "pair": pair_name,
            "auc": block["metrics"]["auc"],
            "auprc": block["metrics"]["auprc"],
            "acc": block["metrics"]["acc"],
            "f1": block["metrics"]["f1"],
            "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan"),
            "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan"),
            "precision": block["metrics"]["precision"],
            "recall": block["metrics"]["recall"],
        }
    ]
    return block


def _summary_row(sid: str, blk: Dict[str, Any], extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "stratum_id": sid,
        "n_samples": blk.get("n_samples", 0),
        "note": blk.get("note", ""),
    }
    if extra:
        row.update(extra)
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


def build_pairwise_tables(
    strata_rows: List[Dict[str, Any]],
    pair_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_cols = ["n_samples", "auc", "auprc", "acc", "f1", "sensitivity", "specificity", "precision", "recall"]
    long_records: List[Dict[str, Any]] = []
    for row in strata_rows:
        sid, slab, n = row["stratum_id"], row["stratum_label"], int(row["n_samples"])
        pws = row.get("pairwise_binary") or []
        rec = pws[0] if pws else None
        base = {"stratum_id": sid, "stratum_label": slab, "n_samples": n, "pair": pair_name}
        if rec is None:
            long_records.append({**base, **{k: np.nan for k in metric_cols}, "pair_computable": False})
        else:
            long_records.append(
                {
                    **base,
                    **{k: rec.get(k, np.nan) for k in metric_cols},
                    "pair_computable": bool(pd.notna(rec.get("auc"))),
                }
            )
    long_df = pd.DataFrame(long_records)
    wrow: Dict[str, Any] = {"pair": pair_name}
    for sid in [r["stratum_id"] for r in strata_rows]:
        sub = long_df[long_df["stratum_id"] == sid]
        if len(sub) == 0:
            for mc in metric_cols:
                wrow[f"{sid}__{mc}"] = np.nan
            wrow[f"{sid}__pair_computable"] = False
        else:
            r = sub.iloc[0]
            for mc in metric_cols:
                wrow[f"{sid}__{mc}"] = r[mc]
            wrow[f"{sid}__pair_computable"] = bool(r["pair_computable"])
    return long_df, pd.DataFrame([wrow])


def _save_pred_slice(
    sub_df: pd.DataFrame,
    y_sub: np.ndarray,
    p_sub: np.ndarray,
    prob_full: np.ndarray,
    class_names: List[str],
    threshold: float,
    id_col: str,
    type_col: str,
    source_col: str,
    agerange_col: str,
    label_col: str,
    path: Path,
) -> None:
    pd.DataFrame(
        {
            id_col: sub_df[id_col],
            type_col: sub_df[type_col],
            source_col: sub_df[source_col],
            agerange_col: sub_df[agerange_col],
            label_col: sub_df[label_col],
            "y_true": y_sub,
            "y_pred": (p_sub >= threshold).astype(int),
            f"prob_{class_names[0]}": prob_full[:, 0],
            f"prob_{class_names[1]}": prob_full[:, 1],
        }
    ).to_csv(path, index=False, encoding="utf-8-sig")


def build_type_slices(
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
    slices_dir: Path,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    slices_dir.mkdir(parents=True, exist_ok=True)
    typ = _type_series(df, type_col)
    unknown = sorted(set(typ.unique()) - TYPE_IE_IBS - TYPE_INFLAM)
    if unknown:
        print(f"    [警告] 未映射的 Type 值（将跳过专属单病种行）: {unknown}")
    report: Dict[str, Any] = {}
    summary_rows: List[Dict[str, Any]] = []
    per_type_rows: List[Dict[str, Any]] = []

    ie_ibs_mask = typ.isin(list(TYPE_IE_IBS)).values
    inflam_mask = typ.isin(list(TYPE_INFLAM)).values

    # 1) 各 Type 单病种：金标准 = Disease（该 Type 内应全为 0 或 1）
    for t in TYPE_ALL_ORDERED:
        mask = (typ == t).values
        if not mask.any():
            continue
        sid = f"type_only_{t.lower()}"
        y_sub = y_full[mask]
        p_sub = y_prob_pos[mask]
        exp = expected_disease_from_type(t)
        n_mis = int(np.sum(y_sub != exp))
        blk = subgroup_block(f"仅 Type={t}", y_sub, p_sub, class_names, threshold)
        arm = "IE_IBS" if exp == 0 else "INFLAM_UC_CD_IC_CRC"
        report[sid] = {
            "description": f"仅 {t} 病例；期望 Disease={exp}（{arm}）",
            "type": t,
            "expected_disease": exp,
            "n_type_disease_mismatch": n_mis,
            **{k: v for k, v in blk.items() if k != "name"},
        }
        _save_pred_slice(
            df.loc[mask],
            y_sub,
            p_sub,
            y_prob_full[mask],
            class_names,
            threshold,
            id_col,
            type_col,
            source_col,
            agerange_col,
            label_col,
            slices_dir / f"predictions_{sid}.csv",
        )
        summary_rows.append(
            _summary_row(
                sid,
                blk,
                extra={"type": t, "expected_disease": exp, "n_type_disease_mismatch": n_mis},
            )
        )
        m = blk.get("metrics", {})
        per_type_rows.append(
            {
                "type": t,
                "n": int(mask.sum()),
                "expected_arm": arm,
                "expected_disease": exp,
                "n_disease_mismatch": n_mis,
                "mean_prob_inflam_arm": float(np.mean(p_sub)),
                "acc": m.get("acc", np.nan),
                "note": blk.get("note", ""),
            }
        )

    # 2) 各炎症 Type vs IE+IBS 池（可算 AUC）
    for t in sorted(TYPE_INFLAM):
        sid = f"{t.lower()}_vs_ie_ibs_pool"
        mask = ((typ == t) | typ.isin(list(TYPE_IE_IBS))).values
        y_sub = y_full[mask]
        p_sub = y_prob_pos[mask]
        blk = subgroup_block(f"{t} vs IE+FDIBS", y_sub, p_sub, class_names, threshold)
        report[sid] = {
            "description": f"Type={t} vs Type∈{{IE,FDIBS}}，金标准={label_col}",
            **{k: v for k, v in blk.items() if k != "name"},
        }
        summary_rows.append(_summary_row(sid, blk, extra={"comparison": sid}))
        _save_pred_slice(
            df.loc[mask],
            y_sub,
            p_sub,
            y_prob_full[mask],
            class_names,
            threshold,
            id_col,
            type_col,
            source_col,
            agerange_col,
            label_col,
            slices_dir / f"predictions_{sid}.csv",
        )

    # 3) IE、FDIBS 分别 vs 炎症池
    for t in sorted(TYPE_IE_IBS):
        sid = f"{t.lower()}_vs_inflam_pool"
        mask = ((typ == t) | inflam_mask).values
        y_sub = y_full[mask]
        p_sub = y_prob_pos[mask]
        blk = subgroup_block(f"{t} vs UC+CD+IC+CRC", y_sub, p_sub, class_names, threshold)
        report[sid] = {
            "description": f"Type={t} vs Type∈{{UC,CD,IC,CRC}}",
            **{k: v for k, v in blk.items() if k != "name"},
        }
        summary_rows.append(_summary_row(sid, blk, extra={"comparison": sid}))
        _save_pred_slice(
            df.loc[mask],
            y_sub,
            p_sub,
            y_prob_full[mask],
            class_names,
            threshold,
            id_col,
            type_col,
            source_col,
            agerange_col,
            label_col,
            slices_dir / f"predictions_{sid}.csv",
        )

    # 4) 全 IE+IBS 池 vs 全炎症池
    sid = "ie_ibs_pool_vs_inflam_pool"
    mask = ie_ibs_mask | inflam_mask
    blk = subgroup_block("IE+FDIBS vs UC+CD+IC+CRC", y_full[mask], y_prob_pos[mask], class_names, threshold)
    report[sid] = {"description": "两训练臂在外部集上的总体对照", **{k: v for k, v in blk.items() if k != "name"}}
    summary_rows.append(_summary_row(sid, blk))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(slices_dir / "type_slices_summary.csv", index=False, encoding="utf-8-sig")
    per_type_df = pd.DataFrame(per_type_rows)
    per_type_df.to_csv(slices_dir / "type_per_disease_recognition.csv", index=False, encoding="utf-8-sig")
    return report, summary_df, per_type_df


def main() -> None:
    p = argparse.ArgumentParser(description="Stage1 外部验证：全量 + Source/Agerange + Type 分层")
    p.add_argument("--model-run-dir", type=str, required=True)
    p.add_argument("--external-csv", type=str, required=True)
    p.add_argument("--id-col", type=str, default="No")
    p.add_argument("--source-col", type=str, default="Source")
    p.add_argument("--type-col", type=str, default="Type")
    p.add_argument("--agerange-col", type=str, default="Agerange")
    p.add_argument("--label-col", type=str, default="Disease")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--skip-type-slices", action="store_true")
    args = p.parse_args()

    run_dir = Path(args.model_run_dir).resolve()
    ext_path = Path(args.external_csv).resolve()
    out_dir = Path(args.output_dir).resolve() if args.output_dir else (run_dir / "external_validation_stage1")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / "final_model.joblib"
    feats_path = run_dir / "final_features_used.txt"
    if not model_path.exists() or not feats_path.exists():
        raise FileNotFoundError(f"缺少: {model_path} 或 {feats_path}")

    pipe = joblib.load(model_path)
    cls_arr = _pipeline_classes(pipe)
    if len(cls_arr) != 2:
        raise ValueError(f"需要二分类模型，classes_={cls_arr}")
    class_names = [str(c) for c in cls_arr]
    class_to_idx = {str(c): i for i, c in enumerate(cls_arr)}

    class_path = run_dir / "class_mapping.json"
    if class_path.exists():
        jti = {str(k): int(v) for k, v in json.loads(class_path.read_text(encoding="utf-8"))["class_to_idx"].items()}
        for i, lab in enumerate(class_names):
            if jti.get(lab) != i:
                raise ValueError("class_mapping.json 与模型 classes_ 列顺序不一致")
        class_to_idx = jti

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
    y_prob_full = pipe.predict_proba(X)
    y_prob_pos = y_prob_full[:, 1]
    y_pred = (y_prob_pos >= args.threshold).astype(int)
    pair_name = f"{class_names[0]}_vs_{class_names[1]}"

    report: Dict[str, Any] = {
        "task": "stage1_binary",
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
        report["by_agerange"][str(a)] = subgroup_block(f"Agerange={a}", y[m], y_prob_pos[m], class_names, args.threshold)

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
        type_report, _, per_type_df = build_type_slices(
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

    with open(out_dir / "external_validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(">>> Stage1 外部验证完成")
    print(f"    输出目录: {out_dir}")
    m = report["overall"]["metrics"]
    print(f"    全验证集 n={report['overall']['n_samples']} auc={m['auc']:.4f} acc={m['acc']:.4f}")
    if not args.skip_type_slices:
        print(f"    Type 分层: {out_dir / 'type_slices'}")
        print(f"    各病种识别简表: {out_dir / 'type_slices' / 'type_per_disease_recognition.csv'}")


if __name__ == "__main__":
    main()
