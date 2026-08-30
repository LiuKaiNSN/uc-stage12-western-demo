# -*- coding: utf-8 -*-
"""Export TCM-integrated Stage1/Stage2 main tables (overall + Source + Agerange)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

from _figure_common import load_external_predictions, load_external_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TCM Stage12 Table export (Stage1 + Stage2, with subgroups)")
    p.add_argument(
        "--s1-run-dir",
        type=str,
        default=r"F:\KeTi\Project\TCM\output\Stage1\6_run_20260708_052605",
    )
    p.add_argument(
        "--s2-run-dir",
        type=str,
        default=r"F:\KeTi\Project\TCM\output\Stage2\6_run_20260707_111716",
    )
    p.add_argument(
        "--workflow-json",
        type=str,
        default=r"F:\KeTi\Project\TCM\output\workflow_batch\tcm\06_6_best_cross\workflow_stage12.json",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=r"F:\KeTi\Project\Figure\Stage12_TCM\data",
    )
    p.add_argument("--bootstrap-iter", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Point estimates only (no bootstrap CI; faster, fully deterministic from data)",
    )
    return p.parse_args()


def _fmt_ci(pt: float, lo: float, hi: float, digits: int = 3) -> str:
    if not np.isfinite(pt):
        return ""
    if not np.isfinite(lo) or not np.isfinite(hi):
        return f"{pt:.{digits}f}"
    return f"{pt:.{digits}f} ({lo:.{digits}f}–{hi:.{digits}f})"


def _binary_point_metrics(
    y: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    uniq = np.unique(y)
    pred = (scores >= threshold).astype(int)
    out: Dict[str, float] = {
        "auc": float(roc_auc_score(y, scores)) if len(uniq) >= 2 else float("nan"),
        "auprc": float(average_precision_score(y, scores)) if len(uniq) >= 2 else float("nan"),
        "acc": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    out["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    out["ppv"] = out["precision"]
    out["npv"] = float(tn / (tn + fn)) if (tn + fn) > 0 else float("nan")
    out["tp"] = float(tp)
    out["tn"] = float(tn)
    out["fp"] = float(fp)
    out["fn"] = float(fn)
    return out


def _bootstrap_binary_ci(
    y: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    metrics: Tuple[str, ...],
    n_boot: int,
    seed: int,
) -> Dict[str, Tuple[float, float, float]]:
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    point = _binary_point_metrics(y, scores, threshold)
    if len(np.unique(y)) < 2 or n_boot <= 0:
        return {k: (point.get(k, float("nan")), point.get(k, float("nan")), point.get(k, float("nan"))) for k in metrics}

    rng = np.random.default_rng(seed)
    n = len(y)
    boots: Dict[str, List[float]] = {k: [] for k in metrics}

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y[idx]
        sc = scores[idx]
        if len(np.unique(yt)) < 2:
            continue
        pred = (sc >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
        vals = {
            "auc": float(roc_auc_score(yt, sc)),
            "auprc": float(average_precision_score(yt, sc)),
            "acc": float(accuracy_score(yt, pred)),
            "f1": float(f1_score(yt, pred, zero_division=0)),
            "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan"),
            "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan"),
            "ppv": float(precision_score(yt, pred, zero_division=0)),
            "npv": float(tn / (tn + fn)) if (tn + fn) > 0 else float("nan"),
        }
        for k in metrics:
            v = vals.get(k, float("nan"))
            if np.isfinite(v):
                boots[k].append(v)

    out: Dict[str, Tuple[float, float, float]] = {}
    for k in metrics:
        pt = point.get(k, float("nan"))
        arr = boots[k]
        if not arr:
            out[k] = (pt, pt, pt)
        else:
            lo, hi = np.percentile(arr, [2.5, 97.5])
            out[k] = (pt, float(lo), float(hi))
    return out


def _iter_strata(df: pd.DataFrame) -> List[Tuple[str, str, pd.DataFrame]]:
    strata: List[Tuple[str, str, pd.DataFrame]] = [("overall", "全验证集", df)]
    for src, sub in sorted(df.groupby("Source", dropna=False), key=lambda x: str(x[0])):
        strata.append((f"source_{src}", f"Source={src}", sub))
    for ag, sub in sorted(df.groupby("Agerange", dropna=False), key=lambda x: str(x[0])):
        strata.append((f"agerange_{ag}", f"Agerange={ag}", sub))
    return strata


def _export_stage(
    stage_key: str,
    run_dir: Path,
    workflow_threshold: float,
    out_dir: Path,
    n_boot: int,
    seed: int,
    use_bootstrap: bool,
) -> None:
    report = load_external_report(run_dir)
    df = load_external_predictions(run_dir)
    class_names = report.get("class_names", ["0", "1"])

    long_rows: List[Dict[str, Any]] = []
    cm_overall: pd.DataFrame | None = None
    seed_offset = 0 if stage_key == "stage1" else 1000

    ci_metrics_default = ("auc", "acc", "f1")
    ci_metrics_workflow = ("acc", "sensitivity", "specificity", "ppv", "npv")

    strata = _iter_strata(df)
    for stratum_idx, (stratum_id, stratum_label, sub) in enumerate(strata):
        stratum_seed = seed + seed_offset + stratum_idx * 17
        y = sub["y_true"].astype(int).to_numpy()
        scores = sub["prob_1"].astype(float).to_numpy()
        n = int(len(sub))
        section = "overall" if stratum_id == "overall" else (
            "source" if stratum_id.startswith("source_") else "agerange"
        )

        for threshold, thresh_section in ((0.5, "threshold_0.5"), (workflow_threshold, "workflow_threshold")):
            point = _binary_point_metrics(y, scores, threshold)
            if use_bootstrap and n_boot > 0:
                ci_keys = ci_metrics_default if threshold == 0.5 else ci_metrics_workflow
                cis = _bootstrap_binary_ci(
                    y,
                    scores,
                    threshold,
                    ci_keys,
                    n_boot,
                    stratum_seed + int(threshold * 1000),
                )
            else:
                cis = {}

            pred = (scores >= threshold).astype(int)
            cm = confusion_matrix(y, pred, labels=[0, 1])
            if stratum_id == "overall" and threshold == 0.5:
                cm_overall = pd.DataFrame(
                    cm,
                    index=[f"true_{c}" for c in class_names],
                    columns=[f"pred_{c}" for c in class_names],
                )

            for metric in (
                "auc", "auprc", "acc", "f1", "sensitivity", "specificity", "ppv", "npv", "precision", "recall",
            ):
                pt = point.get(metric, float("nan"))
                if metric in cis:
                    lo, hi = cis[metric][1], cis[metric][2]
                    formatted = _fmt_ci(pt, lo, hi)
                else:
                    lo, hi = pt, pt
                    formatted = f"{pt:.3f}" if np.isfinite(pt) else ""
                long_rows.append(
                    {
                        "stage": stage_key,
                        "section": section,
                        "threshold_section": thresh_section,
                        "threshold": threshold,
                        "stratum_id": stratum_id,
                        "stratum_label": stratum_label,
                        "n_samples": n,
                        "metric": metric,
                        "value": pt,
                        "ci_low": lo,
                        "ci_high": hi,
                        "formatted": formatted,
                    }
                )

    prefix = f"table1_{stage_key}"
    long_path = out_dir / f"{prefix}_external_validation_long.csv"
    pd.DataFrame(long_rows).to_csv(long_path, index=False, encoding="utf-8-sig")

    if cm_overall is not None:
        cm_path = out_dir / f"{prefix}_confusion_matrix_overall.csv"
        cm_overall.to_csv(cm_path, encoding="utf-8-sig")

    # Subgroup wide tables (AUC with CI at 0.5)
    sub_rows: List[Dict[str, Any]] = []
    for stratum_idx, (stratum_id, stratum_label, sub) in enumerate(strata):
        if stratum_id == "overall":
            continue
        y = sub["y_true"].astype(int).to_numpy()
        scores = sub["prob_1"].astype(float).to_numpy()
        if len(np.unique(y)) < 2:
            continue
        section = "source" if stratum_id.startswith("source_") else "agerange"
        pt = float(roc_auc_score(y, scores))
        if use_bootstrap and n_boot > 0:
            from _figure_common import bootstrap_auc_ci

            auc, lo, hi = bootstrap_auc_ci(
                y,
                scores,
                n_boot=n_boot,
                seed=stratum_seed,
            )
        else:
            auc, lo, hi = pt, pt, pt
        sub_rows.append(
            {
                "stage": stage_key,
                "section": section,
                "stratum_id": stratum_id,
                "stratum_label": stratum_label,
                "n": int(len(sub)),
                "auc": auc,
                "auc_ci_low": lo,
                "auc_ci_high": hi,
                "auc_formatted": _fmt_ci(auc, lo, hi),
            }
        )
    if sub_rows:
        pd.DataFrame(sub_rows).to_csv(
            out_dir / f"{prefix}_subgroup_auc.csv",
            index=False,
            encoding="utf-8-sig",
        )

    overall_blk = report.get("overall", {})
    workflow_pt = _binary_point_metrics(
        df["y_true"].astype(int).to_numpy(),
        df["prob_1"].astype(float).to_numpy(),
        workflow_threshold,
    )
    summary = {
        "stage": stage_key,
        "n_external": int(overall_blk.get("n_samples", len(df))),
        "model_run": str(run_dir),
        "workflow_threshold": workflow_threshold,
        "bootstrap_iter": n_boot if use_bootstrap else 0,
        "seed": seed if use_bootstrap else None,
        "auc_0.5": _fmt_ci(*_bootstrap_binary_ci(
            df["y_true"].astype(int).to_numpy(),
            df["prob_1"].astype(float).to_numpy(),
            0.5,
            ("auc",),
            n_boot if use_bootstrap else 0,
            seed + seed_offset,
        )["auc"]) if use_bootstrap else f"{overall_blk.get('metrics', {}).get('auc', float('nan')):.3f}",
        "workflow_metrics": {k: workflow_pt[k] for k in ("acc", "sensitivity", "specificity", "ppv", "npv")},
    }
    (out_dir / f"{prefix}_external_validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] {stage_key} tables -> {out_dir}")
    print(f"     {long_path}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    workflow = json.loads(Path(args.workflow_json).read_text(encoding="utf-8"))
    pt1 = float(workflow.get("pt1", 0.58))
    pt2 = float(workflow.get("pt2", 0.39))

    use_bootstrap = not args.no_bootstrap
    n_boot = args.bootstrap_iter if use_bootstrap else 0

    _export_stage("stage1", Path(args.s1_run_dir), pt1, out_dir, n_boot, args.seed, use_bootstrap)
    _export_stage("stage2", Path(args.s2_run_dir), pt2, out_dir, n_boot, args.seed, use_bootstrap)

    print(f"[OK] All Stage12 table data under {out_dir}")


if __name__ == "__main__":
    main()
