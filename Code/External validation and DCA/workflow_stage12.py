# -*- coding: utf-8 -*-
"""
W2 workflow summary: Stage1 and Stage2 reported separately on external validation.

Stage1 (n=597): triage at pt1 — organic vs functional.
Stage2 (n=459): UC discrimination at pt2 — UC vs CD+IC+CRC.

Examples::

  python workflow_stage12.py ^
    --s1-run-dir "F:\\KeTi\\Project\\outputs\\Stage1\\3_run_20260707_142952" ^
    --s2-run-dir "F:\\KeTi\\Project\\outputs\\Stage2\\4_run_20260707_030540" ^
    --pt1-from "F:\\KeTi\\Project\\outputs\\Stage1\\dca_batch\\3_run_20260707_142952\\overall\\operating_point.csv" ^
    --pt2-from "F:\\KeTi\\Project\\outputs\\Stage2\\dca_batch\\4_run_20260707_030540\\overall\\operating_point.csv" ^
    --stage-path western_best_cross ^
    --out-dir "F:\\KeTi\\Project\\outputs\\workflow_batch\\western\\03_4_best_cross"

  python workflow_stage12.py ^
    --s1-run-dir "F:\\KeTi\\Project\\TCM\\output\\Stage1\\6_run_20260708_052605" ^
    --s2-run-dir "F:\\KeTi\\Project\\TCM\\output\\Stage2\\6_run_20260707_111716" ^
    --pt1-from "F:\\KeTi\\Project\\TCM\\output\\Stage1\\dca_batch\\6_run_20260708_052605\\overall\\operating_point.csv" ^
    --pt2-from "F:\\KeTi\\Project\\TCM\\output\\Stage2\\dca_batch\\6_run_20260707_111716\\overall\\operating_point.csv" ^
    --stage-path TCM_best_cross ^
    --out-dir "F:\\KeTi\\Project\\TCM\\output\\workflow_batch\\tcm\\06_6_best_cross"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from _dca_common import load_predictions, metrics_at_threshold, resolve_pred_csv  # noqa: E402


STAGE1_POSITIVE_LABEL = "organic (UC+CD+IC+CRC)"
STAGE1_NEGATIVE_LABEL = "functional (IE+FDIBS)"
STAGE2_POSITIVE_LABEL = "UC"
STAGE2_NEGATIVE_LABEL = "non-UC (CD+IC+CRC)"


def _read_pt_from_operating_csv(path: Path) -> float:
    df = pd.read_csv(path)
    if "threshold" not in df.columns:
        raise ValueError(f"Missing threshold column in {path}")
    return float(df.iloc[0]["threshold"])


def _resolve_pt(explicit: float | None, operating_csv: str | None) -> float | None:
    if explicit is not None:
        return float(explicit)
    if operating_csv:
        return _read_pt_from_operating_csv(Path(operating_csv))
    return None


def _panel_summary(
    df: pd.DataFrame,
    pt: float,
    stage_name: str,
    positive_label: str,
    negative_label: str,
) -> dict:
    y = df["y_true"].astype(int).to_numpy()
    prob = df["prob_1"].astype(float).to_numpy()
    m = metrics_at_threshold(y, prob, pt)
    referred = int((prob >= pt).sum())
    not_referred = int(len(prob) - referred)

    by_type = None
    if "Type" in df.columns:
        tmp = df.copy()
        tmp["pred_positive"] = (tmp["prob_1"].astype(float) >= pt).astype(int)
        by_type = (
            tmp.groupby("Type")
            .agg(n=("y_true", "size"), pred_pos_rate=("pred_positive", "mean"), true_pos_rate=("y_true", "mean"))
            .reset_index()
            .to_dict(orient="records")
        )

    return {
        "stage": stage_name,
        "n": int(len(df)),
        "threshold": float(pt),
        "positive_class": positive_label,
        "negative_class": negative_label,
        "referred_or_uc_path_n": referred,
        "not_referred_n": not_referred,
        "confusion": {
            "tp": m["tp"],
            "tn": m["tn"],
            "fp": m["fp"],
            "fn": m["fn"],
        },
        "metrics_at_threshold": m,
        "note": (
            "Illustrative sequential framework (W2): stages evaluated on their respective "
            "external cohorts; not a formal probability cascade."
        ),
        "by_type": by_type,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="W2 workflow table for Stage1 + Stage2 external validation.")
    p.add_argument("--s1-run-dir", type=str, default=None)
    p.add_argument("--s2-run-dir", type=str, default=None)
    p.add_argument("--s1-pred-csv", type=str, default=None)
    p.add_argument("--s2-pred-csv", type=str, default=None)
    p.add_argument("--pt1", type=float, default=None, help="Stage1 operating threshold.")
    p.add_argument("--pt2", type=float, default=None, help="Stage2 operating threshold.")
    p.add_argument(
        "--pt1-from",
        type=str,
        default=None,
        help="operating_point.csv from Stage1 DCA (used if --pt1 omitted).",
    )
    p.add_argument(
        "--pt2-from",
        type=str,
        default=None,
        help="operating_point.csv from Stage2 DCA (used if --pt2 omitted).",
    )
    p.add_argument("--stage-path", type=str, default="model", help="Tag in output, e.g. western / tcm.")
    p.add_argument("--out-dir", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.s1_pred_csv:
        s1_csv = Path(args.s1_pred_csv)
    elif args.s1_run_dir:
        s1_csv = resolve_pred_csv(Path(args.s1_run_dir))
    else:
        raise SystemExit("Provide --s1-run-dir or --s1-pred-csv")

    if args.s2_pred_csv:
        s2_csv = Path(args.s2_pred_csv)
    elif args.s2_run_dir:
        s2_csv = resolve_pred_csv(Path(args.s2_run_dir))
    else:
        raise SystemExit("Provide --s2-run-dir or --s2-pred-csv")

    pt1 = _resolve_pt(args.pt1, args.pt1_from)
    pt2 = _resolve_pt(args.pt2, args.pt2_from)
    if pt1 is None or pt2 is None:
        raise SystemExit("Provide --pt1/--pt2 or --pt1-from/--pt2-from (DCA operating_point.csv).")

    s1_df = load_predictions(s1_csv)
    s2_df = load_predictions(s2_csv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_s1 = _panel_summary(
        s1_df,
        pt1,
        "Stage1_triage",
        STAGE1_POSITIVE_LABEL,
        STAGE1_NEGATIVE_LABEL,
    )
    panel_s2 = _panel_summary(
        s2_df,
        pt2,
        "Stage2_uc_discrimination",
        STAGE2_POSITIVE_LABEL,
        STAGE2_NEGATIVE_LABEL,
    )

    workflow = {
        "framework": "W2_separate_external_stages",
        "stage_path": args.stage_path,
        "s1_predictions": str(s1_csv),
        "s2_predictions": str(s2_csv),
        "pt1": pt1,
        "pt2": pt2,
        "panel_stage1": panel_s1,
        "panel_stage2": panel_s2,
    }

    with open(out_dir / "workflow_stage12.json", "w", encoding="utf-8") as f:
        json.dump(workflow, f, ensure_ascii=False, indent=2, default=str)

    rows = []
    for key, panel in ("panel_stage1", panel_s1), ("panel_stage2", panel_s2):
        m = panel["metrics_at_threshold"]
        rows.append(
            {
                "panel": panel["stage"],
                "n": panel["n"],
                "threshold": panel["threshold"],
                "tp": m["tp"],
                "tn": m["tn"],
                "fp": m["fp"],
                "fn": m["fn"],
                "sensitivity": m["sensitivity"],
                "specificity": m["specificity"],
                "ppv": m["ppv"],
                "npv": m["npv"],
                "referred_or_uc_path_n": panel["referred_or_uc_path_n"],
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "workflow_stage12_summary.csv", index=False)

    if panel_s1.get("by_type"):
        pd.DataFrame(panel_s1["by_type"]).to_csv(out_dir / "workflow_stage1_by_type.csv", index=False)
    if panel_s2.get("by_type"):
        pd.DataFrame(panel_s2["by_type"]).to_csv(out_dir / "workflow_stage2_by_type.csv", index=False)

    print(f"[OK] workflow W2 -> {out_dir}")
    print(f"     Stage1 n={panel_s1['n']} @ pt1={pt1:.4f}: TP={panel_s1['confusion']['tp']} FP={panel_s1['confusion']['fp']}")
    print(f"     Stage2 n={panel_s2['n']} @ pt2={pt2:.4f}: TP={panel_s2['confusion']['tp']} FP={panel_s2['confusion']['fp']}")


if __name__ == "__main__":
    main()
