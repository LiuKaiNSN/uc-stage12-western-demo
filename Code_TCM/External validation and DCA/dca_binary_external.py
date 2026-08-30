# -*- coding: utf-8 -*-
"""
Single-model decision curve analysis on external validation predictions.

Examples::

  python dca_binary_external.py ^
    --model-run-dir "F:\\KeTi\\Project\\outputs\\Stage1\\3_run_20260707_142952" ^
    --stage stage1 ^
    --out-dir "F:\\KeTi\\Project\\outputs\\Stage1\\dca_batch\\3_run_20260707_142952\\overall" ^
    --label-model "Machine learning model"

  python dca_binary_external.py ^
    --pred-csv "F:\\KeTi\\Project\\outputs\\Stage2\\4_run_20260707_030540\\external_validation_binary\\type_montreal_slices\\predictions_uc_vs_cd.csv" ^
    --stage stage2 ^
    --out-dir "F:\\KeTi\\Project\\outputs\\Stage2\\dca_batch\\4_run_20260707_030540\\slices\\uc_vs_cd" ^
    --slice-id uc_vs_cd ^
    --label-model "Machine learning model"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from _dca_common import (  # noqa: E402
    load_predictions,
    resolve_pred_csv,
    run_single_dca,
    slice_pred_csv,
    stage_defaults,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Binary DCA on external validation predictions.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--model-run-dir", type=str, help="Run folder containing external_validation_binary/.")
    src.add_argument("--pred-csv", type=str, help="Direct path to predictions CSV (overall or slice).")

    p.add_argument("--stage", choices=["stage1", "stage2"], required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--label-model", type=str, default="Machine learning model")

    p.add_argument("--slice-id", type=str, default=None, help="Optional slice name for titles/outputs.")
    p.add_argument(
        "--slice-pred-csv",
        type=str,
        default=None,
        help="If set with --model-run-dir, use this slice CSV instead of overall predictions.",
    )

    p.add_argument("--pt-min", type=float, default=0.01)
    p.add_argument("--pt-max", type=float, default=0.80)
    p.add_argument("--pt-step", type=float, default=0.01)
    p.add_argument("--plot-min", type=float, default=None)
    p.add_argument("--plot-max", type=float, default=None)
    p.add_argument("--min-sens", type=float, default=None, help="Stage1 default 0.95; Stage2 default 0.90.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    stage = args.stage.lower()
    defaults = stage_defaults(stage)

    if args.pred_csv:
        pred_path = Path(args.pred_csv)
        slice_id = args.slice_id or pred_path.stem.replace("predictions_", "")
    elif args.slice_pred_csv:
        pred_path = Path(args.slice_pred_csv)
        slice_id = args.slice_id or pred_path.stem.replace("predictions_", "")
    elif args.slice_id:
        pred_path = slice_pred_csv(Path(args.model_run_dir), stage, args.slice_id)
        slice_id = args.slice_id
    else:
        pred_path = resolve_pred_csv(Path(args.model_run_dir))
        slice_id = "overall"

    df = load_predictions(pred_path)
    meta = run_single_dca(
        df,
        stage=stage,
        out_dir=Path(args.out_dir),
        label_model=args.label_model,
        pt_min=args.pt_min,
        pt_max=args.pt_max,
        pt_step=args.pt_step,
        plot_min=args.plot_min if args.plot_min is not None else defaults["plot_min"],
        plot_max=args.plot_max if args.plot_max is not None else defaults["plot_max"],
        min_sens=args.min_sens if args.min_sens is not None else defaults["min_sens"],
        slice_id=slice_id,
        title_suffix=slice_id,
    )

    op = meta["operating_point"]
    print(f"[OK] {pred_path}")
    print(f"     n={meta['n']}, prevalence={meta['prevalence']:.3f}, slice={slice_id}")
    print(
        f"     operating pt={op['threshold']:.4f}, sens={op['sensitivity']:.4f}, spec={op['specificity']:.4f}"
    )
    print(f"     useful pt range: [{meta['useful_pt_min']}, {meta['useful_pt_max']}]")
    print(f"     outputs -> {args.out_dir}")


if __name__ == "__main__":
    main()
