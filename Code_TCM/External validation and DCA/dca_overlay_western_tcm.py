# -*- coding: utf-8 -*-
"""
Western vs TCM-integrated overlay DCA (same algorithm run index, merged by No).

Examples::

  python dca_overlay_western_tcm.py ^
    --western-run-dir "F:\\KeTi\\Project\\outputs\\Stage2\\4_run_20260707_030540" ^
    --tcm-run-dir "F:\\KeTi\\Project\\TCM\\output\\Stage2\\6_run_20260707_111716" ^
    --stage stage2 ^
    --out-dir "F:\\KeTi\\Project\\outputs\\Stage2\\dca_overlay\\6_run\\overall" ^
    --label-western "Western biomarker model" ^
    --label-tcm "TCM-integrated model" ^
    --with-disease-panels
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from _dca_common import (  # noqa: E402
    compute_dca,
    disease_slices_for_stage,
    make_thresholds,
    merge_overlay_predictions,
    plot_dca_multipanel_overlay,
    resolve_pred_csv,
    run_overlay_dca,
    slice_pred_csv,
    stage_defaults,
)


def _resolve_pair_csv(
    western_run_dir: Path | None,
    tcm_run_dir: Path | None,
    western_csv: Path | None,
    tcm_csv: Path | None,
    stage: str,
    slice_id: str | None,
) -> tuple[Path, Path]:
    if western_csv and tcm_csv:
        return Path(western_csv), Path(tcm_csv)
    if western_run_dir is None or tcm_run_dir is None:
        raise ValueError("Provide both --western-run-dir and --tcm-run-dir, or both --pred-western and --pred-tcm.")
    if slice_id:
        return (
            slice_pred_csv(Path(western_run_dir), stage, slice_id),
            slice_pred_csv(Path(tcm_run_dir), stage, slice_id),
        )
    return resolve_pred_csv(Path(western_run_dir)), resolve_pred_csv(Path(tcm_run_dir))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Overlay DCA: Western vs TCM-integrated models.")
    p.add_argument("--western-run-dir", type=str, default=None)
    p.add_argument("--tcm-run-dir", type=str, default=None)
    p.add_argument("--pred-western", type=str, default=None)
    p.add_argument("--pred-tcm", type=str, default=None)
    p.add_argument("--stage", choices=["stage1", "stage2"], required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--label-western", type=str, default="Western biomarker model")
    p.add_argument("--label-tcm", type=str, default="TCM-integrated model")
    p.add_argument("--slice-id", type=str, default=None, help="overall if omitted")
    p.add_argument("--with-disease-panels", action="store_true", help="Also write multipanel overlay for disease slices.")
    p.add_argument("--pt-min", type=float, default=0.01)
    p.add_argument("--pt-max", type=float, default=0.80)
    p.add_argument("--pt-step", type=float, default=0.01)
    p.add_argument("--plot-min", type=float, default=None)
    p.add_argument("--plot-max", type=float, default=None)
    p.add_argument("--min-sens", type=float, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    stage = args.stage.lower()
    defaults = stage_defaults(stage)
    plot_min = args.plot_min if args.plot_min is not None else defaults["plot_min"]
    plot_max = args.plot_max if args.plot_max is not None else defaults["plot_max"]
    min_sens = args.min_sens if args.min_sens is not None else defaults["min_sens"]
    out_dir = Path(args.out_dir)

    slice_id = args.slice_id or "overall"
    w_csv, t_csv = _resolve_pair_csv(
        Path(args.western_run_dir) if args.western_run_dir else None,
        Path(args.tcm_run_dir) if args.tcm_run_dir else None,
        Path(args.pred_western) if args.pred_western else None,
        Path(args.pred_tcm) if args.pred_tcm else None,
        stage,
        args.slice_id,
    )

    meta = run_overlay_dca(
        w_csv,
        t_csv,
        stage=stage,
        out_dir=out_dir,
        label_western=args.label_western,
        label_tcm=args.label_tcm,
        pt_min=args.pt_min,
        pt_max=args.pt_max,
        pt_step=args.pt_step,
        plot_min=plot_min,
        plot_max=plot_max,
        min_sens=min_sens,
        slice_id=slice_id,
    )
    print(f"[OK] overlay {slice_id}: n={meta['n']}, out={out_dir}")

    if args.with_disease_panels and args.slice_id is None and args.western_run_dir and args.tcm_run_dir:
        w_root = Path(args.western_run_dir)
        t_root = Path(args.tcm_run_dir)
        panel_data = []
        for sid in disease_slices_for_stage(stage):
            sub_out = out_dir / "slices" / sid
            w_slice = slice_pred_csv(w_root, stage, sid)
            t_slice = slice_pred_csv(t_root, stage, sid)
            run_overlay_dca(
                w_slice,
                t_slice,
                stage=stage,
                out_dir=sub_out,
                label_western=args.label_western,
                label_tcm=args.label_tcm,
                pt_min=args.pt_min,
                pt_max=args.pt_max,
                pt_step=args.pt_step,
                plot_min=plot_min,
                plot_max=plot_max,
                min_sens=min_sens,
                slice_id=sid,
            )
            y, pw, pt = merge_overlay_predictions(w_slice, t_slice)
            th = make_thresholds(args.pt_min, args.pt_max, args.pt_step)
            dca_w = compute_dca(y, pw, th)
            dca_t = compute_dca(y, pt, th)
            panel_data.append(
                (
                    sid,
                    dca_w[(dca_w["threshold"] >= plot_min) & (dca_w["threshold"] <= plot_max)],
                    dca_t[(dca_t["threshold"] >= plot_min) & (dca_t["threshold"] <= plot_max)],
                )
            )
        plot_dca_multipanel_overlay(
            panel_data,
            label_western=args.label_western,
            label_tcm=args.label_tcm,
            stage=stage,
            out_png=out_dir / "dca_overlay_disease_panels.png",
            suptitle=f"Overlay DCA by disease subset ({stage})",
        )
        print(f"[OK] disease overlay panels -> {out_dir / 'dca_overlay_disease_panels.png'}")


if __name__ == "__main__":
    main()
