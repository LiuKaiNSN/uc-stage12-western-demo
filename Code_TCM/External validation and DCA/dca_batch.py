# -*- coding: utf-8 -*-
"""
Batch DCA for run folders 1–10 under a Stage root (skips 旧/旧2).

Examples::

  python dca_batch.py ^
    --stage-root "F:\\KeTi\\Project\\outputs\\Stage1" ^
    --stage stage1 ^
    --label-model "Machine learning model"

  python dca_batch.py ^
    --stage-root "F:\\KeTi\\Project\\TCM\\output\\Stage2" ^
    --stage stage2 ^
    --with-slices ^
    --label-model "TCM-integrated model"

  python dca_batch.py ^
    --stage-root "F:\\KeTi\\Project\\outputs\\Stage1" ^
    --stage stage1 ^
    --with-overlay ^
    --pair-root "F:\\KeTi\\Project\\TCM\\output\\Stage1" ^
    --with-slices ^
    --label-model "Machine learning model" ^
    --label-western "Western biomarker model" ^
    --label-tcm "TCM-integrated model"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from _dca_common import (  # noqa: E402
    compute_dca,
    discover_run_dirs,
    disease_slices_for_stage,
    load_predictions,
    make_thresholds,
    merge_overlay_predictions,
    pair_run_by_index,
    pick_operating_point,
    plot_dca_multipanel_overlay,
    plot_dca_multipanel_single_model,
    resolve_pred_csv,
    run_index_from_dir,
    run_overlay_dca,
    run_single_dca,
    slice_pred_csv,
    stage_defaults,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch DCA for all model runs under a Stage directory.")
    p.add_argument("--stage-root", type=str, required=True, help="e.g. F:\\KeTi\\Project\\outputs\\Stage1")
    p.add_argument("--stage", choices=["stage1", "stage2"], required=True)
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Default: {stage-root}/dca_batch",
    )
    p.add_argument("--label-model", type=str, default="Machine learning model")
    p.add_argument("--with-slices", action="store_true", help="Disease subset DCA + multipanel figure per run.")
    p.add_argument(
        "--with-overlay",
        action="store_true",
        help="Also pair with --pair-root (Western=stage-root, TCM=pair-root) by run index.",
    )
    p.add_argument(
        "--pair-root",
        type=str,
        default=None,
        help="TCM Stage root when stage-root is Western (or vice versa).",
    )
    p.add_argument("--label-western", type=str, default="Western biomarker model")
    p.add_argument("--label-tcm", type=str, default="TCM-integrated model")
    p.add_argument("--pt-min", type=float, default=0.01)
    p.add_argument("--pt-max", type=float, default=0.80)
    p.add_argument("--pt-step", type=float, default=0.01)
    p.add_argument("--plot-min", type=float, default=None)
    p.add_argument("--plot-max", type=float, default=None)
    p.add_argument("--min-sens", type=float, default=None)
    p.add_argument("--run-indices", type=str, default=None, help="Optional comma list, e.g. 1,3,4,6")
    return p.parse_args()


def _filter_runs(runs: list[Path], run_indices: str | None) -> list[Path]:
    if not run_indices:
        return runs
    wanted = {int(x.strip()) for x in run_indices.split(",") if x.strip()}
    return [r for r in runs if run_index_from_dir(r) in wanted]


def process_single_run(
    run_dir: Path,
    stage: str,
    batch_out: Path,
    label_model: str,
    with_slices: bool,
    pt_min: float,
    pt_max: float,
    pt_step: float,
    plot_min: float,
    plot_max: float,
    min_sens: float,
) -> dict:
    run_out = batch_out / run_dir.name
    overall_out = run_out / "overall"
    df = load_predictions(resolve_pred_csv(run_dir))
    meta = run_single_dca(
        df,
        stage=stage,
        out_dir=overall_out,
        label_model=label_model,
        pt_min=pt_min,
        pt_max=pt_max,
        pt_step=pt_step,
        plot_min=plot_min,
        plot_max=plot_max,
        min_sens=min_sens,
        slice_id="overall",
        title_suffix="overall",
    )

    if with_slices:
        panel_rows = []
        thresholds = make_thresholds(pt_min, pt_max, pt_step)
        for sid in disease_slices_for_stage(stage):
            slice_csv = slice_pred_csv(run_dir, stage, sid)
            slice_out = run_out / "slices" / sid
            sdf = load_predictions(slice_csv)
            run_single_dca(
                sdf,
                stage=stage,
                out_dir=slice_out,
                label_model=label_model,
                pt_min=pt_min,
                pt_max=pt_max,
                pt_step=pt_step,
                plot_min=plot_min,
                plot_max=plot_max,
                min_sens=min_sens,
                slice_id=sid,
                title_suffix=sid,
            )
            y, prob = sdf["y_true"].to_numpy(), sdf["prob_1"].to_numpy()
            dca_df = compute_dca(y, prob, thresholds)
            op = pick_operating_point(y, prob, thresholds, min_sens=min_sens)
            plot_df = dca_df[(dca_df["threshold"] >= plot_min) & (dca_df["threshold"] <= plot_max)]
            panel_rows.append((sid, plot_df, float(op["threshold"])))
        plot_dca_multipanel_single_model(
            panel_rows,
            label_model=label_model,
            stage=stage,
            out_png=run_out / "dca_disease_panels.png",
            suptitle=f"Disease-subset DCA ({stage})",
        )

    return {"run_dir": str(run_dir), "run_index": run_index_from_dir(run_dir), **meta}


def process_overlay_run(
    western_run: Path,
    tcm_run: Path,
    stage: str,
    batch_out: Path,
    label_western: str,
    label_tcm: str,
    with_slices: bool,
    pt_min: float,
    pt_max: float,
    pt_step: float,
    plot_min: float,
    plot_max: float,
    min_sens: float,
) -> dict:
    idx = run_index_from_dir(western_run)
    run_out = batch_out / f"overlay_{idx:02d}_{western_run.name}"
    meta = run_overlay_dca(
        resolve_pred_csv(western_run),
        resolve_pred_csv(tcm_run),
        stage=stage,
        out_dir=run_out / "overall",
        label_western=label_western,
        label_tcm=label_tcm,
        pt_min=pt_min,
        pt_max=pt_max,
        pt_step=pt_step,
        plot_min=plot_min,
        plot_max=plot_max,
        min_sens=min_sens,
        slice_id="overall",
    )

    if with_slices:
        panel_data = []
        for sid in disease_slices_for_stage(stage):
            w_csv = slice_pred_csv(western_run, stage, sid)
            t_csv = slice_pred_csv(tcm_run, stage, sid)
            sub_out = run_out / "slices" / sid
            run_overlay_dca(
                w_csv,
                t_csv,
                stage=stage,
                out_dir=sub_out,
                label_western=label_western,
                label_tcm=label_tcm,
                pt_min=pt_min,
                pt_max=pt_max,
                pt_step=pt_step,
                plot_min=plot_min,
                plot_max=plot_max,
                min_sens=min_sens,
                slice_id=sid,
            )
            y, pw, pt = merge_overlay_predictions(w_csv, t_csv)
            th = make_thresholds(pt_min, pt_max, pt_step)
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
            label_western=label_western,
            label_tcm=label_tcm,
            stage=stage,
            out_png=run_out / "dca_overlay_disease_panels.png",
            suptitle=f"Overlay disease-subset DCA ({stage})",
        )
    return {"western_run": str(western_run), "tcm_run": str(tcm_run), "run_index": idx, **meta}


def main() -> None:
    args = parse_args()
    stage = args.stage.lower()
    defaults = stage_defaults(stage)
    plot_min = args.plot_min if args.plot_min is not None else defaults["plot_min"]
    plot_max = args.plot_max if args.plot_max is not None else defaults["plot_max"]
    min_sens = args.min_sens if args.min_sens is not None else defaults["min_sens"]

    stage_root = Path(args.stage_root)
    batch_out = Path(args.out_dir) if args.out_dir else stage_root / "dca_batch"
    batch_out.mkdir(parents=True, exist_ok=True)

    runs = _filter_runs(discover_run_dirs(stage_root), args.run_indices)
    if not runs:
        raise SystemExit(f"No run folders with external predictions under {stage_root}")

    if args.with_overlay and not args.pair_root:
        raise SystemExit("--with-overlay requires --pair-root")

    pair_root = Path(args.pair_root) if args.pair_root else None
    summary: list[dict] = []

    for run_dir in runs:
        print(f"Processing {run_dir.name} ...")
        summary.append(
            process_single_run(
                run_dir,
                stage,
                batch_out,
                args.label_model,
                args.with_slices,
                args.pt_min,
                args.pt_max,
                args.pt_step,
                plot_min,
                plot_max,
                min_sens,
            )
        )

        if args.with_overlay and pair_root is not None:
            idx = run_index_from_dir(run_dir)
            if idx is None:
                continue
            w_run, t_run = pair_run_by_index(stage_root, pair_root, idx)
            if w_run is None or t_run is None:
                print(f"  [skip overlay] missing pair for index {idx}")
            else:
                print(f"  overlay pair: {w_run.name} vs {t_run.name}")
                summary.append(
                    process_overlay_run(
                        w_run,
                        t_run,
                        stage,
                        batch_out,
                        args.label_western,
                        args.label_tcm,
                        args.with_slices,
                        args.pt_min,
                        args.pt_max,
                        args.pt_step,
                        plot_min,
                        plot_max,
                        min_sens,
                    )
                )

    manifest = {
        "stage_root": str(stage_root),
        "stage": stage,
        "n_runs": len(runs),
        "with_slices": args.with_slices,
        "with_overlay": args.with_overlay,
        "pair_root": args.pair_root,
        "runs": summary,
    }
    manifest_path = batch_out / "dca_batch_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
    print(f"[DONE] {len(runs)} runs -> {batch_out}")
    print(f"       manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
