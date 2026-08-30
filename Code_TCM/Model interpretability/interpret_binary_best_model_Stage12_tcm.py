# -*- coding: utf-8 -*-
"""
TCM-integrated Stage1/Stage2 interpretability (SHAP composite + PDP).

Publication defaults for integrated models; invokes the shared Stage12 interpret
pipeline with TCM run directories and Figure\\Stage12_TCM output paths.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parent
PYTHON = sys.executable
DATA = PROJECT / "TCM" / "data"
FIG_ROOT = PROJECT / "Figure" / "Stage12_TCM"

S1_RUN = PROJECT / "TCM" / "output" / "Stage1" / "6_run_20260708_052605"
S2_RUN = PROJECT / "TCM" / "output" / "Stage2" / "6_run_20260707_111716"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TCM Stage12 interpret (SHAP + PDP)")
    p.add_argument("--stage-label", choices=["stage1", "stage2"], required=True)
    p.add_argument("--plot-format", type=str, default="jpg", choices=["jpg", "jpeg", "png", "pdf"])
    p.add_argument("--grid-points", type=int, default=25)
    p.add_argument("--plot-max-samples", type=int, default=400)
    p.add_argument("--pdp-max-samples", type=int, default=400)
    p.add_argument("--save-individual-pdp", action="store_true")
    p.add_argument("--composite-only", action="store_true")
    p.add_argument("--pdp-only", action="store_true")
    p.add_argument("--pdp-features", type=str, default=None, help="e.g. ShiSuanBaiFen")
    p.add_argument("--skip-pdp", action="store_true", help="SHAP figures only, skip PDP")
    p.add_argument(
        "--composite-bar-ymax-train",
        type=float,
        default=None,
        help="Train composite bar ymax (Stage2 TCM default 0.14 when stage2, match Stage1)",
    )
    p.add_argument(
        "--composite-bar-ymax-external",
        type=float,
        default=None,
        help="External composite bar ymax (Stage2 TCM default 0.14 when stage2, match Stage1)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage_label == "stage1":
        run_dir = S1_RUN
        train_csv = DATA / "ATrain-Stage1.csv"
        external_csv = DATA / "ATest-Stage1.csv"
    else:
        run_dir = S2_RUN
        train_csv = DATA / "ATrain-Stage2.csv"
        external_csv = DATA / "ATest-Stage2.csv"

    pub_dir = FIG_ROOT / "figure5" / args.stage_label
    cmd = [
        PYTHON,
        str(SCRIPT_DIR / "interpret_binary_best_model_Stage12.py"),
        "--model-run-dir",
        str(run_dir),
        "--train-csv",
        str(train_csv),
        "--external-csv",
        str(external_csv),
        "--stage-label",
        args.stage_label,
        "--plot-format",
        args.plot_format,
        "--pub-figure-dir",
        str(pub_dir),
        "--plot-max-samples",
        str(args.plot_max_samples),
        "--pdp-max-samples",
        str(args.pdp_max_samples),
        "--grid-points",
        str(args.grid_points),
    ]
    if args.save_individual_pdp:
        cmd.append("--save-individual-pdp")
    if args.composite_only:
        cmd.append("--composite-only")
    if args.pdp_only:
        cmd.append("--pdp-only")
    if args.pdp_features:
        cmd.extend(["--pdp-features", args.pdp_features])
    if args.skip_pdp:
        cmd.append("--skip-pdp")
    if args.composite_bar_ymax_train is not None:
        cmd.extend(["--composite-bar-ymax-train", str(args.composite_bar_ymax_train)])
    elif args.stage_label == "stage2":
        # Match Stage1 low-SHAP scale (was 1.2, which made bars look ~10× too short).
        cmd.extend(["--composite-bar-ymax-train", "0.14"])
    if args.composite_bar_ymax_external is not None:
        cmd.extend(["--composite-bar-ymax-external", str(args.composite_bar_ymax_external)])
    elif args.stage_label == "stage2":
        cmd.extend(["--composite-bar-ymax-external", "0.14"])

    # Force live logs from child (Windows often buffers subprocess stdout).
    env = {**dict(**{k: v for k, v in __import__("os").environ.items()}), "PYTHONUNBUFFERED": "1"}
    subprocess.run(cmd, check=True, cwd=str(PROJECT), env=env)
    print(f"[OK] TCM interpret {args.stage_label} -> {pub_dir}")
    # Verify publication copies exist and are fresh for BOTH splits.
    for split in ("train", "external"):
        p = pub_dir / f"{split}_shap_importance_correlation.jpg"
        if p.is_file():
            print(f"    verify {p.name}: size={p.stat().st_size} mtime={p.stat().st_mtime}")
        else:
            print(f"    verify MISSING: {p}")


if __name__ == "__main__":
    main()
