# -*- coding: utf-8 -*-
"""
Paper2 Fig2 / Fig3: ΔAUC forest for Stage1+Stage2 only (no Stage3).

Outputs (default -> Figure/比较/For submission only):
  - Fig2_delta_auc_forest_oof_stage12.jpg
  - Fig3_delta_auc_forest_external_stage12.jpg
  - comparison_delta_forest_stage12_oof.csv
  - comparison_delta_forest_stage12_external.csv
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from _comparison_figures_common import build_statistics_forest_figure_stage12

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parent
DEFAULT_OUT = PROJECT / "Figure" / "比较" / "For submission only"
DEFAULT_S1 = PROJECT / "Stage1_TCM_vs_Western_comparison" / "comparison_20260713_190255"
DEFAULT_S2 = PROJECT / "Stage2_TCM_vs_Western_comparison" / "comparison_20260713_190605"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper2 Stage12 ΔAUC forest (OOF + external)")
    p.add_argument("--stage1-comparison-dir", type=str, default=str(DEFAULT_S1))
    p.add_argument("--stage2-comparison-dir", type=str, default=str(DEFAULT_S2))
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    s1 = Path(args.stage1_comparison_dir)
    s2 = Path(args.stage2_comparison_dir)

    oof_jpg = out_dir / "Fig2_delta_auc_forest_oof_stage12.jpg"
    ext_jpg = out_dir / "Fig3_delta_auc_forest_external_stage12.jpg"

    df_oof = build_statistics_forest_figure_stage12(
        s1,
        s2,
        oof_jpg,
        dataset="oof",
        title="TCM-integrated vs Baseline Model — internal validation (OOF; models 1–6)",
    )
    df_ext = build_statistics_forest_figure_stage12(
        s1,
        s2,
        ext_jpg,
        dataset="external",
        title="TCM-integrated vs Baseline Model — external validation (models 1–6)",
    )

    oof_csv = out_dir / "comparison_delta_forest_stage12_oof.csv"
    ext_csv = out_dir / "comparison_delta_forest_stage12_external.csv"
    for df, path in ((df_oof, oof_csv), (df_ext, ext_csv)):
        out = df.rename(
            columns={
                "auc_western": "auc_baseline",
                "western_run_dir": "baseline_run_dir",
            }
        )
        if "metric_label" in out.columns:
            out["metric_label"] = out["metric_label"].astype(str).str.replace(
                "Western", "Baseline", regex=False
            )
        out.to_csv(path, index=False, encoding="utf-8-sig")

    manifest = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "fig2_oof_jpg": str(oof_jpg),
        "fig3_external_jpg": str(ext_jpg),
        "oof_csv": str(oof_csv),
        "external_csv": str(ext_csv),
        "stage1_comparison_dir": str(s1.resolve()),
        "stage2_comparison_dir": str(s2.resolve()),
        "note": "Stage1+Stage2 only; Stage3 excluded; red = FDR < 0.05",
    }
    (out_dir / "manifest_fig2_fig3_delta_forest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK] Fig2 -> {oof_jpg}")
    print(f"[OK] Fig3 -> {ext_jpg}")


if __name__ == "__main__":
    main()
