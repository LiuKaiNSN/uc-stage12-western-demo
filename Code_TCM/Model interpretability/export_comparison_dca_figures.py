# -*- coding: utf-8 -*-
"""
Export TCM vs Western DCA composite figures (main + supplement).

Output: Figure\\比较\\DCA\\
  Main:
    - comparison_dca_stage1_main_lgbm3_rf6.jpg
    - comparison_dca_stage2_main_catboost4_rf6.jpg
    - comparison_dca_stage3_main_lightgbm3_e3.jpg
  Supplement:
    - comparison_dca_stage1_supplement_overall_ml1-10_excl3_6.jpg
    - comparison_dca_stage2_supplement_overall_ml1-10_excl4_6.jpg
    - comparison_dca_stage3_supplement_e3_ml1-10_excl3.jpg
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from _comparison_figures_common import (
    build_overall_grid_figure,
    build_stage1_main_figure,
    build_stage2_main_figure_multi,
    build_stage3_main_figure,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parent
DEFAULT_OUT = PROJECT / "Figure" / "比较" / "DCA"
DEFAULT_S1_BATCH = PROJECT / "Stage1-dca_overlay_batch"
DEFAULT_S2_BATCH = PROJECT / "Stage2-dca_overlay_batch"
DEFAULT_S3_BATCH = PROJECT / "Stage3-dca_overlay_batch"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TCM vs Western DCA composite figures")
    p.add_argument("--stage1-batch-root", type=str, default=str(DEFAULT_S1_BATCH))
    p.add_argument("--stage2-batch-root", type=str, default=str(DEFAULT_S2_BATCH))
    p.add_argument("--stage3-batch-root", type=str, default=str(DEFAULT_S3_BATCH))
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    s1_root = Path(args.stage1_batch_root)
    s2_root = Path(args.stage2_batch_root)
    s3_root = Path(args.stage3_batch_root)

    outputs: dict[str, str] = {}

    main_s1 = out_dir / "comparison_dca_stage1_main_lgbm3_rf6.jpg"
    build_stage1_main_figure(s1_root, main_s1, indices=(3, 6))
    outputs["main_stage1"] = str(main_s1)

    main_s2 = out_dir / "comparison_dca_stage2_main_catboost4_rf6.jpg"
    build_stage2_main_figure_multi(s2_root, main_s2, indices=(4, 6))
    outputs["main_stage2"] = str(main_s2)

    main_s3 = out_dir / "comparison_dca_stage3_main_lightgbm3_e3.jpg"
    build_stage3_main_figure(s3_root, main_s3, model_index=3)
    outputs["main_stage3"] = str(main_s3)

    sup_s1 = out_dir / "comparison_dca_stage1_supplement_overall_ml1-10_excl3_6.jpg"
    build_overall_grid_figure(s1_root, "stage1", sup_s1, [1, 2, 4, 5, 7, 8, 9, 10], ncol=4)
    outputs["supplement_stage1"] = str(sup_s1)

    sup_s2 = out_dir / "comparison_dca_stage2_supplement_overall_ml1-10_excl4_6.jpg"
    build_overall_grid_figure(s2_root, "stage2", sup_s2, [i for i in range(1, 11) if i not in (4, 6)], ncol=3)
    outputs["supplement_stage2"] = str(sup_s2)

    sup_s3 = out_dir / "comparison_dca_stage3_supplement_e3_ml1-10_excl3.jpg"
    build_overall_grid_figure(s3_root, "stage3", sup_s3, [i for i in range(1, 11) if i != 3], ncol=3)
    outputs["supplement_stage3"] = str(sup_s3)

    manifest = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "dpi": 300,
        "format": "jpg",
        "batch_roots": {
            "stage1": str(s1_root.resolve()),
            "stage2": str(s2_root.resolve()),
            "stage3": str(s3_root.resolve()),
        },
        "outputs": outputs,
        "layout": {
            "main_stage1": "2 rows: LightGBM(3) Western best + RF(6) TCM best (overall+disease panels each)",
            "main_stage2": "2 blocks: CatBoost(4) Western best + RF(6) TCM best (overall+disease panels each)",
            "main_stage3": "single: LightGBM(3) E3 vs E1+E2",
            "supplement": "overall overlay only (no disease panels)",
        },
    }
    manifest_path = out_dir / "comparison_dca_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] DCA comparison figures -> {out_dir}")
    for k, v in outputs.items():
        print(f"     {k}: {v}")


if __name__ == "__main__":
    main()
