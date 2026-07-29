# -*- coding: utf-8 -*-
"""
Export nested-CV internal validation tables for Western Stage1 + Stage2 (best models).

Outputs under Figure\\Stage12\\data\\:
  internal_cv_stage1_*.csv, internal_cv_stage2_*.csv
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from _internal_cv_export import export_binary_internal_cv

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parent
DEFAULT_S1 = PROJECT / "outputs" / "Stage1" / "3_run_20260707_142952"
DEFAULT_S2 = PROJECT / "outputs" / "Stage2" / "4_run_20260707_030540"
DEFAULT_OUT = PROJECT / "Figure" / "Stage12" / "data"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Western Stage12 internal CV tables (Stage1 + Stage2)")
    p.add_argument("--s1-run-dir", type=str, default=str(DEFAULT_S1))
    p.add_argument("--s2-run-dir", type=str, default=str(DEFAULT_S2))
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--bootstrap-iter", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    m1 = export_binary_internal_cv(
        Path(args.s1_run_dir),
        "stage1",
        out_dir,
        threshold=args.threshold,
        bootstrap_iter=args.bootstrap_iter,
        seed=args.seed,
    )
    m2 = export_binary_internal_cv(
        Path(args.s2_run_dir),
        "stage2",
        out_dir,
        threshold=args.threshold,
        bootstrap_iter=args.bootstrap_iter,
        seed=args.seed + 1,
    )

    bundle = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "paper": "Western Stage12",
        "stage1": m1,
        "stage2": m2,
    }
    bundle_path = out_dir / "internal_cv_western_stage12_bundle.json"
    bundle_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Western Stage12 internal CV -> {out_dir}")
    print(f"     Stage1: {m1['model']} ({m1['model_run_dir']})")
    print(f"     Stage2: {m2['model']} ({m2['model_run_dir']})")


if __name__ == "__main__":
    main()
