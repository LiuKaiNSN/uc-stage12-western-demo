# -*- coding: utf-8 -*-
"""Smoke test for cloud deploy bundle (run from deploy repo root)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config_loader import load_config
from src.inference import load_stage_artifacts, predict_from_dataframe

PROJECT = Path(r"F:/KeTi/Project")


def main() -> None:
    cfg = load_config()
    for stage in ("stage1", "stage2"):
        art = load_stage_artifacts(cfg, stage)
        csv = PROJECT / "Data" / f"ATest-Stage{stage[-1]}.csv"
        df = pd.read_csv(csv)
        result = predict_from_dataframe(art, df.iloc[[0]])
        print(stage, json.dumps(result.to_dict(), ensure_ascii=False))


if __name__ == "__main__":
    main()
