# -*- coding: utf-8 -*-
"""
Enrich Paper2 Table1/Table2 (and source comparison CSVs) with absolute AUC
bootstrap 95% CIs from the *same* paired bootstrap as ΔAUC
(n_boot=2000, seed=42 by default).

Does not recompute FDR / p-values; only adds / overwrites absolute-AUC CI
columns and formatted display strings.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from compare_stage2_tcm_vs_western import (  # noqa: E402
    _load_external_pair,
    _load_oof_pair,
    bootstrap_delta_auc,
)
from export_paper2_si_combined_tables import (  # noqa: E402
    DEFAULT_OUT,
    S1_COMP,
    S2_COMP,
    export_delta_tables,
)

TOL = 1e-9


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Add absolute AUC bootstrap CIs to Table1/2")
    p.add_argument("--bootstrap-iter", type=int, default=2000)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    p.add_argument("--id-col-external", type=str, default="No")
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    return p.parse_args()


def _fmt_ci(est: float, lo: float, hi: float) -> str:
    if any(pd.isna(x) for x in (est, lo, hi)):
        return ""
    return f"{float(est):.3f} ({float(lo):.3f}; {float(hi):.3f})"


def _run_dirs(row: pd.Series) -> Tuple[Path, Path]:
    w = row.get("western_run_dir", row.get("baseline_run_dir"))
    t = row.get("tcm_run_dir")
    if pd.isna(w) or pd.isna(t):
        raise ValueError(f"Missing run dirs for model_index={row.get('model_index')}")
    return Path(str(w)), Path(str(t))


def enrich_comparison_csv(
    path: Path,
    *,
    n_boot: int,
    seed: int,
    id_col: str,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows: list[Dict[str, Any]] = []
    for _, row in df.iterrows():
        wdir, tdir = _run_dirs(row)
        dataset = str(row["dataset"])
        if dataset == "oof":
            pair = _load_oof_pair(wdir, tdir)
            y = pair["y_true"].to_numpy(dtype=int)
            sa = pair["y_prob_western"].to_numpy(dtype=float)
            sb = pair["y_prob_tcm"].to_numpy(dtype=float)
        elif dataset == "external":
            pair = _load_external_pair(wdir, tdir, id_col)
            y = pair["y_true"].to_numpy(dtype=int)
            sa = pair["prob_1_western"].to_numpy(dtype=float)
            sb = pair["prob_1_tcm"].to_numpy(dtype=float)
        else:
            raise ValueError(f"Unknown dataset={dataset!r} in {path}")

        boot = bootstrap_delta_auc(y, sa, sb, n_boot, seed)

        # Guard: ΔAUC CI must match locked comparison (same seed/params/resamples)
        for col, key in (
            ("bootstrap_delta_ci_low", "bootstrap_delta_ci_low"),
            ("bootstrap_delta_ci_high", "bootstrap_delta_ci_high"),
            ("delta_auc", "delta_auc"),
            ("auc_western", "auc_western"),
            ("auc_tcm", "auc_tcm"),
        ):
            if col not in row or pd.isna(row[col]):
                continue
            old = float(row[col])
            new = float(boot[key])
            if abs(old - new) > 1e-6:
                raise AssertionError(
                    f"Mismatch {path.name} model={row['model_index']} {dataset} "
                    f"{col}: old={old} new={new}"
                )

        out = row.to_dict()
        out.update(
            {
                "bootstrap_auc_western_ci_low": boot["bootstrap_auc_western_ci_low"],
                "bootstrap_auc_western_ci_high": boot["bootstrap_auc_western_ci_high"],
                "bootstrap_auc_tcm_ci_low": boot["bootstrap_auc_tcm_ci_low"],
                "bootstrap_auc_tcm_ci_high": boot["bootstrap_auc_tcm_ci_high"],
            }
        )
        rows.append(out)

    out_df = pd.DataFrame(rows)
    # Stable column order: insert absolute AUC CI next to AUC columns
    preferred = [
        "model_index",
        "model_name",
        "dataset",
        "n_samples",
        "n_pos_uc",
        "auc_western",
        "bootstrap_auc_western_ci_low",
        "bootstrap_auc_western_ci_high",
        "auc_tcm",
        "bootstrap_auc_tcm_ci_low",
        "bootstrap_auc_tcm_ci_high",
        "delta_auc",
        "bootstrap_delta_ci_low",
        "bootstrap_delta_ci_high",
        "bootstrap_p_two_sided",
        "auc_p_raw",
        "auc_p_fdr_bh",
        "bootstrap_n_valid",
        "delong_z",
        "delong_p_two_sided",
        "western_run_dir",
        "tcm_run_dir",
    ]
    cols = [c for c in preferred if c in out_df.columns] + [
        c for c in out_df.columns if c not in preferred
    ]
    out_df = out_df[cols]
    out_df.to_csv(path, index=False, encoding="utf-8-sig")
    return out_df


def _add_formatted(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Rename western → baseline for Paper2 main tables if needed
    ren = {}
    if "auc_western" in out.columns and "auc_baseline" not in out.columns:
        ren["auc_western"] = "auc_baseline"
    if "western_run_dir" in out.columns and "baseline_run_dir" not in out.columns:
        ren["western_run_dir"] = "baseline_run_dir"
    if "bootstrap_auc_western_ci_low" in out.columns:
        ren["bootstrap_auc_western_ci_low"] = "bootstrap_auc_baseline_ci_low"
        ren["bootstrap_auc_western_ci_high"] = "bootstrap_auc_baseline_ci_high"
    out = out.rename(columns=ren)

    out["auc_baseline_formatted"] = [
        _fmt_ci(a, lo, hi)
        for a, lo, hi in zip(
            out["auc_baseline"],
            out["bootstrap_auc_baseline_ci_low"],
            out["bootstrap_auc_baseline_ci_high"],
        )
    ]
    out["auc_tcm_formatted"] = [
        _fmt_ci(a, lo, hi)
        for a, lo, hi in zip(
            out["auc_tcm"],
            out["bootstrap_auc_tcm_ci_low"],
            out["bootstrap_auc_tcm_ci_high"],
        )
    ]
    out["delta_auc_formatted"] = [
        _fmt_ci(a, lo, hi)
        for a, lo, hi in zip(
            out["delta_auc"],
            out["bootstrap_delta_ci_low"],
            out["bootstrap_delta_ci_high"],
        )
    ]

    # Place formatted columns next to numeric blocks
    front = [
        "stage",
        "analysis_role",
        "model_index",
        "model_name",
        "dataset",
        "n_samples",
        "n_pos_uc",
        "auc_baseline",
        "bootstrap_auc_baseline_ci_low",
        "bootstrap_auc_baseline_ci_high",
        "auc_baseline_formatted",
        "auc_tcm",
        "bootstrap_auc_tcm_ci_low",
        "bootstrap_auc_tcm_ci_high",
        "auc_tcm_formatted",
        "delta_auc",
        "bootstrap_delta_ci_low",
        "bootstrap_delta_ci_high",
        "delta_auc_formatted",
    ]
    cols = [c for c in front if c in out.columns] + [c for c in out.columns if c not in front]
    return out[cols]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        S1_COMP / "stage2_oof_tcm_vs_western.csv",
        S1_COMP / "stage2_external_tcm_vs_western.csv",
        S1_COMP / "stage2_combined_tcm_vs_western.csv",
        S2_COMP / "stage2_oof_tcm_vs_western.csv",
        S2_COMP / "stage2_external_tcm_vs_western.csv",
        S2_COMP / "stage2_combined_tcm_vs_western.csv",
    ]
    for path in targets:
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"[enrich] {path}")
        enrich_comparison_csv(
            path,
            n_boot=args.bootstrap_iter,
            seed=args.bootstrap_seed,
            id_col=args.id_col_external,
        )

    oof_path, ext_path = export_delta_tables(out_dir)
    oof = _add_formatted(pd.read_csv(oof_path))
    ext = _add_formatted(pd.read_csv(ext_path))
    oof.to_csv(oof_path, index=False, encoding="utf-8-sig")
    ext.to_csv(ext_path, index=False, encoding="utf-8-sig")

    main_dir = out_dir / "main"
    main_dir.mkdir(parents=True, exist_ok=True)
    table1 = main_dir / "Table1.csv"
    table2 = main_dir / "Table2.csv"
    oof.to_csv(table1, index=False, encoding="utf-8-sig")
    ext.to_csv(table2, index=False, encoding="utf-8-sig")
    print(f"[OK] {table1}")
    print(f"[OK] {table2}")
    print(oof[["stage", "model_index", "auc_baseline_formatted", "auc_tcm_formatted", "delta_auc_formatted"]].head(4).to_string(index=False))


if __name__ == "__main__":
    main()
