# -*- coding: utf-8 -*-
"""Batch launcher for TCM-integrated Stage1/2 publication figures."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parent
PYTHON = sys.executable

FIG_ROOT = PROJECT / "Figure" / "Stage12_TCM"
S1_RUN = PROJECT / "TCM" / "output" / "Stage1" / "6_run_20260708_052605"
S2_RUN = PROJECT / "TCM" / "output" / "Stage2" / "6_run_20260707_111716"
S1_DCA = PROJECT / "TCM" / "output" / "Stage1" / "dca_batch" / "6_run_20260708_052605" / "overall"
S2_DCA = PROJECT / "TCM" / "output" / "Stage2" / "dca_batch" / "6_run_20260707_111716" / "overall"
WORKFLOW_JSON = (
    PROJECT / "TCM" / "output" / "workflow_batch" / "tcm" / "06_6_best_cross" / "workflow_stage12.json"
)
S1_MODEL_LABEL = "RF (run 6)"
S2_MODEL_LABEL = "RF (run 6)"


def run(cmd: list[str], label: str) -> None:
    print(f"\n>>> {label}")
    subprocess.run(cmd, check=True, cwd=str(PROJECT))


def main() -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)

    run(
        [
            PYTHON,
            str(SCRIPT_DIR / "export_internal_cv_tcm_stage12.py"),
            "--s1-run-dir",
            str(S1_RUN),
            "--s2-run-dir",
            str(S2_RUN),
            "--out-dir",
            str(FIG_ROOT / "data"),
        ],
        "Internal CV tables (Stage1 + Stage2)",
    )

    run(
        [
            PYTHON,
            str(SCRIPT_DIR / "export_stage12_tcm_tables.py"),
            "--s1-run-dir",
            str(S1_RUN),
            "--s2-run-dir",
            str(S2_RUN),
            "--workflow-json",
            str(WORKFLOW_JSON),
            "--out-dir",
            str(FIG_ROOT / "data"),
            "--bootstrap-iter",
            "2000",
            "--seed",
            "42",
        ],
        "Table export (Stage1 + Stage2, overall + subgroups)",
    )

    run(
        [PYTHON, str(SCRIPT_DIR / "plot_figure2_tcm_model_comparison.py"), "--out-dir", str(FIG_ROOT / "figure2")],
        "Figure 2 supplement: 10-model comparison",
    )

    run(
        [
            PYTHON,
            str(SCRIPT_DIR / "plot_figure3_tcm_external_validation.py"),
            "--s1-run-dir",
            str(S1_RUN),
            "--s2-run-dir",
            str(S2_RUN),
            "--workflow-json",
            str(WORKFLOW_JSON),
            "--s1-dca-dir",
            str(S1_DCA),
            "--s2-dca-dir",
            str(S2_DCA),
            "--out-dir",
            str(FIG_ROOT / "figure3"),
            "--write-supplement",
        ],
        "Figure 3 (ROC+DCA panel + supplement)",
    )

    run(
        [
            PYTHON,
            str(SCRIPT_DIR / "interpret_binary_best_model_Stage12_tcm.py"),
            "--stage-label",
            "stage1",
            "--save-individual-pdp",
        ],
        "Figure 5 Stage1 (SHAP + PDP + individual)",
    )

    run(
        [
            PYTHON,
            str(SCRIPT_DIR / "interpret_binary_best_model_Stage12_tcm.py"),
            "--stage-label",
            "stage2",
            "--save-individual-pdp",
        ],
        "Figure 5 Stage2 (SHAP + PDP + individual)",
    )

    run(
        [
            PYTHON,
            str(SCRIPT_DIR / "plot_figure6_tcm_workflow.py"),
            "--workflow-json",
            str(WORKFLOW_JSON),
            "--out-dir",
            str(FIG_ROOT / "figure6"),
            "--s1-model",
            S1_MODEL_LABEL,
            "--s2-model",
            S2_MODEL_LABEL,
        ],
        "Figure 6 workflow panels",
    )

    print(f"\n[OK] TCM Stage12 figures under {FIG_ROOT}")
    print("  data/     internal_cv + table1_stage1|stage2")
    print("  figure2/  10-model comparison (supplement)")
    print("  figure3/  ROC+DCA + supplement")
    print("  figure5/  SHAP + PDP")
    print("  figure6/  workflow_w2_panels.jpg")


if __name__ == "__main__":
    main()
