# -*- coding: utf-8 -*-
"""
Paper2 SI combined tables (Baseline | TCM side-by-side) for Stage1+Stage2.

Script outputs (long names) under Figure/比较/For submission only — then map to
brief final names in main\\ / si\\ (see Paper2_figure_table_titles_and_captions.txt):
  - SI_Table_hyperparameters_wm_tcm_stage12.csv          → si\\TableS3.csv
  - SI_Table_internal_cv_wm_tcm_stage12.csv              → si\\TableS4.csv
  - SI_Table_external_overall_wm_tcm_stage12.csv         → si\\TableS5.csv
  - SI_Table_final_features_wm_tcm_stage12.csv           → si\\TableS6.csv
  - SI_Table_external_disease_slice_auc_wm_tcm_stage12.csv → si\\TableS7.csv
  - Table2_delta_auc_oof_stage12.csv                     → main\\Table1.csv
      (#1–#10; confirmatory #1–#6 + exploratory DL #7–#10, no FDR)
  - Table3_delta_auc_external_stage12.csv                → main\\Table2.csv
      (same scope as Table1; Figure 2/3 forests stay #1–#6 only)
  - Table4_*_subgroup_* (Stage1+Stage2)                  → main\\Table3.csv
  - (codebook) build_paper2_table_s7_codebook.py         → si\\TableS2.csv

Note: absolute-AUC companion CSVs from delta-forest are NOT separate SI tables;
      auc_baseline / auc_tcm already live in main Table1 / Table2.
      Table S2 is the variable coding dictionary (objective features + TCM); do not overwrite
      it with hyperparameters when remapping combined-table outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parent
DEFAULT_OUT = PROJECT / "Figure" / "比较" / "For submission only"

WM_S1_RUNS = [
    ("1_run_20260706_222656", "XGBoost"),
    ("2_run_20260707_134652", "LR"),
    ("3_run_20260707_142952", "LightGBM"),
    ("4_run_20260707_151303", "CatBoost"),
    ("5_run_20260707_155710", "SVM"),
    ("6_run_20260707_164844", "RF"),
    ("7_run_mlp_20260707_174603", "MLP"),
    ("8_run_dcnv2_20260707_182503", "DCNV2"),
    ("9_run_fttransformer_20260707_191306", "FT-Transformer"),
    ("10_run_tabtransformer_20260707_201217", "TabTransformer"),
]
WM_S2_RUNS = [
    ("1_run_20260706_204357", "XGBoost"),
    ("2_run_20260707_012220", "LR"),
    ("3_run_20260707_023818", "LightGBM"),
    ("4_run_20260707_030540", "CatBoost"),
    ("5_run_20260707_033332", "SVM"),
    ("6_run_20260707_040110", "RF"),
    ("7_run_mlp_20260707_042856", "MLP"),
    ("8_run_dcnv2_20260707_045208", "DCNV2"),
    ("9_run_fttransformer_20260707_051518", "FT-Transformer"),
    ("10_run_tabtransformer_20260707_053845", "TabTransformer"),
]
TCM_S1_RUNS = [
    ("1_run_20260706_231752", "XGBoost"),
    ("2_run_20260708_021008", "LR"),
    ("3_run_20260708_025841", "LightGBM"),
    ("4_run_20260708_034754", "CatBoost"),
    ("5_run_20260708_043707", "SVM"),
    ("6_run_20260708_052605", "RF"),
    ("7_run_mlp_20260708_061524", "MLP"),
    ("8_run_dcnv2_20260708_070028", "DCNV2"),
    ("9_run_fttransformer_20260708_074616", "FT-Transformer"),
    ("10_run_tabtransformer_20260708_083216", "TabTransformer"),
]
TCM_S2_RUNS = [
    ("1_run_20260706_211837", "XGBoost"),
    ("2_run_20260707_094816", "LR"),
    ("3_run_20260707_101730", "LightGBM"),
    ("4_run_20260711_190700", "CatBoost"),
    ("5_run_20260707_104803", "SVM"),
    ("6_run_20260707_111716", "RF"),
    ("7_run_mlp_20260707_114635", "MLP"),
    ("8_run_dcnv2_20260707_121227", "DCNV2"),
    ("9_run_fttransformer_20260707_123841", "FT-Transformer"),
    ("10_run_tabtransformer_20260707_130521", "TabTransformer"),
]

S1_COMP = PROJECT / "Stage1_TCM_vs_Western_comparison" / "comparison_20260713_190255"
S2_COMP = PROJECT / "Stage2_TCM_vs_Western_comparison" / "comparison_20260713_190605"

# Locked best-model runs for final feature lists (Paper2 strategy arms)
WM_BEST = {
    "Stage 1": PROJECT / "outputs" / "Stage1" / "3_run_20260707_142952" / "final_features_used.txt",
    "Stage 2": PROJECT / "outputs" / "Stage2" / "4_run_20260707_030540" / "final_features_used.txt",
}
TCM_BEST = {
    "Stage 1": PROJECT / "TCM" / "output" / "Stage1" / "6_run_20260708_052605" / "final_features_used.txt",
    "Stage 2": PROJECT / "TCM" / "output" / "Stage2" / "6_run_20260707_111716" / "final_features_used.txt",
}
WM_BEST_MODEL = {"Stage 1": "LightGBM (Baseline Model)", "Stage 2": "CatBoost (Baseline Model)"}
TCM_BEST_MODEL = {"Stage 1": "RF (TCM-integrated Model)", "Stage 2": "RF (TCM-integrated Model)"}

# DCA-aligned disease slices (descriptive external AUC only; no bootstrap / tests)
# Stage1: disease-positive slices + IE/FDIBS (class-0) vs inflammatory-neoplastic pool
STAGE1_DISEASE_SLICES = (
    "uc_vs_ie_ibs_pool",
    "cd_vs_ie_ibs_pool",
    "ic_vs_ie_ibs_pool",
    "crc_vs_ie_ibs_pool",
    "fdibs_vs_inflam_pool",
    "ie_vs_inflam_pool",
)
# Stage2: UC vs each disease + Montreal extent E1/E2/E3 vs non-UC pool
STAGE2_DISEASE_SLICES = (
    "uc_vs_cd",
    "uc_vs_ic",
    "uc_vs_crc",
    "na_pool_vs_uc_e1",
    "na_pool_vs_uc_e2",
    "na_pool_vs_uc_e3",
)
SLICE_LABELS = {
    "uc_vs_ie_ibs_pool": "UC vs (IE & FDIBS)",
    "cd_vs_ie_ibs_pool": "CD vs (IE & FDIBS)",
    "ic_vs_ie_ibs_pool": "IC vs (IE & FDIBS)",
    "crc_vs_ie_ibs_pool": "CRC vs (IE & FDIBS)",
    "fdibs_vs_inflam_pool": "FDIBS vs (UC & CD & IC & CRC)",
    "ie_vs_inflam_pool": "IE vs (UC & CD & IC & CRC)",
    "uc_vs_cd": "UC vs CD",
    "uc_vs_ic": "UC vs IC",
    "uc_vs_crc": "UC vs CRC",
    "na_pool_vs_uc_e1": "(CD & CRC & IC) vs UC (E1)",
    "na_pool_vs_uc_e2": "(CD & CRC & IC) vs UC (E2)",
    "na_pool_vs_uc_e3": "(CD & CRC & IC) vs UC (E3)",
}
DISEASE_SLICE_SOURCES = {
    "stage1": {
        "western": PROJECT
        / "outputs"
        / "Stage1"
        / "3_run_20260707_142952"
        / "external_validation_binary"
        / "type_slices"
        / "type_slices_summary.csv",
        "tcm": PROJECT
        / "TCM"
        / "output"
        / "Stage1"
        / "6_run_20260708_052605"
        / "external_validation_binary"
        / "type_slices"
        / "type_slices_summary.csv",
        "id_col": "stratum_id",
        "slices": STAGE1_DISEASE_SLICES,
        "model_w": "LightGBM (Baseline Model)",
        "model_t": "RF (TCM-integrated Model)",
    },
    "stage2": {
        "western": PROJECT
        / "outputs"
        / "Stage2"
        / "4_run_20260707_030540"
        / "external_validation_binary"
        / "type_montreal_slices"
        / "type_montreal_slices_summary.csv",
        "tcm": PROJECT
        / "TCM"
        / "output"
        / "Stage2"
        / "6_run_20260707_111716"
        / "external_validation_binary"
        / "type_montreal_slices"
        / "type_montreal_slices_summary.csv",
        "id_col": "slice_id",
        "slices": STAGE2_DISEASE_SLICES,
        "model_w": "CatBoost (Baseline Model)",
        "model_t": "RF (TCM-integrated Model)",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper2 SI combined WM|TCM tables")
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    return p.parse_args()


def _fmt_val(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int) and not isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if float(v).is_integer():
            return str(int(v))
        return f"{float(v):.3f}"
    return str(v)


def _params_to_str(d: dict) -> str:
    return "; ".join(f"{k} = {_fmt_val(v)}" for k, v in d.items())


def _load_hp(stage_dir: Path, runs: List[Tuple[str, str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for folder, name in runs:
        meta_path = stage_dir / folder / "final_model_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        out[name] = _params_to_str(meta.get("final_params") or {})
    return out


def export_hyperparams(out_dir: Path) -> Path:
    wm_s1 = _load_hp(PROJECT / "outputs" / "Stage1", WM_S1_RUNS)
    wm_s2 = _load_hp(PROJECT / "outputs" / "Stage2", WM_S2_RUNS)
    tcm_s1 = _load_hp(PROJECT / "TCM" / "output" / "Stage1", TCM_S1_RUNS)
    tcm_s2 = _load_hp(PROJECT / "TCM" / "output" / "Stage2", TCM_S2_RUNS)
    models = [n for _, n in WM_S1_RUNS]
    rows = []
    for stage, wm, tcm in [
        ("Stage 1", wm_s1, tcm_s1),
        ("Stage 2", wm_s2, tcm_s2),
    ]:
        rows.append({"Stage": stage, "Model": "", "Baseline_hyperparameters": "", "TCM_hyperparameters": ""})
        for m in models:
            rows.append(
                {
                    "Stage": stage,
                    "Model": m,
                    "Baseline_hyperparameters": wm.get(m, ""),
                    "TCM_hyperparameters": tcm.get(m, ""),
                }
            )
    out = out_dir / "SI_Table_hyperparameters_wm_tcm_stage12.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    return out


def _read_mean_sd(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # expect model + auc mean/sd style; keep flexible
    return df


def export_internal_cv(out_dir: Path) -> Path:
    pairs = [
        (
            "Stage1",
            PROJECT / "Figure" / "Stage12" / "data" / "internal_cv_stage1_fold_mean_sd.csv",
            PROJECT / "Figure" / "Stage12_TCM" / "data" / "internal_cv_stage1_fold_mean_sd.csv",
        ),
        (
            "Stage2",
            PROJECT / "Figure" / "Stage12" / "data" / "internal_cv_stage2_fold_mean_sd.csv",
            PROJECT / "Figure" / "Stage12_TCM" / "data" / "internal_cv_stage2_fold_mean_sd.csv",
        ),
    ]
    frames = []
    for stage, wm_path, tcm_path in pairs:
        wm = _read_mean_sd(wm_path).copy()
        tcm = _read_mean_sd(tcm_path).copy()
        # standardize merge key
        key = "model" if "model" in wm.columns else ("model_name" if "model_name" in wm.columns else wm.columns[0])
        wm = wm.rename(columns={c: f"Baseline_{c}" for c in wm.columns if c != key})
        tcm = tcm.rename(columns={c: f"TCM_{c}" for c in tcm.columns if c != key})
        merged = wm.merge(tcm, on=key, how="outer")
        merged.insert(0, "Stage", stage)
        frames.append(merged)
    out = out_dir / "SI_Table_internal_cv_wm_tcm_stage12.csv"
    pd.concat(frames, ignore_index=True).to_csv(out, index=False, encoding="utf-8-sig")
    return out


def export_external_overall(out_dir: Path) -> Path:
    pairs = [
        (
            "Stage1",
            PROJECT / "Figure" / "Stage12" / "data" / "table1_stage1_external_validation_long.csv",
            PROJECT / "Figure" / "Stage12_TCM" / "data" / "table1_stage1_external_validation_long.csv",
        ),
        (
            "Stage2",
            PROJECT / "Figure" / "Stage12" / "data" / "table1_stage2_external_validation_long.csv",
            PROJECT / "Figure" / "Stage12_TCM" / "data" / "table1_stage2_external_validation_long.csv",
        ),
    ]
    frames = []
    for stage, wm_path, tcm_path in pairs:
        wm = pd.read_csv(wm_path)
        tcm = pd.read_csv(tcm_path)

        def _overall(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            if "section" in out.columns:
                out = out[out["section"].astype(str).str.lower() == "overall"]
            if "stratum_id" in out.columns:
                out = out[out["stratum_id"].astype(str).str.lower() == "overall"]
            keep = [
                c
                for c in (
                    "threshold_section",
                    "threshold",
                    "n_samples",
                    "metric",
                    "value",
                    "ci_low",
                    "ci_high",
                    "formatted",
                )
                if c in out.columns
            ]
            return out[keep].copy()

        wm = _overall(wm)
        tcm = _overall(tcm)
        # Keep default probability threshold only (drop workflow / DCA operating points)
        if "threshold_section" in wm.columns:
            wm = wm[wm["threshold_section"].astype(str).str.lower().isin(["default", "threshold_0.5", "0.5", "prob_0.5"]) | wm["threshold_section"].isna()]
            if wm.empty:
                wm = _overall(pd.read_csv(wm_path))
                if "threshold_section" in wm.columns:
                    # fallback: first threshold block
                    first = wm["threshold_section"].dropna().astype(str).iloc[0]
                    wm = wm[wm["threshold_section"].astype(str) == first]
        if "threshold_section" in tcm.columns:
            tcm = tcm[tcm["threshold_section"].astype(str).str.lower().isin(["default", "threshold_0.5", "0.5", "prob_0.5"]) | tcm["threshold_section"].isna()]
            if tcm.empty:
                tcm = _overall(pd.read_csv(tcm_path))
                if "threshold_section" in tcm.columns:
                    first = tcm["threshold_section"].dropna().astype(str).iloc[0]
                    tcm = tcm[tcm["threshold_section"].astype(str) == first]
        # Compact SI Table S5 shape: Stage, n_samples, metric, Baseline_formatted, TCM_formatted
        def _compact(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
            out = df.copy()
            cols = {}
            if "n_samples" in out.columns:
                cols["n_samples"] = "n_samples"
            if "formatted" in out.columns:
                cols["formatted"] = f"{prefix}_formatted"
            elif "value" in out.columns:
                cols["value"] = f"{prefix}_formatted"
            keep = ["metric"] + [c for c in ("n_samples", "formatted", "value") if c in out.columns]
            out = out[keep].rename(columns={k: v for k, v in cols.items() if k in out.columns})
            return out

        wm_c = _compact(wm, "Baseline")
        tcm_c = _compact(tcm, "TCM")
        merged = wm_c.merge(tcm_c, on="metric", how="outer", suffixes=("", "_dup"))
        if "n_samples_x" in merged.columns:
            merged["n_samples"] = merged["n_samples_x"].fillna(merged.get("n_samples_y"))
            merged = merged.drop(columns=[c for c in merged.columns if c.startswith("n_samples_")])
        merged.insert(0, "Stage", stage)
        frames.append(merged)
    out = out_dir / "SI_Table_external_overall_wm_tcm_stage12.csv"
    pd.concat(frames, ignore_index=True).to_csv(out, index=False, encoding="utf-8-sig")
    return out


def _read_feature_list(path: Path) -> List[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def export_final_features(out_dir: Path) -> Path:
    """Compact Table S6: Stage | Algorithm | Included_features (SHAP English labels)."""
    from _interpret_feature_labels import STAGE1_LABELS, STAGE2_LABELS, _ALL_FEATURE_LABELS

    tcm_en = {
        "TCM_Tongue_red": "Red tongue",
        "TCM_Tongue_pale": "Pale tongue",
        "TCM_Tongue_dark_purple": "Dark-purple tongue",
        "TCM_Tooth_mark": "Tongue tooth mark",
        "TCM_Fissure": "Tongue fissure",
        "TCM_Coat_yellow": "Yellow tongue coating",
        "TCM_Coat_white": "White tongue coating",
        "TCM_Coat_greasy_thick": "Greasy/thick coating",
        "TCM_Coat_dry_scant": "Dry/scant coating",
        "TCM_Pulse_string": "String-like pulse",
        "TCM_Pulse_slip_rapid": "Slippery-rapid pulse",
        "TCM_Pulse_thin_sink": "Thin-sunken pulse",
        "TCM_Pulse_ru_hua": "Soggy pulse",
        "TCM_Syn_damp_heat": "Syndrome: damp-heat",
        "TCM_Syn_qi_blood_def": "Syndrome: qi-blood deficiency",
        "TCM_Syn_qi_stagnation_blood_stasis": "Syndrome: qi stagnation / blood stasis",
        "TCM_Syn_spleen_kidney_yang": "Syndrome: spleen-kidney yang deficiency",
        "TCM_Syn_yin_def_fire": "Syndrome: yin deficiency with fire",
        "TCM_Syn_toxic_heat": "Syndrome: toxic heat",
        "TCM_Nature_shi": "Nature: excess (shi)",
        "TCM_Nature_xu": "Nature: deficiency (xu)",
        "TCM_Nature_mix": "Nature: mixed",
    }

    def _en(code: str, stage: str) -> str:
        if code.startswith("TCM_"):
            return tcm_en.get(code, code)
        primary = STAGE1_LABELS if stage == "Stage 1" else STAGE2_LABELS
        return primary.get(code, _ALL_FEATURE_LABELS.get(code, code))

    rows: List[dict] = []
    for stage, algo_b, algo_t in (
        ("Stage 1", WM_BEST_MODEL["Stage 1"], TCM_BEST_MODEL["Stage 1"]),
        ("Stage 2", WM_BEST_MODEL["Stage 2"], TCM_BEST_MODEL["Stage 2"]),
    ):
        wm = _read_feature_list(WM_BEST[stage])
        tcm = _read_feature_list(TCM_BEST[stage])
        rows.append(
            {
                "Stage": stage,
                "Algorithm": algo_b,
                "Included_features": "; ".join(_en(c, stage) for c in wm),
            }
        )
        rows.append(
            {
                "Stage": stage,
                "Algorithm": algo_t,
                "Included_features": "; ".join(_en(c, stage) for c in tcm),
            }
        )
    out = out_dir / "SI_Table_final_features_wm_tcm_stage12.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    return out


def export_subgroup_side_by_side(out_dir: Path) -> Path:
    """Stage1+Stage2 center/age external AUC — best models; descriptive; key join."""

    def _merge(wm: pd.DataFrame, tcm: pd.DataFrame, stage: str) -> pd.DataFrame:
        key_cols = [c for c in ("section", "stratum_id") if c in wm.columns and c in tcm.columns]
        if not key_cols:
            key_cols = [c for c in ("subgroup", "group", "Source", "Agerange", "level") if c in wm.columns]
        if not key_cols:
            key_cols = [wm.columns[0]]
        value_cols = [
            c
            for c in ("stratum_label", "n", "auc", "auc_ci_low", "auc_ci_high", "auc_formatted")
            if c in wm.columns
        ]
        tcm_vals = [c for c in ("n", "auc", "auc_ci_low", "auc_ci_high", "auc_formatted") if c in tcm.columns]
        wm2 = wm[key_cols + value_cols].copy()
        tcm2 = tcm[key_cols + tcm_vals].rename(columns={c: f"TCM_{c}" for c in tcm_vals})
        wm2 = wm2.rename(columns={c: f"Baseline_{c}" for c in value_cols if c != "stratum_label"})
        merged = wm2.merge(tcm2, on=key_cols, how="outer")
        merged.insert(0, "stage", stage)
        return merged

    frames = [
        _merge(
            pd.read_csv(PROJECT / "Figure" / "Stage12" / "data" / "table1_stage1_subgroup_auc.csv"),
            pd.read_csv(PROJECT / "Figure" / "Stage12_TCM" / "data" / "table1_stage1_subgroup_auc.csv"),
            "stage1",
        ),
        _merge(
            pd.read_csv(PROJECT / "Figure" / "Stage12" / "data" / "table1_stage2_subgroup_auc.csv"),
            pd.read_csv(PROJECT / "Figure" / "Stage12_TCM" / "data" / "table1_stage2_subgroup_auc.csv"),
            "stage2",
        ),
    ]
    out = out_dir / "Table4_stage12_subgroup_auc_wm_tcm_descriptive.csv"
    pd.concat(frames, ignore_index=True).to_csv(out, index=False, encoding="utf-8-sig")
    return out


def _load_slice_auc(path: Path, id_col: str, slices: Tuple[str, ...]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if id_col not in df.columns and "stratum_id" in df.columns:
        id_col = "stratum_id"
    if id_col not in df.columns and "slice_id" in df.columns:
        id_col = "slice_id"
    df = df[df[id_col].isin(slices)].copy()
    df["slice_id"] = df[id_col].astype(str)
    df["n"] = pd.to_numeric(df.get("n_samples", df.get("n")), errors="coerce")
    df["auc"] = pd.to_numeric(df["auc"], errors="coerce")
    df = df.dropna(subset=["auc"])
    return df[["slice_id", "n", "auc"]]


def export_disease_slice_auc(out_dir: Path) -> Path:
    """
    External disease-slice AUC for locked best models (Western | TCM).
    Descriptive only: point AUC + optional delta; no bootstrap / hypothesis tests.
    Slice IDs aligned with DCA disease panels.
    """
    rows = []
    for stage, cfg in DISEASE_SLICE_SOURCES.items():
        wm = _load_slice_auc(cfg["western"], cfg["id_col"], cfg["slices"])
        tcm = _load_slice_auc(cfg["tcm"], cfg["id_col"], cfg["slices"])
        merged = wm.merge(tcm, on="slice_id", how="outer", suffixes=("_baseline", "_tcm"))
        # restore order
        order = {s: i for i, s in enumerate(cfg["slices"])}
        merged["_ord"] = merged["slice_id"].map(order)
        merged = merged.sort_values("_ord")
        for _, r in merged.iterrows():
            w_auc = r.get("auc_baseline")
            t_auc = r.get("auc_tcm")
            delta = (
                float(t_auc) - float(w_auc)
                if pd.notna(w_auc) and pd.notna(t_auc)
                else float("nan")
            )
            rows.append(
                {
                    "stage": stage,
                    "slice_id": r["slice_id"],
                    "slice_label": SLICE_LABELS.get(str(r["slice_id"]), str(r["slice_id"])),
                    "Baseline_best_model": cfg["model_w"],
                    "Baseline_n": int(r["n_baseline"]) if pd.notna(r.get("n_baseline")) else "",
                    "Baseline_auc": None if pd.isna(w_auc) else round(float(w_auc), 6),
                    "Baseline_auc_formatted": "" if pd.isna(w_auc) else f"{float(w_auc):.3f}",
                    "TCM_best_model": cfg["model_t"],
                    "TCM_n": int(r["n_tcm"]) if pd.notna(r.get("n_tcm")) else "",
                    "TCM_auc": None if pd.isna(t_auc) else round(float(t_auc), 6),
                    "TCM_auc_formatted": "" if pd.isna(t_auc) else f"{float(t_auc):.3f}",
                    "delta_auc_TCM_minus_Baseline": None if pd.isna(delta) else round(delta, 6),
                    "delta_auc_formatted": "" if pd.isna(delta) else f"{delta:+.3f}",
                    "note": "Descriptive external AUC by disease slice; no bootstrap CI; no hypothesis test",
                }
            )
    out = out_dir / "SI_Table_external_disease_slice_auc_wm_tcm_stage12.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    return out


def export_delta_tables(out_dir: Path) -> Tuple[Path, Path]:
    """
    Main Table1 (OOF) / Table2 (external): algorithms #1–#10.
      #1–#6 confirmatory (BH-FDR applied in source compare scripts);
      #7–#10 exploratory deep-learning models (no FDR; auc_p_fdr_bh blank).
    Figure 2 / Figure 3 forests remain confirmatory #1–#6 only.
    """
    frames_oof = []
    frames_ext = []
    for stage, comp in [("stage1", S1_COMP), ("stage2", S2_COMP)]:
        oof = pd.read_csv(comp / "stage2_oof_tcm_vs_western.csv")
        ext = pd.read_csv(comp / "stage2_external_tcm_vs_western.csv")
        oof = oof[oof["model_index"].isin(range(1, 11))].copy()
        ext = ext[ext["model_index"].isin(range(1, 11))].copy()
        for df in (oof, ext):
            df.insert(0, "stage", stage)
            df.insert(
                1,
                "analysis_role",
                df["model_index"].map(
                    lambda i: "confirmatory" if int(i) <= 6 else "exploratory"
                ),
            )
            if "auc_western" in df.columns and "auc_baseline" not in df.columns:
                df.rename(columns={"auc_western": "auc_baseline"}, inplace=True)
            if "western_run_dir" in df.columns and "baseline_run_dir" not in df.columns:
                df.rename(columns={"western_run_dir": "baseline_run_dir"}, inplace=True)
            if (
                "bootstrap_auc_western_ci_low" in df.columns
                and "bootstrap_auc_baseline_ci_low" not in df.columns
            ):
                df.rename(
                    columns={
                        "bootstrap_auc_western_ci_low": "bootstrap_auc_baseline_ci_low",
                        "bootstrap_auc_western_ci_high": "bootstrap_auc_baseline_ci_high",
                    },
                    inplace=True,
                )
        frames_oof.append(oof)
        frames_ext.append(ext)
    out_oof = out_dir / "Table2_delta_auc_oof_stage12.csv"
    out_ext = out_dir / "Table3_delta_auc_external_stage12.csv"
    pd.concat(frames_oof, ignore_index=True).to_csv(out_oof, index=False, encoding="utf-8-sig")
    pd.concat(frames_ext, ignore_index=True).to_csv(out_ext, index=False, encoding="utf-8-sig")
    return out_oof, out_ext


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        export_hyperparams(out_dir),
        export_internal_cv(out_dir),
        export_external_overall(out_dir),
        export_subgroup_side_by_side(out_dir),
        export_final_features(out_dir),
        export_disease_slice_auc(out_dir),
    ]
    oof, ext = export_delta_tables(out_dir)
    paths.extend([oof, ext])
    for p in paths:
        print(f"[OK] {p}")


if __name__ == "__main__":
    main()
