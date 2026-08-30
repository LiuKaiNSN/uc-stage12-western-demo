# -*- coding: utf-8 -*-
"""
Paper2 SI baseline table (Stage1 & Stage2) from BaseLine_Stage12_TCM.csv.

Mirrors Figure/Stage12/generate_supplementary_table_s1.R:
  - Stage1 / Stage2 analysis sets are independent (Stage1 in {0,1}, Stage2 in {0,1})
  - Same centers and objective-feature variable blocks
  - Appends all TCM_* binary features (Yes/No n(%))

If BaseLine_Stage12_TCM.csv has no TCM_* columns, merges TCM_* from
Data/BaseLine_TCM.csv on key No (fallback until Stage12_TCM file is complete).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parent
DATA_DIR = PROJECT / "Data"
DEFAULT_BASELINE = DATA_DIR / "BaseLine_Stage12_TCM.csv"
FALLBACK_TCM = DATA_DIR / "BaseLine_TCM.csv"
DEFAULT_OUT = PROJECT / "Figure" / "比较" / "For submission only"

SUBROW = "  "

CENTER_MAP = {
    "DongFang": "Center A (development)",
    "DZM": "Center B (development)",
    "SanFu": "Center C (external validation)",
    "YanTai": "Center D (external validation)",
}
CENTER_ORDER = ["DongFang", "DZM", "SanFu", "YanTai"]

STAGE1_DIAG = ["CD", "UC", "IC", "CRC", "IE", "FDIBS"]
STAGE2_DIAG = ["UC", "CD", "IC", "CRC"]
DIAG_NAMES = {
    "CD": "Crohn's disease",
    "UC": "Ulcerative colitis",
    "IC": "Ischemic colitis",
    "CRC": "Colorectal cancer",
    "IE": "Infectious enteritis",
    "FDIBS": "Functional diarrhea/IBS",
}

BINARY_SYMPTOM = [
    "JiaoTi",
    "PaiBianKunn",
    "FuTong",
    "FuZhang",
    "EXinOuTu",
    "LiJiHouZhong",
    "XiaoShou",
    "Cha",
    "TouYunTouTong",
    "FaLi",
    "WeiHanFaRe",
]
ORDINAL_SYMPTOM = ["NianYe", "BianXue", "BianZhi", "PaiBianPinLv"]
PARENT_NAMES = {
    "JiaoTi": "Alternating bowel habit",
    "PaiBianKunn": "Defecation difficulty",
    "FuTong": "Abdominal pain",
    "FuZhang": "Abdominal distension",
    "EXinOuTu": "Nausea/vomiting",
    "LiJiHouZhong": "Tenesmus",
    "XiaoShou": "Weight loss",
    "Cha": "Anorexia",
    "TouYunTouTong": "Dizziness/headache",
    "FaLi": "Fatigue",
    "WeiHanFaRe": "Chills/fever",
    "NianYe": "Mucus visibility",
    "BianXue": "Hematochezia",
    "BianZhi": "Stool consistency",
    "PaiBianPinLv": "Bowel frequency",
}
ORDINAL_LEVELS = {
    "NianYe": {"1": "None", "2": "Scant mucus", "3": "Copious mucus"},
    "BianXue": {
        "1": "Negative",
        "2": "Positive",
        "3": "Visible blood (supplement form)",
        "4": "Visible blood (mixed)",
        "5": "Massive or tarry bloody stool",
    },
    "BianZhi": {"1": "Hard", "2": "Formed", "3": "Soft", "4": "Mushy", "5": "Watery"},
    "PaiBianPinLv": {
        "1": "<3 times/week",
        "2": "1-3 times/day",
        "3": "4-6 times/day",
        "4": "7-10 times/day",
        "5": ">10 times/day",
    },
}
LAB_COLS = [
    "WBC",
    "ZhongXingBaiFen",
    "LinBaBaiFen",
    "ShiSuanBaiFen",
    "ZhongXingJiShu",
    "LinBaJiShu",
    "RBC",
    "Hb",
    "HCT",
    "MCV",
    "MCH",
    "RDWCV",
    "PLT",
    "ALT",
    "ALP",
    "GGT",
    "TP",
    "Alb",
    "Glob",
    "AG",
    "K",
    "Fe",
    "PT",
    "FIB",
    "Ddimer",
    "PCT",
    "FC",
    "CRP",
    "ESR",
    "CEA",
    "CA199",
    "CA724",
    "Ferritin",
]
LAB_NAMES = {
    "WBC": "White blood cell count, ×10^9/L",
    "ZhongXingBaiFen": "Neutrophil percentage, %",
    "LinBaBaiFen": "Lymphocyte percentage, %",
    "ShiSuanBaiFen": "Eosinophil percentage, %",
    "ZhongXingJiShu": "Neutrophil count, ×10^9/L",
    "LinBaJiShu": "Lymphocyte count, ×10^9/L",
    "RBC": "Red blood cell count, ×10^12/L",
    "Hb": "Hemoglobin, g/L",
    "HCT": "Hematocrit, L/L",
    "MCV": "Mean corpuscular volume, fL",
    "MCH": "Mean corpuscular hemoglobin, pg",
    "RDWCV": "Red cell distribution width (coefficient of variation), %",
    "PLT": "Platelet count, ×10^9/L",
    "ALT": "Alanine aminotransferase, U/L",
    "ALP": "Alkaline phosphatase, U/L",
    "GGT": "Gamma-glutamyl transferase, U/L",
    "TP": "Total protein, g/L",
    "Alb": "Albumin, g/L",
    "Glob": "Globulin, g/L",
    "AG": "Albumin-to-globulin ratio",
    "K": "Potassium, mmol/L",
    "Fe": "Serum iron, μmol/L",
    "PT": "Prothrombin activity, %",
    "FIB": "Fibrinogen, g/mL",
    "Ddimer": "D-dimer, μg/mL",
    "PCT": "Procalcitonin, ng/mL",
    "FC": "Fecal calprotectin, μg/g",
    "CRP": "C-reactive protein, mg/L",
    "ESR": "Erythrocyte sedimentation rate, mm/1h",
    "CEA": "Carcinoembryonic antigen, ng/mL",
    "CA199": "Carbohydrate antigen 19-9, U/mL",
    "CA724": "Carbohydrate antigen 72-4, U/mL",
    "Ferritin": "Ferritin, ng/mL",
}
TCM_PARENT_NAMES = {
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
    "TCM_Tongue_red": "Red tongue",
    "TCM_Tongue_pale": "Pale tongue",
    "TCM_Tongue_dark_purple": "Dark-purple tongue",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper2 SI baseline table with TCM features")
    p.add_argument("--baseline-csv", type=str, default=str(DEFAULT_BASELINE))
    p.add_argument("--fallback-tcm-csv", type=str, default=str(FALLBACK_TCM))
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    return p.parse_args()


def _empty_vals() -> Dict[str, str]:
    return {CENTER_MAP[k]: "" for k in CENTER_ORDER}


def _fmt_n_pct(n: int, denom: int) -> str:
    if denom <= 0:
        return ""
    return f"{n} ({100.0 * n / denom:.1f})"


def _fmt_mean_sd(x: pd.Series) -> str:
    v = pd.to_numeric(x, errors="coerce").dropna()
    if v.empty:
        return ""
    return f"{v.mean():.2f} ({v.std(ddof=1):.2f})"


def _fmt_median_iqr(x: pd.Series) -> str:
    v = pd.to_numeric(x, errors="coerce").dropna()
    if v.empty:
        return ""
    q1, q3 = np.percentile(v, [25, 75])
    return f"{v.median():.2f} ({q1:.2f}–{q3:.2f})"


def _center_vals(d: pd.DataFrame, fun: Callable[[pd.DataFrame, int], str]) -> Dict[str, str]:
    out = _empty_vals()
    for src in CENTER_ORDER:
        dd = d[d["Source"] == src]
        out[CENTER_MAP[src]] = fun(dd, len(dd))
    return out


def _recode_disease_stage1(x: pd.Series) -> pd.Series:
    s = x.astype(str)
    s = s.replace({"Suspected (CD)": "CD", "Suspected (IC)": "IC"})
    return s


def _recode_disease_stage2(x: pd.Series) -> pd.Series:
    s = x.astype(str)
    s = s.str.replace(r"(?i)second time \(UC\)", "UC", regex=True)
    s = s.str.replace(r"(?i)second time \(IC\)", "IC", regex=True)
    s = s.str.replace(r"(?i)second time \(CD\)", "CD", regex=True)
    # keep only disease codes if still long labels containing code
    return s


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    if "PaiBianKunNan" in df.columns and "PaiBianKunn" not in df.columns:
        rename["PaiBianKunNan"] = "PaiBianKunn"
    if "NaCha" in df.columns and "Cha" not in df.columns:
        rename["NaCha"] = "Cha"
    if rename:
        df = df.rename(columns=rename)
    # drop unnamed empty columns
    drop = [c for c in df.columns if str(c).startswith("Unnamed") or str(c).strip() == ""]
    if drop:
        df = df.drop(columns=drop)
    return df


def load_baseline(path: Path, fallback_tcm: Path) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(path, na_values=["", "NA"])
    df = _normalize_cols(df)
    note = f"primary={path}"
    tcm_cols = [c for c in df.columns if str(c).startswith("TCM_")]
    if not tcm_cols:
        if not fallback_tcm.is_file():
            raise FileNotFoundError(
                f"{path} has no TCM_* columns and fallback missing: {fallback_tcm}"
            )
        fb = pd.read_csv(fallback_tcm, na_values=["", "NA"])
        fb = _normalize_cols(fb)
        tcm_cols = [c for c in fb.columns if str(c).startswith("TCM_")]
        if "No" not in df.columns or "No" not in fb.columns:
            raise ValueError("Cannot merge TCM features: missing No key")
        merge_cols = ["No"] + tcm_cols
        df = df.merge(fb[merge_cols], on="No", how="left", validate="one_to_one")
        note += f"; merged TCM_* from {fallback_tcm} on No"
        print(f"[WARN] {path.name} has no TCM_* columns; merged from {fallback_tcm.name}")
    df["Stage1_num"] = pd.to_numeric(df["Stage1"], errors="coerce")
    df["Stage2_num"] = pd.to_numeric(df["Stage2"], errors="coerce")
    return df, note


def _make_row(section: str, variable: str, values: Dict[str, str]) -> Dict[str, Any]:
    row = {"Section": section, "Variable": variable}
    row.update(values)
    return row


def _binary_block(
    d: pd.DataFrame,
    cols: Sequence[str],
    parent_names: Dict[str, str],
    section: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for col in cols:
        if col not in d.columns:
            continue
        parent = parent_names.get(col, col)
        rows.append(_make_row(section, f"{parent}, n (%)", _empty_vals()))
        rows.append(
            _make_row(
                section,
                f"{SUBROW}Yes",
                _center_vals(
                    d,
                    lambda dd, nn, c=col: _fmt_n_pct(
                        int((pd.to_numeric(dd[c], errors="coerce") == 1).sum()), nn
                    ),
                ),
            )
        )
        rows.append(
            _make_row(
                section,
                f"{SUBROW}No",
                _center_vals(
                    d,
                    lambda dd, nn, c=col: _fmt_n_pct(
                        int((pd.to_numeric(dd[c], errors="coerce") == 0).sum()), nn
                    ),
                ),
            )
        )
    return rows


def build_block(df: pd.DataFrame, section: str, diag_codes: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    rows.append(_make_row(section, section, _empty_vals()))

    rows.append(
        _make_row(section, "Sample size, n", _center_vals(df, lambda dd, nn: str(nn)))
    )
    rows.append(_make_row(section, "Age, years", _empty_vals()))
    rows.append(
        _make_row(section, f"{SUBROW}Mean (SD)", _center_vals(df, lambda dd, nn: _fmt_mean_sd(dd["Age"])))
    )
    rows.append(
        _make_row(
            section,
            f"{SUBROW}Median (IQR)",
            _center_vals(df, lambda dd, nn: _fmt_median_iqr(dd["Age"])),
        )
    )
    rows.append(_make_row(section, "Sex, n (%)", _empty_vals()))
    rows.append(
        _make_row(
            section,
            f"{SUBROW}Male",
            _center_vals(df, lambda dd, nn: _fmt_n_pct(int((dd["Sex"] == 1).sum()), nn)),
        )
    )
    rows.append(
        _make_row(
            section,
            f"{SUBROW}Female",
            _center_vals(df, lambda dd, nn: _fmt_n_pct(int((dd["Sex"] == 0).sum()), nn)),
        )
    )
    rows.append(_make_row(section, "Final diagnosis, n (%)", _empty_vals()))
    for code in diag_codes:
        rows.append(
            _make_row(
                section,
                f"{SUBROW}{DIAG_NAMES[code]}",
                _center_vals(
                    df,
                    lambda dd, nn, c=code: _fmt_n_pct(int((dd["Disease_std"] == c).sum()), nn),
                ),
            )
        )

    rows.append(_make_row(section, "Symptoms and signs", _empty_vals()))
    rows.extend(_binary_block(df, BINARY_SYMPTOM, PARENT_NAMES, section))
    for col in ORDINAL_SYMPTOM:
        if col not in df.columns:
            continue
        parent = PARENT_NAMES.get(col, col)
        rows.append(_make_row(section, f"{parent}, n (%)", _empty_vals()))
        for lv, label in ORDINAL_LEVELS[col].items():
            lv_num = float(lv)
            rows.append(
                _make_row(
                    section,
                    f"{SUBROW}{label}",
                    _center_vals(
                        df,
                        lambda dd, nn, c=col, v=lv_num: _fmt_n_pct(
                            int((pd.to_numeric(dd[c], errors="coerce") == v).sum()), nn
                        ),
                    ),
                )
            )

    rows.append(_make_row(section, "Laboratory tests", _empty_vals()))
    for col in LAB_COLS:
        if col not in df.columns:
            continue
        parent = LAB_NAMES.get(col, col)
        rows.append(_make_row(section, parent, _empty_vals()))
        rows.append(
            _make_row(
                section,
                f"{SUBROW}Mean (SD)",
                _center_vals(df, lambda dd, nn, c=col: _fmt_mean_sd(dd[c])),
            )
        )
        rows.append(
            _make_row(
                section,
                f"{SUBROW}Median (IQR)",
                _center_vals(df, lambda dd, nn, c=col: _fmt_median_iqr(dd[c])),
            )
        )

    tcm_cols = [c for c in df.columns if str(c).startswith("TCM_")]
    rows.append(_make_row(section, "Traditional Chinese Medicine features", _empty_vals()))
    rows.extend(_binary_block(df, tcm_cols, TCM_PARENT_NAMES, section))

    return pd.DataFrame(rows)


def write_footnotes(path: Path, data_note: str) -> None:
    text = f"""Supplementary Table. Baseline characteristics by center (Stage 1 & Stage 2; TCM paper)
Source data note: {data_note}

Centers
Center A (development cohort): Dongfang Hospital, Beijing University of Chinese Medicine.
Center B (development cohort): Dongzhimen Hospital, Beijing University of Chinese Medicine.
Center C (external validation cohort): Beijing University of Chinese Medicine Third Affiliated Hospital.
Center D (external validation cohort): Yantai Hospital of Traditional Chinese Medicine.

Analysis sets
Stage 1 analysis set: patients with Stage1 coded 0 or 1; suspected Crohn's disease and suspected ischemic colitis were recoded to Crohn's disease and ischemic colitis, respectively.
Stage 2 analysis set: patients with Stage2 coded 0 or 1; second-time supplemented cases were recoded to ulcerative colitis, ischemic colitis, or Crohn's disease as appropriate. Final diagnosis includes ulcerative colitis, Crohn's disease, ischemic colitis, and colorectal cancer.
Stage 1 and Stage 2 sample sizes are independent.

Traditional Chinese Medicine features
All TCM_* variables present in the baseline file are summarized (not restricted to final-model features). Binary TCM features are reported as Yes/No n (%).

Presentation
n (%): number (percentage); denominator = patients in that center and analysis set.
Continuous variables: mean (SD) and median (IQR).
No inferential comparisons were performed.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw, data_note = load_baseline(Path(args.baseline_csv), Path(args.fallback_tcm_csv))
    df_s1 = raw[raw["Stage1_num"].isin([0, 1])].copy()
    df_s1["Disease_std"] = _recode_disease_stage1(df_s1["Disease"])
    df_s2 = raw[raw["Stage2_num"].isin([0, 1])].copy()
    df_s2["Disease_std"] = _recode_disease_stage2(df_s2["Disease"])

    print(f"Stage 1 cohort: n = {len(df_s1)}")
    print(f"Stage 2 cohort: n = {len(df_s2)}")

    block_s1 = build_block(df_s1, "Stage 1 analysis set", STAGE1_DIAG)
    block_s2 = build_block(df_s2, "Stage 2 analysis set", STAGE2_DIAG)

    header_vals = _empty_vals()
    for src in CENTER_ORDER:
        cn = CENTER_MAP[src]
        header_vals[cn] = f"S1 n={int((df_s1['Source'] == src).sum())}; S2 n={int((df_s2['Source'] == src).sum())}"
    header = pd.DataFrame([_make_row("", "Variable", header_vals)])
    spacer = pd.DataFrame([_make_row("", "", _empty_vals())])
    final = pd.concat([header, block_s1, spacer, block_s2], ignore_index=True)

    out_csv = out_dir / "Supplementary_Table_S1_baseline_by_center_TCM.csv"
    out_notes = out_dir / "Supplementary_Table_S1_baseline_by_center_TCM_footnotes.txt"
    final.to_csv(out_csv, index=False, encoding="utf-8-sig")
    write_footnotes(out_notes, data_note)
    print(f"[OK] Baseline table -> {out_csv}")
    print(f"[OK] Footnotes -> {out_notes}")
    for src in CENTER_ORDER:
        print(
            f"  {src}: Stage1={int((df_s1['Source'] == src).sum())}, "
            f"Stage2={int((df_s2['Source'] == src).sum())}"
        )


if __name__ == "__main__":
    main()
