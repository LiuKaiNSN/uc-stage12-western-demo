# -*- coding: utf-8 -*-
"""Build Paper2 Table S2 codebook (objective features + TCM), English headers for SCI."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT = Path(r"F:/KeTi/Project")
OUT = None
for p in (PROJECT / "Figure").iterdir():
    if (p / "For submission only" / "si").is_dir():
        OUT = p / "For submission only" / "si" / "TableS2.csv"
        break
if OUT is None:
    raise SystemExit("si out dir not found")

NA = "Not Applicable"

# Chinese labels retained as a bilingual reference column
ZH = {
    "Age": "年龄",
    "Sex": "性别",
    "JiaoTi": "交替（排便习惯）",
    "PaiBianKunn": "排便困难",
    "FuTong": "腹痛",
    "FuZhang": "腹胀",
    "EXinOuTu": "恶心呕吐",
    "LiJiHouZhong": "里急后重",
    "XiaoShou": "消瘦",
    "Cha": "纳差（食欲减退）",
    "TouYunTouTong": "头晕头痛",
    "FaLi": "乏力",
    "WeiHanFaRe": "畏寒发热",
    "NianYe": "粘液可见度",
    "BianXue": "便血程度",
    "BianZhi": "便质",
    "PaiBianPinLv": "排便频率",
    "WBC": "白细胞计数",
    "ZhongXingBaiFen": "中性粒细胞百分比",
    "LinBaBaiFen": "淋巴细胞百分比",
    "ShiSuanBaiFen": "嗜酸性粒细胞百分比",
    "ZhongXingJiShu": "中性粒细胞计数",
    "LinBaJiShu": "淋巴细胞绝对值",
    "RBC": "红细胞计数",
    "Hb": "血红蛋白",
    "HCT": "红细胞比积",
    "MCV": "平均红细胞体积",
    "MCH": "平均红细胞血红蛋白含量",
    "RDWCV": "红细胞分布宽度变异系数",
    "PLT": "血小板计数",
    "ALT": "丙氨酸氨基转移酶",
    "ALP": "碱性磷酸酶",
    "GGT": "γ-谷氨酰转肽酶",
    "TP": "总蛋白",
    "Alb": "白蛋白",
    "Glob": "球蛋白",
    "AG": "白球比",
    "K": "钾",
    "Fe": "血清铁",
    "PT": "凝血酶原活动度",
    "FIB": "纤维蛋白原",
    "Ddimer": "D-二聚体",
    "PCT": "降钙素原",
    "FC": "粪钙卫蛋白",
    "CRP": "C-反应蛋白",
    "ESR": "红细胞沉降率",
    "CEA": "癌胚抗原",
    "CA199": "糖类抗原 CA19-9",
    "CA724": "糖类抗原 CA72-4",
    "Ferritin": "铁蛋白",
    "TCM_Tongue_red": "舌红（偏红/绛）",
    "TCM_Tongue_pale": "舌淡（淡红/淡胖等）",
    "TCM_Tongue_dark_purple": "舌暗紫/瘀斑瘀点",
    "TCM_Tooth_mark": "齿痕舌",
    "TCM_Fissure": "裂纹舌",
    "TCM_Coat_yellow": "黄苔",
    "TCM_Coat_white": "白苔",
    "TCM_Coat_greasy_thick": "腻苔/厚苔",
    "TCM_Coat_dry_scant": "少苔/燥苔/剥脱",
    "TCM_Pulse_string": "弦脉（含弦细/弦滑等）",
    "TCM_Pulse_slip_rapid": "滑/数脉类",
    "TCM_Pulse_thin_sink": "细/沉/弱脉类",
    "TCM_Pulse_ru_hua": "濡/缓脉类",
    "TCM_Syn_damp_heat": "湿热证",
    "TCM_Syn_qi_blood_def": "气血两虚/脾胃气虚类",
    "TCM_Syn_qi_stagnation_blood_stasis": "气滞血瘀/肝郁类",
    "TCM_Syn_spleen_kidney_yang": "脾肾阳虚/虚寒类",
    "TCM_Syn_yin_def_fire": "阴虚火旺/阴虚内热类",
    "TCM_Syn_toxic_heat": "热毒/瘀毒类",
    "TCM_Nature_shi": "病性属实",
    "TCM_Nature_xu": "病性属虚",
    "TCM_Nature_mix": "虚实夹杂/本虚标实",
}

# English names + units aligned with Table S1 / export_paper2_baseline_stage12_tcm.py
EN = {
    "Age": "Age, years",
    "Sex": "Sex",
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

CONT = "Continuous; use raw measured values; no categorical coding"
BIN_YN = "1 = yes; 0 = no"
BIN_SEX = "1 = male; 0 = female"
BIN_TCM = "1 = yes; 0 = no"

ORD = {
    "NianYe": "1 = none; 2 = scant mucus; 3 = copious mucus",
    "BianXue": (
        "1 = negative; 2 = positive (not visible to the naked eye); "
        "3 = visible blood (supplement form); "
        "4 = visible blood (mixed); "
        "5 = massive or tarry bloody stool"
    ),
    "BianZhi": "1 = hard; 2 = formed; 3 = soft; 4 = mushy; 5 = watery",
    "PaiBianPinLv": (
        "1 = <3 times/week; 2 = 1-3 times/day; "
        "3 = 4-6 times/day; 4 = 7-10 times/day; "
        "5 = >10 times/day"
    ),
}

BINARY = [
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
ORDINAL = ["NianYe", "BianXue", "BianZhi", "PaiBianPinLv"]
LABS = [
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

# TCM: (var, trigger_ZH, trigger_EN, note_EN)
TCM_ROWS = [
    (
        "TCM_Tongue_red",
        "舌红、舌质红、舌红绛、舌绛等",
        "red tongue; crimson tongue; deep-red tongue tip; etc.",
        "Tongue inspection; rule-based extraction from four-diagnosis text",
    ),
    (
        "TCM_Tongue_pale",
        "舌淡、舌淡红、舌淡胖、舌体胖大等",
        "pale tongue; pale-red tongue; swollen pale tongue; etc.",
        "Tongue inspection; rule-based extraction from four-diagnosis text",
    ),
    (
        "TCM_Tongue_dark_purple",
        "舌紫暗、舌暗紫、舌暗红、瘀斑、瘀点等",
        "dark-purple tongue; dusky-red tongue; ecchymosis/petechiae; etc.",
        "Tongue inspection; rule-based extraction from four-diagnosis text",
    ),
    (
        "TCM_Tooth_mark",
        "齿痕舌、齿痕、齿印、边有齿痕等",
        "tooth-marked tongue; dental impressions on tongue edge; etc.",
        "Tongue inspection; rule-based extraction from four-diagnosis text",
    ),
    (
        "TCM_Fissure",
        "有裂纹、裂纹等",
        "fissured tongue; tongue cracks; etc.",
        "Tongue inspection; rule-based extraction from four-diagnosis text",
    ),
    (
        "TCM_Coat_yellow",
        "苔黄、黄苔、苔黄腻、苔薄黄等",
        "yellow coating; thin yellow coating; yellow greasy coating; etc.",
        "Tongue inspection; rule-based extraction from four-diagnosis text",
    ),
    (
        "TCM_Coat_white",
        "苔白、苔薄白、苔白腻、苔白厚等",
        "white coating; thin white coating; white greasy coating; etc.",
        "Tongue inspection; rule-based extraction from four-diagnosis text",
    ),
    (
        "TCM_Coat_greasy_thick",
        "苔腻、白腻、黄腻、苔厚、苔白厚腻等",
        "greasy coating; thick coating; yellow greasy coating; etc.",
        "Tongue inspection; rule-based extraction from four-diagnosis text",
    ),
    (
        "TCM_Coat_dry_scant",
        "苔少、少苔、无苔、少津、剥脱等",
        "scanty coating; peeled coating; dry coating; etc.",
        "Tongue inspection; rule-based extraction from four-diagnosis text",
    ),
    (
        "TCM_Pulse_string",
        "脉弦、脉弦细、脉弦滑、脉弦数等",
        "string-like pulse; wiry-thin pulse; wiry-slippery pulse; etc.",
        "Pulse diagnosis; rule-based extraction from four-diagnosis text",
    ),
    (
        "TCM_Pulse_slip_rapid",
        "脉滑、脉滑数、脉数、脉洪数等",
        "slippery pulse; slippery-rapid pulse; rapid pulse; etc.",
        "Pulse diagnosis; rule-based extraction from four-diagnosis text",
    ),
    (
        "TCM_Pulse_thin_sink",
        "脉细、脉沉、脉沉细、脉弱等",
        "thready pulse; deep pulse; deep-thready pulse; weak pulse; etc.",
        "Pulse diagnosis; rule-based extraction from four-diagnosis text",
    ),
    (
        "TCM_Pulse_ru_hua",
        "脉濡、脉濡缓、脉缓等",
        "soggy pulse; soft-slow pulse; moderate pulse; etc.",
        "Pulse diagnosis; rule-based extraction from four-diagnosis text",
    ),
    (
        "TCM_Syn_damp_heat",
        "湿热内蕴、湿热下注、湿热证、湿热瘀阻等",
        "damp-heat internal accumulation; damp-heat pouring downward; damp-heat syndrome; etc.",
        "Syndrome text; retained in Stage 1/2 locked TCM-integrated models",
    ),
    (
        "TCM_Syn_qi_blood_def",
        "气血不足、气血两虚、脾胃虚弱、脾气虚弱等",
        "qi-blood deficiency; spleen-stomach weakness; spleen qi deficiency; etc.",
        "Syndrome text; retained in Stage 1 locked TCM-integrated model",
    ),
    (
        "TCM_Syn_qi_stagnation_blood_stasis",
        "气滞血瘀、肝气郁结、肝郁脾虚、瘀阻肠络等",
        "qi stagnation and blood stasis; liver qi constraint; blood stasis in intestinal collaterals; etc.",
        "Syndrome text; retained in Stage 1 locked TCM-integrated model",
    ),
    (
        "TCM_Syn_spleen_kidney_yang",
        "脾肾阳虚、阳虚、寒湿内停、温煦失司等",
        "spleen-kidney yang deficiency; yang deficiency; cold-damp retention; etc.",
        "Syndrome text; retained in Stage 1/2 locked TCM-integrated models",
    ),
    (
        "TCM_Syn_yin_def_fire",
        "阴虚、阴虚内热、阴虚火旺、津亏等",
        "yin deficiency; yin deficiency with internal heat; fluid depletion; etc.",
        "Syndrome text",
    ),
    (
        "TCM_Syn_toxic_heat",
        "热毒炽盛、热毒、瘀毒、湿热毒等",
        "toxic heat exuberance; heat toxin; stasis toxin; etc.",
        "Syndrome text; retained in Stage 2 locked TCM-integrated model",
    ),
    (
        "TCM_Nature_shi",
        "病性属实、病性偏实、证属实证等",
        "excess pattern; excess-heat nature; excess syndrome; etc.",
        "Disease-nature label; rule-based extraction from syndrome text",
    ),
    (
        "TCM_Nature_xu",
        "病性属虚、病性属虚寒、证属虚证等",
        "deficiency pattern; deficiency-cold nature; deficiency syndrome; etc.",
        "Disease-nature label; rule-based extraction from syndrome text",
    ),
    (
        "TCM_Nature_mix",
        "虚实夹杂、本虚标实、虚中夹实等",
        "mixed excess-deficiency; root deficiency with tip excess; etc.",
        "Disease-nature label; retained in Stage 1 locked TCM-integrated model",
    ),
]


def row(var, coding, trig_zh=NA, trig_en=NA, note=""):
    return {
        "Variable_name": var,
        "Chinese_name": ZH[var],
        "English_name_Table_S1": EN[var],
        "Coding": coding,
        "Main_trigger_words_ZH": trig_zh,
        "Main_trigger_words_EN": trig_en,
        "Notes": note if note else NA,
    }


rows = []
rows.append(row("Age", CONT, note="Demographics; continuous"))
rows.append(row("Sex", BIN_SEX, note="Demographics; binary"))
for v in BINARY:
    rows.append(row(v, BIN_YN, note="Symptoms and signs; binary"))
for v in ORDINAL:
    rows.append(row(v, ORD[v], note="Symptoms and signs; ordinal"))
for v in LABS:
    rows.append(row(v, CONT, note="Laboratory test; continuous"))
for var, zh_t, en_t, note in TCM_ROWS:
    rows.append(row(var, BIN_TCM, zh_t, en_t, note))

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"[OK] {OUT}  n={len(df)}")
print("columns:", list(df.columns))
print(df.head(2).to_string(index=False))
print("...")
print(df.tail(2).to_string(index=False))
