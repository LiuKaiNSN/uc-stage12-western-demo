# -*- coding: utf-8 -*-
"""English display labels for interpretability plots."""

from __future__ import annotations

from typing import Dict, List

STAGE1_LABELS: Dict[str, str] = {
    "Age": "Age",
    "ALT": "Alanine aminotransferase",
    "Alb": "Albumin",
    "WBC": "White blood cell count",
    "BianXue": "Stool blood severity",
    "BianZhi": "Stool consistency",
    "CRP": "C-reactive protein",
    "Fe": "Serum iron",
    "HCT": "Hematocrit",
    "LinBaJiShu": "Lymphocyte count",
    "RBC": "Red blood cell count",
    "TP": "Total protein",
    "Unnamed: 40": "Sodium",
    "Glob": "Globulin",
    "K": "Potassium",
    "NianYe": "Visible stool mucus",
    "PaiBianKunn": "Difficult defecation",
    "PCT": "Procalcitonin",
}

STAGE3_LABELS: Dict[str, str] = {
    "Age": "Age",
    "Alb": "Albumin",
    "BianXue": "Stool blood severity",
    "BianZhi": "Stool consistency",
    "Cha": "Physical examination findings",
    "CRP": "C-reactive protein",
    "FC": "Fecal calprotectin",
    "FaLi": "Fatigue",
    "Fe": "Serum iron",
    "FIB": "Fibrinogen",
    "NianYe": "Visible stool mucus",
    "PaiBianPinLv": "Bowel movement frequency",
    "XiaoShou": "Weight loss",
}

STAGE3_CLASS_DISPLAY: Dict[str, str] = {
    "1": "Limited (E1)",
    "2": "Intermediate (E2)",
    "3": "Extensive (E3)",
}

STAGE2_LABELS: Dict[str, str] = {
    "ALT": "Alanine aminotransferase",
    "Age": "Age",
    "BianZhi": "Stool consistency",
    "CRP": "C-reactive protein",
    "FC": "Fecal calprotectin",
    "LiJiHouZhong": "Tenesmus",
    "LinBaJiShu": "Lymphocyte count",
    "NianYe": "Visible stool mucus",
    "PCT": "Procalcitonin",
    "PaiBianKunn": "Difficult defecation",
    "PaiBianPinLv": "Bowel movement frequency",
    "ShiSuanBaiFen": "Eosinophil percentage",
    "XiaoShou": "Weight loss",
    "ESR": "Erythrocyte sedimentation rate",
}

# Union fallback: any label defined in any stage resolves even if stage dict omits it.
_ALL_FEATURE_LABELS: Dict[str, str] = {}
for _table in (STAGE1_LABELS, STAGE2_LABELS, STAGE3_LABELS):
    _ALL_FEATURE_LABELS.update(_table)


def display_names(stage_label: str, feature_names: List[str]) -> List[str]:
    primary = {
        "stage1": STAGE1_LABELS,
        "stage2": STAGE2_LABELS,
        "stage3": STAGE3_LABELS,
    }.get(stage_label.lower(), {})
    return [primary.get(f, _ALL_FEATURE_LABELS.get(f, f)) for f in feature_names]
