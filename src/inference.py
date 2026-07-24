from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from .artifacts import MulticlassStageArtifacts, StageArtifacts, load_multiclass_artifacts, load_stage_artifacts
from .preprocess import build_feature_matrix, dict_to_feature_row


@dataclass
class PredictionResult:
    stage_key: str
    probability: float
    threshold: float
    refer: bool
    probability_label: str
    positive_label: str
    refer_message: str
    no_refer_message: str
    missing_features: list

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_key": self.stage_key,
            "probability": self.probability,
            "threshold": self.threshold,
            "refer": self.refer,
            "probability_label": self.probability_label,
            "positive_label": self.positive_label,
            "refer_message": self.refer_message,
            "no_refer_message": self.no_refer_message,
            "missing_features": self.missing_features,
        }


@dataclass
class MulticlassPredictionResult:
    probabilities: Dict[str, float]
    predicted_class: str
    predicted_label_zh: str
    predicted_label_en: str
    e3_probability: float
    e3_refer: bool
    e3_refer_message: str
    e3_no_refer_message: str
    missing_features: list

    def to_dict(self) -> Dict[str, Any]:
        return {
            "probabilities": self.probabilities,
            "predicted_class": self.predicted_class,
            "predicted_label_zh": self.predicted_label_zh,
            "predicted_label_en": self.predicted_label_en,
            "e3_probability": self.e3_probability,
            "e3_refer": self.e3_refer,
            "missing_features": self.missing_features,
        }


def predict_stage(artifacts: StageArtifacts, feature_values: Dict[str, Optional[float]]) -> PredictionResult:
    X = dict_to_feature_row(artifacts.feature_names, feature_values)
    X_aligned, missing = build_feature_matrix(X, artifacts.feature_names)
    proba = artifacts.pipeline.predict_proba(X_aligned)
    if proba.shape[1] != 2:
        raise ValueError(f"Expected binary predict_proba, got shape {proba.shape}")
    p = float(proba[0, 1])
    refer = p >= artifacts.threshold
    sc = artifacts.stage_cfg
    return PredictionResult(
        stage_key=artifacts.key,
        probability=p,
        threshold=artifacts.threshold,
        refer=refer,
        probability_label=sc["probability_label"],
        positive_label=sc["positive_label"],
        refer_message=sc["refer_message"],
        no_refer_message=sc["no_refer_message"],
        missing_features=missing,
    )


def predict_multiclass(
    artifacts: MulticlassStageArtifacts,
    feature_values: Dict[str, Optional[float]],
) -> MulticlassPredictionResult:
    X = dict_to_feature_row(artifacts.feature_names, feature_values)
    X_aligned, missing = build_feature_matrix(X, artifacts.feature_names)
    proba = artifacts.pipeline.predict_proba(X_aligned)
    if proba.shape[1] != len(artifacts.class_names):
        raise ValueError(
            f"Expected {len(artifacts.class_names)}-class proba, got {proba.shape}"
        )

    prob_map = {
        cls: float(proba[0, i]) for i, cls in enumerate(artifacts.class_names)
    }
    pred_idx = int(proba.argmax(axis=1)[0])
    pred_class = artifacts.class_names[pred_idx]
    class_display = artifacts.model_cfg.get("class_display", {})
    disp = class_display.get(pred_class, {})
    e3_prob = float(proba[0, artifacts.e3_class_index])
    e3_refer = e3_prob >= artifacts.e3_threshold

    return MulticlassPredictionResult(
        probabilities=prob_map,
        predicted_class=pred_class,
        predicted_label_zh=str(disp.get("zh", f"Class {pred_class}")),
        predicted_label_en=str(disp.get("en", f"Class {pred_class}")),
        e3_probability=e3_prob,
        e3_refer=e3_refer,
        e3_refer_message=str(artifacts.model_cfg.get("e3_refer_message", "")),
        e3_no_refer_message=str(artifacts.model_cfg.get("e3_no_refer_message", "")),
        missing_features=missing,
    )


def predict_from_dataframe(artifacts: StageArtifacts, df: pd.DataFrame) -> PredictionResult:
    if len(df) != 1:
        raise ValueError("Expected a single-row DataFrame for smoke testing.")
    values = {c: df.iloc[0].get(c) for c in artifacts.feature_names}
    return predict_stage(artifacts, values)


def load_both_stages(cfg: dict) -> Dict[str, StageArtifacts]:
    return {
        "stage1": load_stage_artifacts(cfg, "stage1"),
        "stage2": load_stage_artifacts(cfg, "stage2"),
    }


def load_stage3_model(cfg: dict) -> MulticlassStageArtifacts:
    return load_multiclass_artifacts(cfg)
