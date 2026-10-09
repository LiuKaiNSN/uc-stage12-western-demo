from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from .paths import resolve_project_path
from .preprocess import read_feature_list


@dataclass(frozen=True)
class StageArtifacts:
    key: str
    run_dir: Path
    pipeline: Pipeline
    feature_names: List[str]
    threshold: float
    operating_point_meta: Dict[str, Any]
    metrics: Dict[str, Any]
    stage_cfg: Dict[str, Any]


@dataclass(frozen=True)
class MulticlassStageArtifacts:
    run_dir: Path
    pipeline: Pipeline
    feature_names: List[str]
    class_names: List[str]
    e3_class_index: int
    e3_threshold: float
    operating_point_meta: Dict[str, Any]
    metrics: Dict[str, Any]
    model_cfg: Dict[str, Any]


def _load_operating_point(path: Path) -> tuple[float, Dict[str, Any]]:
    df = pd.read_csv(path)
    if "threshold" not in df.columns:
        raise ValueError(f"operating_point.csv missing 'threshold': {path}")
    row = df.iloc[0]
    meta = {
        k: (None if pd.isna(v) else (float(v) if isinstance(v, (int, float)) else v))
        for k, v in row.items()
    }
    return float(row["threshold"]), meta


def _load_binary_external_metrics(report_path: Path) -> Dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    overall = report.get("overall", {})
    metrics = overall.get("metrics", {})
    pairwise = overall.get("pairwise_binary") or []
    spec = pairwise[0].get("specificity") if pairwise else None
    return {
        "n_external": overall.get("n_samples"),
        "auc": metrics.get("auc"),
        "auprc": metrics.get("auprc"),
        "f1": metrics.get("f1"),
        "class_names": report.get("class_names"),
        "specificity_reported": spec,
    }


def _load_multiclass_external_metrics(report_path: Path) -> Dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    overall = report.get("overall", {})
    metrics = overall.get("metrics", {})
    return {
        "n_external": overall.get("n_samples"),
        "macro_auc_ovr": metrics.get("macro_auc_ovr"),
        "balanced_acc": metrics.get("balanced_acc"),
        "macro_f1": metrics.get("macro_f1"),
        "weighted_f1": metrics.get("weighted_f1"),
        "acc": metrics.get("acc"),
        "class_names": report.get("class_names"),
    }


def _load_pipeline(model_path: Path) -> Pipeline:
    obj = joblib.load(model_path)
    if isinstance(obj, dict):
        raise TypeError(
            f"{model_path} is a deep-learning bundle. "
            "This demo expects sklearn Pipeline."
        )
    if not isinstance(obj, Pipeline):
        raise TypeError(f"Expected sklearn Pipeline, got {type(obj).__name__}")
    return obj


def _pipeline_class_names(pipe: Pipeline, run_dir: Path) -> List[str]:
    mapping_path = run_dir / "class_mapping.json"
    if mapping_path.is_file():
        data = json.loads(mapping_path.read_text(encoding="utf-8"))
        return [str(c) for c in data["class_names"]]
    classes = getattr(pipe, "classes_", None)
    if classes is not None:
        return [str(c) for c in classes]
    raise ValueError(f"No class names found for {run_dir}")


def load_stage_artifacts(cfg: dict, stage_key: str) -> StageArtifacts:
    stage_cfg = cfg["stages"][stage_key]
    run_dir = resolve_project_path(cfg, stage_cfg["run_dir"])
    model_path = run_dir / stage_cfg["model_file"]
    feats_path = run_dir / stage_cfg["features_file"]
    op_path = resolve_project_path(cfg, stage_cfg["operating_point"])
    report_path = resolve_project_path(cfg, stage_cfg["external_report"])

    for p in (model_path, feats_path, op_path, report_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing artifact: {p}")

    threshold, op_meta = _load_operating_point(op_path)
    return StageArtifacts(
        key=stage_key,
        run_dir=run_dir,
        pipeline=_load_pipeline(model_path),
        feature_names=read_feature_list(feats_path),
        threshold=threshold,
        operating_point_meta=op_meta,
        metrics=_load_binary_external_metrics(report_path),
        stage_cfg=stage_cfg,
    )


def load_multiclass_artifacts(cfg: dict) -> MulticlassStageArtifacts:
    model_cfg = cfg["model"]
    run_dir = resolve_project_path(cfg, model_cfg["run_dir"])
    model_path = run_dir / model_cfg["model_file"]
    feats_path = run_dir / model_cfg["features_file"]
    op_path = resolve_project_path(cfg, model_cfg["operating_point"])
    report_path = resolve_project_path(cfg, model_cfg["external_report"])

    for p in (model_path, feats_path, op_path, report_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing artifact: {p}")

    pipe = _load_pipeline(model_path)
    class_names = _pipeline_class_names(pipe, run_dir)
    e3_label = str(model_cfg.get("e3_class_name", "3"))
    if e3_label not in class_names:
        raise ValueError(f"E3 class {e3_label!r} not in {class_names}")
    e3_index = class_names.index(e3_label)

    threshold, op_meta = _load_operating_point(op_path)
    return MulticlassStageArtifacts(
        run_dir=run_dir,
        pipeline=pipe,
        feature_names=read_feature_list(feats_path),
        class_names=class_names,
        e3_class_index=e3_index,
        e3_threshold=threshold,
        operating_point_meta=op_meta,
        metrics=_load_multiclass_external_metrics(report_path),
        model_cfg=model_cfg,
    )


def _validate_binary_stage_paths(cfg: dict, stage_key: str) -> None:
    """Check artifact files exist without joblib-loading models (saves Cloud RAM)."""
    stage_cfg = cfg["stages"][stage_key]
    run_dir = resolve_project_path(cfg, stage_cfg["run_dir"])
    paths = (
        run_dir / stage_cfg["model_file"],
        run_dir / stage_cfg["features_file"],
        resolve_project_path(cfg, stage_cfg["operating_point"]),
        resolve_project_path(cfg, stage_cfg["external_report"]),
    )
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing artifact: {p}")


def validate_config(cfg: dict) -> List[str]:
    profile = str(cfg.get("profile_id", ""))
    errors: List[str] = []
    if profile in ("stage3_western", "stage3_tcm"):
        try:
            load_multiclass_artifacts(cfg)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"model: {exc}")
        return errors

    # Path checks only — do not load Stage1+Stage2 pipelines at startup.
    for stage_key in ("stage1", "stage2"):
        if stage_key not in cfg.get("stages", {}):
            continue
        try:
            _validate_binary_stage_paths(cfg, stage_key)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{stage_key}: {exc}")
    return errors
