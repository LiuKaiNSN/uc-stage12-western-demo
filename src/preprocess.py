from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def read_feature_list(path) -> List[str]:
    text = path.read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def build_feature_matrix(df: pd.DataFrame, final_feats: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Align columns to training feature order; missing columns become NaN for imputer."""
    missing = [c for c in final_feats if c not in df.columns]
    X = df.reindex(columns=final_feats)
    X = X.apply(pd.to_numeric, errors="coerce")
    return X, missing


def dict_to_feature_row(
    features: List[str],
    values: Dict[str, Optional[float]],
) -> pd.DataFrame:
    row: Dict[str, float] = {}
    for name in features:
        v = values.get(name)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            row[name] = np.nan
        else:
            row[name] = float(v)
    return pd.DataFrame([row])
