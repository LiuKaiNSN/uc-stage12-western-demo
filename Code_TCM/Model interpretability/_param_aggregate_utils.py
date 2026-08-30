# -*- coding: utf-8 -*-
"""聚合各折超参时的稳健 mode/median 工具（处理 None/NaN 导致 pandas.mode 为空）。"""

from __future__ import annotations

from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd

_NONE_SENTINEL = "__NONE__"


def _normalize_for_mode(value: Any) -> Any:
    if value is None:
        return _NONE_SENTINEL
    if isinstance(value, float) and np.isnan(value):
        return _NONE_SENTINEL
    if isinstance(value, str) and value.strip().lower() in ("none", "nan", ""):
        return _NONE_SENTINEL
    return value


def series_mode_value(series: pd.Series, *, default: Any = None) -> Any:
    """
    返回列的众数；若全为 None/NaN 则返回 default。
    避免 ``series.mode().iloc[0]`` 在空 mode 时 IndexError。
    """
    if series.empty:
        return default
    normalized = [_normalize_for_mode(v) for v in series.tolist()]
    counts = pd.Series(normalized).value_counts()
    if counts.empty:
        return default
    top = counts.index[0]
    if top == _NONE_SENTINEL:
        return None
    return top


def aggregate_rf_max_depth(col: pd.Series) -> Optional[int]:
    top = series_mode_value(col, default=None)
    if top is None:
        return None
    return int(float(top))


def aggregate_rf_max_features(col: pd.Series, *, default: str = "sqrt") -> Union[str, float]:
    top = series_mode_value(col, default=default)
    if top is None:
        return default
    try:
        return float(top)
    except (TypeError, ValueError):
        return str(top)
