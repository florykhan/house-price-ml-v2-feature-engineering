from __future__ import annotations

from typing import Tuple, Dict

import numpy as np
import pandas as pd


def compute_standardization_params(X: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Compute mean and std (standard deviation) for each feature.
    """
    means = X.mean()
    stds = X.std(ddof=0)

    return {
        "means": means,
        "stds": stds,
        "columns": X.columns.to_list(),
    }


def apply_standardization(
    X: pd.DataFrame,
    params: Dict[str, pd.Series],
) -> pd.DataFrame:
    """
    Apply standardization using precomputed means and stds.
    """
    means = params["means"]
    stds = params["stds"].replace(0, 1.0) # avoid divide-by-zero

    X_scaled = (X - means) / stds

    return pd.DataFrame(X_scaled, columns=params["columns"], index=X.index)
