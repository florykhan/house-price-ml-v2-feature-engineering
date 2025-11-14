from __future__ import annotations

from dataclasses import asdict
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import Config


def load_raw_data(config: Config) -> pd.DataFrame:
    """
    Load the raw housing CSV into a pandas DataFrame.
    Assumes the file is located at config.data_path.
    """
    df = pd.read_csv(config.data_path)
    return df


def train_test_split_data(
    df: pd.DataFrame,
    target_col: str,
    config: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the DataFrame into train/test sets.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    return X_train, X_test, y_train, y_test

