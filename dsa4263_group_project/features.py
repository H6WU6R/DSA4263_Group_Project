"""Feature engineering utilities."""

import pandas as pd


def add_default_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *frame* with placeholder engineered features."""

    engineered = frame.copy()
    engineered["feature_sum"] = engineered.select_dtypes("number").sum(axis=1)
    return engineered
