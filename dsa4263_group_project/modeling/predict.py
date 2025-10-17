"""Inference utilities."""

from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd

from .. import features


def predict_from_csv(model_path: Path, data_path: Path, id_columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Run predictions with a persisted model and return a DataFrame of scores."""

    model = joblib.load(model_path)
    frame = pd.read_csv(data_path)
    feature_frame = features.add_default_features(frame.copy())

    predictions = model.predict_proba(feature_frame)[:, 1]

    ids = frame[list(id_columns)] if id_columns else pd.DataFrame()
    result = ids.copy() if not ids.empty else pd.DataFrame(index=frame.index)
    result["fraud_probability"] = predictions
    return result.reset_index(drop=True)
